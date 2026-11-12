"""D-82: could the trained network narrow the search's choices on big problems (E-81)?

E-81: on s4-s5 the search picks only 0-2 moves of plans 22-174 steps long, because each leaf
needs a finish of minutes and every step has dozens of legal moves. The value head cannot
rank sibling moves (E-46, E-47, E-50), but the policy head might still say which few moves
are worth a finish. If it puts the search's own winning moves near the top, the search could
try only those, and go deeper for the same time.

**Check.** On val s3-s5, run method-v3's search (`switch`, first plan, CPU budget `seconds`)
and keep its plan where it beats A3's best run (`records`, fluid). Each move the tree chose
(the plan minus the finisher's tail) is a good move. At each such position, the network
(`agent`, the policy head over the env's legal moves, planner hints as in training) ranks all
legal moves; the record holds the move's rank and the number of legal moves. s3 is the
contrast (there the tree goes deep).

Reading (D-82, fixed before the run), on s4-s5: the network can narrow the choices if the
search's moves sit in its top 10% of legal moves at least half the time (a flat prior: 10%),
and in its top 5 on at least half the positions with 50 or more legal moves. Otherwise it
cannot pick the moves the search finds, and the network stays out of the method.

    OMP_NUM_THREADS=1 uv run python scripts/prior_check.py workers=24 \\
        out=outputs/bench/prior.jsonl
    uv run python scripts/prior_check.py summary=outputs/bench/prior.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.analysis.evaluate import build_env
from meda.core.actions import Action
from meda.core.problem import Instance
from meda.generators.benchmark_io import BenchmarkSet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import goal_check as gc  # noqa: E402
import search_check as sc  # noqa: E402

SETS = ("val_s3_v2", "val_s4_v2", "val_s5_v2")
AGENT = "outputs/expert/round2/copy_v/imitation.pt"  # round 2's network (E-35, E-39, E-51)
_NET: dict[str, Any] = {}


def best_runs(records: Path) -> dict[str, float]:
    """A3's best of the 4 runs per problem (fluid), the reference the search must beat."""
    out = {}
    for line in records.read_text().splitlines():
        r = json.loads(line)
        if r.get("goal") == "fluid" and r["set"] in SETS and r.get("best_planner") is not None:
            out[r["id"]] = float(r["best_planner"])
    return out


def policy(agent: Any, obs: dict[str, Any], n: int) -> np.ndarray:
    """The policy head's probabilities over the `n` legal moves."""
    import torch

    from meda.agents.ppo.policy import batch_obs, masked_distribution

    with torch.no_grad():
        t = batch_obs({k: v[None] for k, v in obs.items()}, agent.device)
        logits, _ = agent.net(t)
        p = masked_distribution(logits, t["action_mask"]).probs[0].numpy()
    return np.asarray(p[:n], dtype=float)


def rank_of(p: np.ndarray, i: int) -> int:
    """1-based rank of move `i` (ties count against it: every move at least as likely)."""
    return int((p >= p[i]).sum())


def ranks(inst: Instance, prefix: tuple[Action, ...], agent: Any) -> list[dict[str, Any]]:
    """For each move of `prefix`, its rank under the network among the legal moves there."""
    cfg = {"costs": sc._load(sc.COSTS), "env": sc._load("env/sample_prep.yaml")}
    cfg["env"]["hint_source"] = "planner"
    cfg["env"]["T_pad"] = max(cfg["env"]["T_pad"], len(inst.targets))
    env = build_env(cfg, [inst])
    obs, _ = env.reset(options={"instance": inst})
    out = []
    for depth, action in enumerate(prefix):
        legal = env.legal
        i = legal.index(action)
        p = policy(agent, obs, len(legal))
        out.append({"depth": depth, "legal": len(legal), "rank": rank_of(p, i)})
        obs, *_ = env.step(i)
    return out


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    gc.use_goal("fluid")
    inst = Instance.from_json(task["instance"])
    stask = {
        "set": task["set"],
        "id": task["id"],
        "cell": "",
        "instance": task["instance"],
        "optimum": None,
        "planner": task["best_run"],
        "mrcm": None,
        "greedy": None,
    }
    sopts = {
        "agent": None,
        "seconds": opts["seconds"],
        "hints": "planner",
        "reward_scale": 0.01,
        "keep_plan": True,
        "first_plan": True,
    }
    rec = sc.run_task(stask, "uniform", 256, sopts, "switch")
    out = {k: v for k, v in task.items() if k != "instance"}
    out.update(search_cost=rec["search_cost"], leaves=rec["leaves"], moves=[])
    plan = rec["plan"]
    if plan is None or rec["search_cost"] is None:
        return out
    out["wins"] = rec["search_cost"] < task["best_run"] - 1e-9
    if not out["wins"]:
        return out
    if opts["agent"] not in _NET:
        import torch

        from meda.agents.ppo.policy import PPOAgent

        torch.set_num_threads(1)
        _NET[opts["agent"]] = PPOAgent.load(opts["agent"])
    prefix = plan[: len(plan) - rec["planner_tail"]]
    out["moves"] = ranks(inst, prefix, _NET[opts["agent"]])
    return out


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


def summarise(recs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| set | wins / problems | tree moves | legal moves (median) | median rank | "
        "in top 10% | in top 5 (50+ legal) |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        groups[r["set"]].append(r)
        if r["set"] != "val_s3_v2":
            groups["s4-s5"].append(r)
    verdict = ""
    for name in [*sorted(g for g in groups if g != "s4-s5"), "s4-s5"]:
        rs = groups[name]
        ms = [m for r in rs for m in r.get("moves") or []]
        wins = sum(bool(r.get("wins")) for r in rs)
        top10 = sum(m["rank"] <= max(1, 0.1 * m["legal"]) for m in ms)
        big = [m for m in ms if m["legal"] >= 50]
        top5 = sum(m["rank"] <= 5 for m in big)
        leg = f"{np.median([m['legal'] for m in ms]):.0f}" if ms else "-"
        rk = f"{np.median([m['rank'] for m in ms]):.0f}" if ms else "-"
        lines.append(
            f"| {name} | {wins} / {len(rs)} | {len(ms)} | {leg} | {rk} | {top10} of {len(ms)} | "
            f"{top5} of {len(big)} |"
        )
        if name == "s4-s5":
            holds = len(ms) > 0 and top10 >= len(ms) / 2 and (not big or top5 >= len(big) / 2)
            verdict = (
                f"Reading (D-82), s4-s5: the search's moves are in the network's top 10% on "
                f"{top10} of {len(ms)}, in its top 5 on {top5} of {len(big)} positions with 50+ "
                "legal moves: "
                + (
                    "the network can narrow the choices"
                    if holds
                    else "the network cannot pick the search's moves"
                )
            )
    return [*lines, "", verdict]


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        recs = [json.loads(x) for x in Path(args["summary"]).read_text().splitlines()]
        print("\n".join(summarise(recs)))
        return
    ref = best_runs(Path(args.get("records", "outputs/bench/pilot_a3.jsonl")))
    opts = {"seconds": float(args.get("seconds", 300)), "agent": args.get("agent", AGENT)}
    todo = []
    for s in args.get("sets", ",".join(SETS)).split(","):
        for i in BenchmarkSet.read(s, ROOT / "benchmarks").instances:
            if i.instance_id in ref:
                task = {"set": s, "id": i.instance_id, "instance": i.to_json()}
                todo.append((task | {"best_run": ref[i.instance_id]}, opts))
    out = Path(args.get("out", "outputs/bench/prior.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problems", flush=True)
    recs = []
    with out.open("w") as f:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for i, r in enumerate(results, 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            recs.append(r)
            print(f"{i}/{len(todo)} {r['id']} wins {r.get('wins')} moves {r['moves']}", flush=True)
    print("\n".join(summarise(recs)))


if __name__ == "__main__":
    main(sys.argv[1:])
