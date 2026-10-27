"""Hint coverage (plan 3.2; paper log D-34, D-35): how often the candidates carry a hint.

The hint map is fixed per target (single-target recipes and pure-fluid chains,
`solvers/hints.py`); it ignores what is on the chip and reuse across targets. This measures,
from the hint columns of the observation the network actually sees:

- **Teacher** (the sharing pass's plan played through the env): how many of the teacher's mixes
  carry a recipe or chain hint, how many recipe hints only say "4 or more", and what share of
  all mix candidates is hinted. Tells 3.3 where hints are missing (stages 3-5).
- **Agent** (greedy, `agent=<agent.pt>`): at every step, whether any mix candidate is hinted,
  whether the agent's own mix was, the probability of its move and, with `planner=1`, of the
  planner's next move from the same state (`finish_from`'s first action) and whether that move
  is hinted. Split by solved / failed episodes and by quarter of the episode.

Reading, fixed in advance (D-34), on the planner's next move (the move that makes progress
from the state the agent is in; counted where that move is a mix, since only mixes carry
hints): **blind off the map** if, in the last quarter of failed
episodes, it is hinted on < 50% of steps and on less than half as often as in solved episodes;
**will not commit** if, in failed episodes, it is hinted on >= 70% of steps but gets a median
probability < 0.3. Both, or neither, are possible. ("Any mix candidate hinted" is reported but
is not a test: the chain hint marks many compositions, so it stays near 100%.)

    uv run python scripts/hint_coverage.py agent=outputs/warm/mixed_s0/checkpoints/agent.pt \
        workers=30 out=notes/hint_coverage/records.jsonl
    uv run python scripts/hint_coverage.py summary=notes/hint_coverage/records.jsonl

Arguments are key=value: `sets` (default all six val sets), `limit=N` problems per set,
`planner=0|1` (default 1; ~2 s a step on stage 4), `workers` (on the Mac keep 1).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from meda.agents.ppo.imitation import UidMap, teacher_index
from meda.analysis.evaluate import build_env
from meda.core.actions import Mix
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.envs.encoding import CHAIN_MAX, HINT_COL, OP_MIX, RECIPE_MAX
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines.share import finish_from, share_plan

ROOT = Path(__file__).resolve().parents[1]
SETS = ("val_s1_v2", "val_s2_v2", "val_s3_v2", "val_s4_v2", "val_s5_v2", "heldout_combo_v2")
_load = lambda rel: OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / rel))  # noqa: E731
CFG = {"costs": _load("costs/geometric.yaml"), "env": _load("env/sample_prep.yaml")}
WEIGHTS = CostWeights.from_dict(CFG["costs"])
CHAIN_COL = HINT_COL + RECIPE_MAX + 2
# Fixed in advance (module docstring).
BLIND_RATIO, COMMIT_HINTED, COMMIT_P = 0.5, 0.7, 0.3


def hints(obs: dict[str, Any], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the first `n` candidates: is a mix, recipe hint (0..RECIPE_MAX, -1 = none or not a
    mix; RECIPE_MAX means "that many or more"), chain hint (0..CHAIN_MAX, -1 = none)."""
    cand = obs["candidates"][:n]
    is_mix = cand[:, OP_MIX] == 1.0
    recipe = cand[:, HINT_COL : HINT_COL + RECIPE_MAX + 2].argmax(1)
    chain = cand[:, CHAIN_COL : CHAIN_COL + CHAIN_MAX + 2].argmax(1)
    recipe = np.where(is_mix & (recipe <= RECIPE_MAX), recipe, -1)
    chain = np.where(is_mix & (chain <= CHAIN_MAX), chain, -1)
    return is_mix, recipe, chain


def step_row(obs: dict[str, Any], n: int, index: int) -> dict[str, Any]:
    is_mix, recipe, chain = hints(obs, n)
    hinted = (recipe >= 0) | (chain >= 0)
    return {
        "mixes": int(is_mix.sum()),
        "hinted": int(hinted.sum()),
        "move_mix": bool(is_mix[index]),
        "move_hinted": bool(hinted[index]),
        "move_recipe": int(recipe[index]),
    }


def teacher_trace(inst: Instance) -> list[dict[str, Any]]:
    """The sharing pass's plan played through the env (as demonstrations are), per step."""
    env = build_env(CFG, [inst])
    plan = share_plan(inst, WEIGHTS)
    assert plan is not None, inst.instance_id
    obs, _ = env.reset(options={"instance": inst})
    uids, rows = UidMap(), []
    for action in plan:
        meant = uids.action(action)
        index = teacher_index(env.legal, meant, inst.instance_id, env.state)
        uids.follow(meant, env.legal[index], env.state)
        rows.append(step_row(obs, len(env.legal), index))
        obs, *_ = env.step(index)
    return rows


_AGENT: dict[str, Any] = {}


def agent_trace(inst: Instance, path: str, planner: bool) -> tuple[bool, list[dict[str, Any]]]:
    """The greedy agent's episode, per step; with `planner`, the planner's next move too."""
    import torch

    from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution

    if path not in _AGENT:
        torch.set_num_threads(1)
        _AGENT[path] = PPOAgent.load(path)
    agent = _AGENT[path]
    env = build_env(CFG, [inst])
    obs, _ = env.reset(options={"instance": inst})
    rows = []
    while True:
        with torch.no_grad():
            t = batch_obs({k: v[None] for k, v in obs.items()}, agent.device)
            logits, _ = agent.net(t)
            p = masked_distribution(logits, t["action_mask"]).probs[0].numpy()
        index = int(p.argmax())
        row = step_row(obs, len(env.legal), index)
        row["p_move"] = float(p[index])
        if planner:
            found = finish_from(env.state, inst, WEIGHTS)
            try:
                nxt = None
                if found is not None and found[0]:
                    nxt = teacher_index(env.legal, found[0][0], inst.instance_id, env.state)
            except ValueError:  # recorded, not hidden: the summary counts these
                row["planner_unmatched"] = True
            if nxt is not None:
                _, recipe, chain = hints(obs, len(env.legal))
                row["p_planner"] = float(p[nxt])
                row["planner_mix"] = isinstance(found[0][0], Mix)
                row["planner_hinted"] = bool(recipe[nxt] >= 0 or chain[nxt] >= 0)
                row["agent_is_planner"] = nxt == index
        rows.append(row)
        obs, _, term, trunc, info = env.step(index)
        if term or trunc:
            return bool(info["success"]), rows


def run_task(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    name, inst_json, opts = task
    inst = Instance.from_json(inst_json)
    t0 = time.perf_counter()
    rec: dict[str, Any] = {
        "set": name,
        "id": inst.instance_id,
        "cell": inst.metadata.get("cell", ""),
        "teacher": teacher_trace(inst),
    }
    if opts.get("agent"):
        rec["solved"], rec["agent"] = agent_trace(inst, opts["agent"], opts["planner"])
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


# --------------------------------------------------------------------------- summary


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _f(x: float | None, pct: bool = True) -> str:
    return "-" if x is None else (f"{100 * x:.0f}" if pct else f"{x:.2f}")


def teacher_row(rs: list[dict[str, Any]]) -> str:
    steps = [s for r in rs for s in r["teacher"]]
    mixes = [s for s in steps if s["move_mix"]]
    recipe = [s for s in mixes if s["move_recipe"] >= 0]
    far = [s for s in recipe if s["move_recipe"] == RECIPE_MAX]
    cands = [s["hinted"] / s["mixes"] for s in steps if s["mixes"]]
    return (
        f"{len(mixes)} | {_f(sum(s['move_hinted'] for s in mixes) / max(len(mixes), 1))} "
        f"| {_f(len(recipe) / max(len(mixes), 1))} | {_f(len(far) / max(len(recipe), 1))} "
        f"| {_f(_mean(cands))}"
    )


def agent_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [s for r in rs for s in r["agent"]]
    last = [s for r in rs for s in r["agent"][3 * len(r["agent"]) // 4 :]]
    first = [s for r in rs for s in r["agent"][: max(1, len(r["agent"]) // 4)]]
    any_hint = lambda ss: _mean([float(s["hinted"] > 0) for s in ss if s["mixes"]])  # noqa: E731
    p_hint = lambda ss: _mean(  # noqa: E731
        [float(s["planner_hinted"]) for s in ss if "p_planner" in s and s["planner_mix"]]
    )
    own = [s for s in steps if s["move_mix"]]
    plan = [s for s in steps if "p_planner" in s]
    return {
        "n": len(rs),
        "steps": len(steps),
        "any_first": any_hint(first),
        "any_last": any_hint(last),
        "own_hinted": _mean([float(s["move_hinted"]) for s in own]),
        "p_move": float(np.median([s["p_move"] for s in steps])) if steps else None,
        "planner_hinted": p_hint(steps),
        "planner_first": p_hint(first),
        "planner_last": p_hint(last),
        "unmatched": sum(1 for s in steps if s.get("planner_unmatched")),
        "p_planner": float(np.median([s["p_planner"] for s in plan])) if plan else None,
        "is_planner": _mean([float(s["agent_is_planner"]) for s in plan]),
    }


def agent_line(label: str, s: dict[str, Any]) -> str:
    return (
        f"| {label} | {s['n']} | {s['steps']} | {_f(s['any_first'])} / {_f(s['any_last'])} "
        f"| {_f(s['own_hinted'])} | {_f(s['p_move'], False)} "
        f"| {_f(s['planner_first'])} / {_f(s['planner_last'])} "
        f"| {_f(s['p_planner'], False)} | {_f(s['is_planner'])} | {s['unmatched']} |"
    )


def verdict(solved: dict[str, Any], failed: dict[str, Any]) -> str:
    out = []
    fl, sl = failed["planner_last"], solved["planner_last"]
    if fl is not None and fl < BLIND_RATIO and (sl is None or fl < BLIND_RATIO * sl):
        out.append("blind off the map")
    ph, pp = failed["planner_hinted"], failed["p_planner"]
    if ph is not None and pp is not None and ph >= COMMIT_HINTED and pp < COMMIT_P:
        out.append("will not commit")
    return ", ".join(out) or "neither rule met"


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        groups[(r["set"], "all")].append(r)
        groups[(r["set"], r["cell"])].append(r)
    keys = sorted(groups, key=lambda k: (k[0], k[1] != "all", k[1]))
    lines = [
        "## Teacher (sharing pass) states",
        "",
        "| set | cell | teacher mixes | hinted % | recipe-hinted % | of those '4+' % "
        "| mix candidates hinted % (mean per step) |",
        "|---" * 7 + "|",
    ]
    lines += [f"| {s} | {c} | {teacher_row(groups[(s, c)])} |" for s, c in keys]
    with_agent = [r for r in recs if "agent" in r]
    if with_agent:
        lines += [
            "",
            "## Agent episodes (greedy)",
            "",
            "any hinted % = steps with at least one hinted mix candidate (first / last quarter "
            "of each episode); p = median probability.",
            "",
            "| set / episodes | n | steps | any hinted % first / last | own mixes hinted % "
            "| p move | planner's move hinted % first / last | p planner's move "
            "| agent = planner % | planner move unmatched |",
            "|---" * 10 + "|",
        ]
        by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in with_agent:
            by_set[r["set"]].append(r)
        for name in sorted(by_set):
            rs = by_set[name]
            solved = agent_stats([r for r in rs if r["solved"]])
            failed = agent_stats([r for r in rs if not r["solved"]])
            lines.append(agent_line(f"{name} solved", solved))
            if failed["n"]:
                lines.append(agent_line(f"{name} failed", failed))
                lines.append(f"| {name} reading | {verdict(solved, failed)} |" + " |" * 8)
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    opts = {"agent": args.get("agent"), "planner": args.get("planner", "1") == "1"}
    limit = int(args["limit"]) if "limit" in args else None
    todo = []
    for name in args.get("sets", ",".join(SETS)).split(","):
        bench = BenchmarkSet.read(name)
        todo += [(name, inst.to_json(), opts) for inst in list(bench.instances)[:limit]]
    out = Path(args.get("out", "notes/hint_coverage/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    t0, done = time.perf_counter(), 0
    print(f"{len(todo)} problems, {workers} workers, agent {opts['agent']}", flush=True)
    with out.open("w") as fh:
        results = (
            map(run_task, todo) if workers == 1 else Pool(workers).imap_unordered(run_task, todo)
        )
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {done} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
