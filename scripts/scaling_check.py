"""Data scaling check (D-53, Q-124, after E-48): is the value head held back by how much it sees?

Five value targets have failed to give the network an edge on stages 4-5 (E-40-E-48). Each was
noisy (whatever one search happened to find) and came from ~1,600 problems. This check removes
the noise and varies only the amount: the label at every position is the planner's own cost to
finish from there (`finish_from`, exact for what it is, no search, no luck), and the value head
is trained on 10k, 30k, 100k and 300k such positions. The question is the least a replacement
for the planner must do: rank sibling moves as the planner ranks them.
- `part=label`: the round-3 network plays fresh stage 4-5 problems (never a validation, held-out
  or search-test problem) by sampling its own move probabilities, up to twice the planner's
  plan length (a sampled line can wander). At `groups` positions on the line played, drawn at
  random, the position, the played move and up to `kids` - 1 other legal moves are labelled
  with the planner finish (left out where it does not finish within the step limit), so each
  position's children form a sibling group; few positions per problem keep many problems.
  Writes `held.jsonl` (`held` problems, a separate draw) and `train_<n>.jsonl` for each n in
  `sizes`: the first problems of one draw up to n positions, so each file contains the smaller
  ones. Records load as copy-step targets
  (`imitation_check.py arm=visits`, no visit counts: the value head alone learns).
- `part=check`: each network in `agents` (comma-separated, smallest data first) on
  `held.jsonl`: how often the head's choice among siblings (spent + predicted finish) is the
  planner's (spent + planner finish), against the bound's choice and chance.

Reading, fixed in advance (D-53), held-out agreement with the planner's choice over the sizes in
order: rises by >= 0.05 at every step → data is the bottleneck: build bulk generation; rises by
< 0.02 from the first to the last → not data: more labels will not let the head replace the
planner on s3-s5; otherwise it helps, slowly.

    uv run python scripts/scaling_check.py part=label \\
        agent=outputs/expert/round3/copy/imitation.pt out=outputs/scaling workers=36
    uv run python scripts/scaling_check.py part=check held=outputs/scaling/held.jsonl \\
        agents=outputs/scaling/n10000/imitation.pt,...,outputs/scaling/n300000/imitation.pt

Arguments are key=value: `part`, `agent`, `agents`, `held` (problems for part=label, the file
for part=check), `sizes`, `groups` (positions per problem, default 8), `kids` (default 6),
`workers`, `threads`, `device`, `reward_scale`, `max_chunk_rows`, `out`.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution
from meda.analysis.evaluate import build_env
from meda.core.chip import ChipLimits
from meda.core.problem import Instance
from meda.generators.benchmark_io import BenchmarkSet, instance_hash
from meda.generators.stages import StageDesign
from meda.solvers.baselines.share import finish_from
from meda.solvers.exact_search import fluid_bound_heuristic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_iteration as ei  # noqa: E402
import fit_check as fc  # noqa: E402
import imitation_check as ic  # noqa: E402
import resolution_check as rc  # noqa: E402
import search_check as sc  # noqa: E402

DEFAULTS: dict[str, Any] = {
    "part": "",
    "agent": "",
    "agents": "",
    "held": "60",
    "sizes": "10000,30000,100000,300000",
    "groups": 8,
    "kids": 6,
    "workers": 8,
    "threads": 8,
    "device": "cpu",
    "reward_scale": 0.01,
    "max_chunk_rows": 32768,
    "out": "",
}
STAGES = ("s4_targets", "s5_storage")
STEP, FLAT = 0.05, 0.02  # fixed in advance (D-53)
MAX_PROBLEMS = 20000  # a draw that never reaches the largest size is an error, not a hang
BATCH = 72  # problems per round of the pool
LONGER = 2  # a line of play stops at this many times the planner's plan length


def problems(seed: int) -> Iterator[Instance]:
    """Fresh stage 4-5 problems, endlessly (seed 0 trains, 1 is held out)."""
    design = StageDesign.from_dict(sc._load("stages/track2_v1.yaml"))
    chip = ChipLimits.from_dict(sc._load(f"chip/{design.chip}.yaml"))
    held = set(BenchmarkSet.read(design.held_out_val).hashes())
    for s in design.stages:
        held |= BenchmarkSet.read(s.val).hashes()
    tests = {
        instance_hash(Instance.from_json(t["instance"])) for t in sc.tasks({"A", "B", "C"}, None)
    }
    source = design.source({s: 0.5 for s in STAGES}, chip, sc.WEIGHTS, frozenset(held | tests))
    rng = np.random.default_rng([seed, 53])
    i = 0
    while True:
        inst = source(rng)
        yield Instance(
            f"scale-{seed}-{i}",
            inst.fluids,
            inst.targets,
            inst.chip,
            inst.step_limit,
            inst.metadata,
        )
        i += 1


def probs(agent: Any, obs: dict[str, Any], n: int) -> npt.NDArray[np.float64]:
    with torch.no_grad():
        t = batch_obs({k: v[None] for k, v in obs.items()}, agent.device)
        logits, _ = agent.net(t)
        p = masked_distribution(logits, t["action_mask"]).probs[0].numpy()[:n].astype(np.float64)
    return p / p.sum()


def play(task: tuple[str, str, int, int, int]) -> dict[str, Any]:
    """One problem: the network's sampled line of play; at `groups` positions on it, drawn at
    random, the position and its children labelled with the planner finish."""
    inst_json, agent_path, groups, kids, seed = task
    inst = Instance.from_json(inst_json)
    agent = ei._agent(agent_path)
    rng = np.random.default_rng([seed, zlib.crc32(inst.instance_id.encode())])
    env = build_env(fc._cfg(), [inst])
    obs, _ = env.reset(options={"instance": inst})
    root = finish_from(env.state, inst, sc.WEIGHTS)
    longest = LONGER * len(root[0]) if root is not None else 0
    line: list[Any] = []
    snaps = []
    while len(line) < longest:
        snap = env.snapshot()
        if not snap.legal:
            break
        snaps.append(snap)
        played = int(rng.choice(len(snap.legal), p=probs(agent, obs, len(snap.legal))))
        line.append(snap.legal[played])
        obs, _, terminated, truncated, _ = env.step(played)
        if terminated or truncated:
            break
    labels: dict[tuple[Any, ...], float] = {}
    for d in sorted(rng.choice(len(snaps), min(groups, len(snaps)), replace=False)):
        env.restore(snaps[d])
        here = tuple(line[:d])
        labels[here] = rc.planner_finish(env, inst)
        if np.isnan(labels[here]):
            continue
        legal = list(snaps[d].legal)
        played = legal.index(line[d])
        others = [i for i in range(len(legal)) if i != played]
        for i in [played, *rng.choice(others, min(kids - 1, len(others)), replace=False)]:
            env.restore(snaps[d])
            env.step(int(i))
            labels[(*here, legal[i])] = rc.planner_finish(env, inst)
    samples = tuple((a, None, v) for a, v in labels.items() if not np.isnan(v))
    return {
        "id": inst.instance_id,
        "cell": inst.metadata.get("cell", ""),
        "instance": inst_json,
        "samples": ei._pack(samples),
        "positions": len(samples),
    }


def label(args: dict[str, Any]) -> None:
    started = time.perf_counter()
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted(int(x) for x in args["sizes"].split(","))
    gen = problems(1)
    held = [
        (next(gen).to_json(), args["agent"], args["groups"], args["kids"], 1)
        for _ in range(int(args["held"]))
    ]
    with Pool(args["workers"]) as pool:
        recs = pool.map(play, held, chunksize=1)
        (out / "held.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(
            f"held: {len(recs)} problems, {sum(r['positions'] for r in recs)} positions", flush=True
        )
        gen = problems(0)
        files = {n: (out / f"train_{n}.jsonl").open("w") for n in sizes}
        k = total = 0
        while total < sizes[-1]:
            if k >= MAX_PROBLEMS:
                raise RuntimeError(f"{k} problems gave {total} positions < {sizes[-1]}")
            batch = [
                (next(gen).to_json(), args["agent"], args["groups"], args["kids"], 0)
                for _ in range(BATCH)
            ]
            for rec in pool.map(play, batch, chunksize=1):
                for n, f in files.items():
                    if total < n:
                        f.write(json.dumps(rec) + "\n")
                total += rec["positions"]
                k += 1
            print(
                f"{k} problems, {total} positions  {time.perf_counter() - started:.0f}s", flush=True
            )
        for f in files.values():
            f.close()
    print(
        f"labelled {k} training problems, {total} positions; {time.perf_counter() - started:.0f}s"
    )


def replay(task: tuple[str, str, float]) -> tuple[ic.SearchTargets, list[float], list[float]]:
    """The held-out positions, the cost spent at each and the lower bound on the cost to finish."""
    inst_json, samples, reward_scale = task
    inst = Instance.from_json(inst_json)
    store = ic.SearchTargets()
    spent: list[float] = []
    bound: list[float] = []
    lower = fluid_bound_heuristic(sc.WEIGHTS)

    def at(env: Any) -> None:
        spent.append(float(env.snapshot().plan_cost))
        bound.append(float(lower(env.state, inst)))

    store.add_problem(build_env(fc._cfg(), [inst]), inst, ei._unpack(samples), reward_scale, at)
    return store, spent, bound


def metrics(rows: dict[str, npt.NDArray[Any]]) -> dict[str, Any]:
    """`rows` per position: parent (-1 at the top), spent, target (planner finish), pred and
    bound (costs to finish)."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i in np.flatnonzero(rows["parent"] >= 0):
        groups[int(rows["parent"][i])].append(int(i))
    gaps, head, bound, chance = [], [], [], []
    for kids in (g for g in groups.values() if len(g) >= 2):
        k = np.array(kids)
        total = rows["spent"][k] + rows["target"][k]
        order = np.sort(total)
        gaps.append(order[1] - order[0])
        best = np.isclose(total, order[0])
        head.append(bool(best[np.argmin(rows["spent"][k] + rows["pred"][k])]))
        bound.append(bool(best[np.argmin(rows["spent"][k] + rows["bound"][k])]))
        chance.append(best.mean())
    return {
        "positions": len(rows["parent"]),
        "groups": len(gaps),
        "target": rc._median(rows["target"]),
        "head_err": rc._median(np.abs(rows["pred"] - rows["target"])),
        "runner_gap": rc._median(gaps),
        "head_best": float(np.mean(head)) if head else None,
        "bound_best": float(np.mean(bound)) if bound else None,
        "chance_best": float(np.mean(chance)) if chance else None,
    }


def reading(curve: list[float]) -> str:
    """The rule fixed in advance (D-53): held-out agreement over the sizes, in order."""
    steps = np.diff(curve)
    shown = " → ".join(f"{x:.2f}" for x in curve)
    if len(steps) and (steps >= STEP).all():
        return f"data is the bottleneck ({shown}, +{STEP} or more at every step): build bulk labels"
    if curve[-1] - curve[0] < FLAT:
        return (
            f"not data ({shown}, under +{FLAT} overall): more labels will not replace the planner"
        )
    return f"data helps, slowly ({shown})"


def check(args: dict[str, Any]) -> None:
    started = time.perf_counter()
    torch.set_num_threads(args["threads"])
    scale = args["reward_scale"]
    lines = [json.loads(x) for x in Path(args["held"]).read_text().split("\n") if x]
    with Pool(args["workers"]) as pool:
        replayed = pool.map(replay, [(r["instance"], r["samples"], scale) for r in lines])
    print(f"replayed {len(replayed)} held-out problems in {time.perf_counter() - started:.0f}s")
    device = torch.device(args["device"])
    table = {}
    for path in args["agents"].split(","):
        net = PPOAgent.load(path, device).net
        net.eval()
        cols: dict[str, list[Any]] = defaultdict(list)
        for store, spent, bound in replayed:
            if not len(store):
                continue
            _, value = fc.predict(net, store, device, args["max_chunk_rows"])
            offset = sum(len(x) for x in cols["spent"])
            cols["parent"].append(np.array([p + offset if p >= 0 else -1 for p in store.parent]))
            cols["spent"].append(np.array(spent))
            cols["target"].append(-np.array(store.value, dtype=np.float64) / scale)
            cols["pred"].append(np.maximum(0.0, -value / scale))
            cols["bound"].append(np.array(bound))
        table[Path(path).parent.name] = metrics({k: np.concatenate(v) for k, v in cols.items()})
    names = list(next(iter(table.values())))
    print("| network | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for name, row in table.items():
        print(f"| {name} | " + " | ".join(fc._fmt(row[c]) for c in names) + " |")
    curve = [row["head_best"] for row in table.values()]
    if len(curve) >= 2 and None not in curve:
        print(f"Reading (fixed in advance, D-53): {reading(curve)}")
    print(f"{time.perf_counter() - started:.0f}s")
    if args["out"]:
        Path(args["out"]).write_text(json.dumps(table, indent=1))


def main(argv: list[str]) -> None:
    args = dict(DEFAULTS)
    for item in argv:
        key, _, value = item.partition("=")
        if key not in args:
            raise SystemExit(f"unknown argument '{key}'; known: {sorted(args)}")
        args[key] = type(DEFAULTS[key])(value)
    if args["part"] == "label":
        if not args["agent"] or not args["out"]:
            raise SystemExit("part=label needs agent= and out=")
        label(args)
    elif args["part"] == "check":
        if not args["agents"] or not args["held"].endswith(".jsonl"):
            raise SystemExit("part=check needs agents= and held=<held.jsonl>")
        check(args)
    else:
        raise SystemExit("part= must be label or check")


if __name__ == "__main__":
    main(sys.argv[1:])
