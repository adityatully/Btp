"""Resolution check (D-50, Q-121, after E-45): is there anything for the network to do?

Two guesses for why neither head has helped, measured on the searches' own data before any
training is built on them:
- **Policy: the search may not need a guide.** If positions have few candidate moves, 256
  simulations try all of them and a prior cannot steer. Measured at positions with visit
  counts: candidates per position, and the share of them the search tried.
- **Value: the choice may be finer than the head's error.** The head predicts the whole cost
  to finish; sibling moves (children of one position) may differ by much less. Measured per
  sibling group, on whole-plan cost (spent so far + remaining, as the search compares them):
  the gap between the best and the runner-up target, against the head's error; how often the
  head and the planner finish (`finish_from`, what the planner-leaf search uses) pick the
  best sibling, against chance; and how far the targets sit from the planner's finish (the
  part a head predicting "cost above the planner" would have to learn).

Targets are the demos' (cheapest whole plan the search saw through a position), so a rarely
tried sibling's target is noisy; the planner finish is exact for what it is.

Reading, fixed in advance, on the held-out s4-s5 problems (new to the network, as in a search):
- policy: median candidates <= 8 and median share tried >= 0.8 → the search tries nearly every
  move, a prior has little to steer; else a prior has room.
- value: median runner-up gap < median head error → the head cannot resolve the choice that
  matters (resolution); else it could, and the problem is elsewhere.
- residual: median |target - planner finish| < half the median target → "cost above the
  planner" is a smaller quantity than the whole cost.

    uv run python scripts/resolution_check.py agent=outputs/expert/round3/copy/imitation.pt \\
        demos=outputs/expert/round2/demos.jsonl,outputs/expert/round3/demos.jsonl workers=36

Arguments as `fit_check.py` (`per_stage` default 50: the planner finish runs at every position).
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
import numpy.typing as npt
import torch

from meda.agents.ppo.policy import PPOAgent
from meda.analysis.evaluate import build_env
from meda.core.problem import Instance
from meda.solvers.baselines.share import finish_from

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_iteration as ei  # noqa: E402
import fit_check as fc  # noqa: E402
import imitation_check as ic  # noqa: E402
import search_check as sc  # noqa: E402

DEFAULTS: dict[str, Any] = {**fc.DEFAULTS, "per_stage": 50}
FEW_MOVES, MOST_TRIED, SMALLER = 8, 0.8, 0.5


def replay(task: tuple[str, str, float]) -> tuple[ic.SearchTargets, list[float], list[float]]:
    """The positions, and at each the cost spent so far and the planner's cost to finish (nan
    where it does not finish within the step limit)."""
    inst_json, samples, reward_scale = task
    inst = Instance.from_json(inst_json)
    store = ic.SearchTargets()
    spent: list[float] = []
    planner: list[float] = []

    def at(env: Any) -> None:
        spent.append(float(env.snapshot().plan_cost))
        found = finish_from(env.state, inst, sc.WEIGHTS)
        ok = found is not None and env.state.step + len(found[0]) <= inst.step_limit
        planner.append(float(found[1]) if ok and found is not None else np.nan)

    store.add_problem(build_env(fc._cfg(), [inst]), inst, ei._unpack(samples), reward_scale, at)
    return store, spent, planner


def _median(x: list[float] | npt.NDArray[Any]) -> float | None:
    return float(np.median(x)) if len(x) else None


def metrics(rows: dict[str, npt.NDArray[Any]]) -> dict[str, Any]:
    """`rows`: per position problem, parent (index in the rows, -1 at the top), n (candidates),
    tried (share with visits; nan without visit counts), spent, target, pred and planner (costs
    to finish, nan where none)."""
    has_pi = ~np.isnan(rows["tried"])
    has_t = ~np.isnan(rows["target"])
    err = np.abs(rows["pred"] - rows["target"])[has_t]
    both = has_t & ~np.isnan(rows["planner"])
    groups: dict[int, list[int]] = defaultdict(list)
    for i in np.flatnonzero(has_t & (rows["parent"] >= 0)):
        groups[int(rows["parent"][i])].append(int(i))
    gaps, rel, head_best, planner_best, chance = [], [], [], [], []
    for kids in (g for g in groups.values() if len(g) >= 2):
        k = np.array(kids)
        total = rows["spent"][k] + rows["target"][k]
        order = np.sort(total)
        gaps.append(order[1] - order[0])
        rel.append((order[1] - order[0]) / max(order[0], 1e-9))
        best = np.isclose(total, order[0])
        head_best.append(bool(best[np.argmin(rows["spent"][k] + rows["pred"][k])]))
        plan = rows["spent"][k] + rows["planner"][k]
        if not np.isnan(plan).all():
            planner_best.append(bool(best[np.nanargmin(plan)]))
        chance.append(best.mean())
    head_err = _median(err)
    return {
        "positions": len(has_t),
        "with_visits": int(has_pi.sum()),
        "candidates": _median(rows["n"][has_pi]),
        "candidates_p90": float(np.percentile(rows["n"][has_pi], 90)) if has_pi.any() else None,
        "tried": _median(rows["tried"][has_pi]),
        "all_tried": float((rows["tried"][has_pi] >= 1).mean()) if has_pi.any() else None,
        "groups": len(gaps),
        "runner_gap": _median(gaps),
        "runner_gap_rel": _median(rel),
        "head_err": head_err,
        "resolvable": float(np.mean(np.array(gaps) > head_err))
        if gaps and head_err is not None
        else None,
        "head_best": float(np.mean(head_best)) if head_best else None,
        "planner_best": float(np.mean(planner_best)) if planner_best else None,
        "chance_best": float(np.mean(chance)) if chance else None,
        "target": _median(rows["target"][has_t]),
        "residual": _median(np.abs(rows["target"] - rows["planner"])[both]),
        "planner_missing": float(np.isnan(rows["planner"]).mean()) if len(has_t) else None,
    }


def reading(row: dict[str, Any]) -> list[str]:
    """The rule fixed in advance (held-out s4-s5)."""
    out = []
    if row["candidates"] is None or row["tried"] is None:
        out.append("policy: no data")
    elif row["candidates"] <= FEW_MOVES and row["tried"] >= MOST_TRIED:
        out.append(
            f"policy: the search tries nearly every move (median {row['candidates']:.0f} "
            f"candidates, {row['tried']:.0%} tried); a prior has little to steer"
        )
    else:
        out.append(
            f"policy: a prior has room (median {row['candidates']:.0f} candidates, "
            f"{row['tried']:.0%} tried)"
        )
    if row["runner_gap"] is None or row["head_err"] is None:
        out.append("value: no data")
    elif row["runner_gap"] < row["head_err"]:
        out.append(
            f"value: the head cannot resolve the choice that matters (runner-up gap "
            f"{row['runner_gap']:.1f} < head error {row['head_err']:.1f})"
        )
    else:
        out.append(
            f"value: the head's error is below the runner-up gap ({row['head_err']:.1f} <= "
            f"{row['runner_gap']:.1f}); resolution is not the problem"
        )
    if row["residual"] is not None and row["target"]:
        smaller = row["residual"] < SMALLER * row["target"]
        out.append(
            f"residual: |target - planner| {row['residual']:.1f} vs target {row['target']:.1f}"
            + (
                ": the cost above the planner is the smaller quantity"
                if smaller
                else ": not smaller"
            )
        )
    return out


def main(argv: list[str]) -> None:
    args = dict(DEFAULTS)
    for item in argv:
        key, _, value = item.partition("=")
        if key not in args:
            raise SystemExit(f"unknown argument '{key}'; known: {sorted(args)}")
        args[key] = type(DEFAULTS[key])(value)
    if not args["agent"] or not args["demos"]:
        raise SystemExit("agent= and demos= are required")
    started = time.perf_counter()
    torch.set_num_threads(args["threads"])
    scale = args["reward_scale"]
    lines = [
        json.loads(x)
        for p in args["demos"].split(",")
        for x in Path(p).read_text().split("\n")
        if x
    ]
    chosen = fc.split(lines, args["per_stage"])
    cells = fc.stage_names()
    tasks = [(lines[i]["instance"], lines[i]["samples"], scale) for _, i in chosen]
    with Pool(args["workers"]) as pool:
        replayed = pool.map(replay, tasks, chunksize=1)
    print(f"replayed {len(replayed)} problems in {time.perf_counter() - started:.0f}s", flush=True)

    device = torch.device(args["device"])
    net = PPOAgent.load(args["agent"], device).net
    net.eval()
    groups: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for (part, i), (store, spent, planner) in zip(chosen, replayed, strict=True):
        if not len(store):
            continue
        _, value = fc.predict(net, store, device, args["max_chunk_rows"])
        stage = cells.get(lines[i]["cell"], "small")
        for key in (stage, "s4-s5") if stage in ("s4_targets", "s5_storage") else (stage,):
            g = groups[(part, key)]
            offset = sum(len(x) for x in g["n"])
            g["n"].append(np.array([o["action_mask"].shape[1] for o in store.obs]))
            g["tried"].append(np.array([np.nan if p is None else (p > 0).mean() for p in store.pi]))
            g["parent"].append(np.array([p + offset if p >= 0 else -1 for p in store.parent]))
            g["spent"].append(np.array(spent))
            g["target"].append(-np.array(store.value, dtype=np.float64) / scale)
            g["pred"].append(np.maximum(0.0, -value / scale))
            g["planner"].append(np.array(planner))
            g["problem"].append(np.full(len(store), i))
    table = {
        f"{part}/{key}": metrics({k: np.concatenate(v) for k, v in g.items()})
        for (part, key), g in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0]))
    }
    for cols in (
        ["positions", "with_visits", "candidates", "candidates_p90", "tried", "all_tried"],
        ["groups", "runner_gap", "runner_gap_rel", "head_err", "resolvable"],
        ["head_best", "planner_best", "chance_best", "target", "residual", "planner_missing"],
    ):
        print("| problems | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for name, row in table.items():
            print(f"| {name} | " + " | ".join(fc._fmt(row[c]) for c in cols) + " |")
        print()
    if "held/s4-s5" in table:
        for line in reading(table["held/s4-s5"]):
            print(f"Reading (fixed in advance): {line}")
    print(f"{time.perf_counter() - started:.0f}s")
    if args["out"]:
        Path(args["out"]).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
