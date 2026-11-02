"""Gain head (D-52, Q-123, after E-47): can the network learn where the planner's finish can be
beaten?

The value head's target becomes the gain at a position: the planner's cost to finish minus the
cheapest finish the search saw from there (0 or more; 0 at about three positions in four,
E-47). In a search a position would be worth spent + planner finish - predicted gain. Two parts:

- `part=label`: replay up to `per_stage` training problems per stage of the demos and all the
  held-out ones (the last 5%, as the copy step splits them), run the planner finish at every
  position, and write the demos again with the gain in place of the remaining cost (nan where
  the planner does not finish or there is no target). The original samples are kept as
  `samples_cost`. The held-out problems go last, so the copy step's held-out 5% is a subset of
  them. Then the copy step (`imitation_check.py arm=visits`) trains on the file as on any demos;
  its value head learns -gain x reward_scale.
- `part=check`: on the copy step's held-out problems (the last 5% of the label file), the
  sibling test of E-46: how often spent + planner - predicted gain picks the cheapest sibling,
  against spent + planner alone; and how well the predicted gain separates positions with a
  gain from those without (AUC).

Reading, fixed in advance (D-52), on held-out s4-s5: passes if the gain picks the best sibling
>= 0.05 more often than the planner alone and AUC >= 0.7 → next, the search on set C at equal
time; else it fails, and the line says which part.

    uv run python scripts/gain_check.py part=label \\
        demos=outputs/expert/round2/demos.jsonl,outputs/expert/round3/demos.jsonl \\
        out=outputs/gain/labels.jsonl workers=36
    uv run python scripts/gain_check.py part=check agent=outputs/gain/copy/imitation.pt \\
        labels=outputs/gain/labels.jsonl workers=36

Arguments are key=value: `part`, `demos`, `labels`, `agent`, `per_stage` (default 300),
`workers`, `threads`, `device`, `reward_scale`, `max_chunk_rows`, `out`.
"""

from __future__ import annotations

import json
import math
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_iteration as ei  # noqa: E402
import fit_check as fc  # noqa: E402
import resolution_check as rc  # noqa: E402

DEFAULTS: dict[str, Any] = {
    "part": "",
    "demos": "",
    "labels": "",
    "agent": "",
    "per_stage": 300,
    "workers": 8,
    "threads": 8,
    "device": "cpu",
    "reward_scale": 0.01,
    "max_chunk_rows": 32768,
    "out": "",
}
MORE_BEST, AUC = 0.05, 0.7


def gains(task: tuple[str, str]) -> tuple[tuple[Any, ...], int, int]:
    """The samples with the gain in place of the remaining cost, how many have a gain target
    and how many a gain above 0."""
    inst_json, samples = task
    inst = Instance.from_json(inst_json)
    env = build_env(fc._cfg(), [inst])
    out = []
    for actions, visits, remaining in ei._unpack(samples):
        env.reset(options={"instance": inst})
        for a in actions:
            env.step(env.legal.index(a))
        planner = rc.planner_finish(env, inst)
        known = remaining is not None and math.isfinite(remaining) and not math.isnan(planner)
        out.append((actions, visits, max(0.0, planner - remaining) if known else math.nan))
    n = sum(not math.isnan(g) for _, _, g in out)
    return tuple(out), n, sum(g > 1e-9 for _, _, g in out)


def replay(task: tuple[str, str, float]) -> Any:
    """`resolution_check.replay`, by a name this module's workers can find."""
    return rc.replay(task)


def label(args: dict[str, Any]) -> None:
    lines = [
        json.loads(x)
        for p in args["demos"].split(",")
        for x in Path(p).read_text().split("\n")
        if x
    ]
    chosen = [i for part, i in fc.split(lines, args["per_stage"]) if part == "train"]
    chosen += [i for part, i in fc.split(lines, 0) if part == "held"]
    started = time.perf_counter()
    tasks = [(lines[i]["instance"], lines[i]["samples"]) for i in chosen]
    out = Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    n = positive = 0
    with Pool(args["workers"]) as pool, out.open("w") as f:
        for k, (i, (samples, got, pos)) in enumerate(
            zip(chosen, pool.imap(gains, tasks, chunksize=1), strict=True), 1
        ):
            rec = {**lines[i], "samples_cost": lines[i]["samples"], "samples": ei._pack(samples)}
            f.write(json.dumps(rec) + "\n")
            n, positive = n + got, positive + pos
            if k % 25 == 0:
                print(f"{k}/{len(tasks)}  {time.perf_counter() - started:.0f}s", flush=True)
    print(
        f"labelled {len(tasks)} problems: {n} positions "
        f"with a gain target, {positive} above 0 ({positive / max(n, 1):.0%}); "
        f"{time.perf_counter() - started:.0f}s"
    )


def auc(score: npt.NDArray[np.float64], positive: npt.NDArray[np.bool_]) -> float | None:
    """Area under the ROC curve: the chance a random positive outscores a random negative
    (ties count half)."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts  # rank (from 0) of each distinct value's first copy
    ranks = first[inverse] + (counts[inverse] - 1) / 2 + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(rows: dict[str, npt.NDArray[Any]]) -> dict[str, Any]:
    """`rows` per position: parent (-1 at the top), spent, target, planner (costs to finish, nan
    where none) and pred (predicted gain)."""
    known = ~np.isnan(rows["target"]) & ~np.isnan(rows["planner"])
    gain = np.maximum(0.0, rows["planner"] - rows["target"])
    groups: dict[int, list[int]] = defaultdict(list)
    for i in np.flatnonzero(known & (rows["parent"] >= 0)):
        groups[int(rows["parent"][i])].append(int(i))
    planner_best, gain_best, chance = [], [], []
    for kids in (g for g in groups.values() if len(g) >= 2):
        k = np.array(kids)
        total = rows["spent"][k] + rows["target"][k]
        best = np.isclose(total, total.min())
        plan = rows["spent"][k] + rows["planner"][k]
        planner_best.append(bool(best[np.argmin(plan)]))
        gain_best.append(bool(best[np.argmin(plan - rows["pred"][k])]))
        chance.append(best.mean())
    pos = known & (gain > 1e-9)
    return {
        "positions": int(known.sum()),
        "with_gain": float(pos.sum() / known.sum()) if known.any() else None,
        "gain": rc._median(gain[pos]),
        "pred_if_gain": rc._median(rows["pred"][pos]),
        "pred_if_none": rc._median(rows["pred"][known & ~pos]),
        "auc": auc(rows["pred"][known], pos[known]),
        "groups": len(chance),
        "planner_best": float(np.mean(planner_best)) if planner_best else None,
        "gain_best": float(np.mean(gain_best)) if gain_best else None,
        "chance_best": float(np.mean(chance)) if chance else None,
    }


def reading(row: dict[str, Any]) -> str:
    """The rule fixed in advance (D-52), held-out s4-s5."""
    if row["gain_best"] is None or row["auc"] is None:
        return "no data"
    more = row["gain_best"] - row["planner_best"]
    picks = f"picks the best sibling {row['gain_best']:.2f} vs planner {row['planner_best']:.2f}"
    if more >= MORE_BEST and row["auc"] >= AUC:
        return f"passes: {picks}, AUC {row['auc']:.2f}; next, the search on set C"
    fails = [f"{picks} (needs +{MORE_BEST})"] if more < MORE_BEST else []
    fails += [f"AUC {row['auc']:.2f} < {AUC}"] if row["auc"] < AUC else []
    return "fails: " + "; ".join(fails)


def check(args: dict[str, Any]) -> None:
    started = time.perf_counter()
    torch.set_num_threads(args["threads"])
    scale = args["reward_scale"]
    lines = [json.loads(x) for x in Path(args["labels"]).read_text().split("\n") if x]
    held = range(len(lines) - max(1, len(lines) // 20), len(lines))  # as the copy step
    tasks = [(lines[i]["instance"], lines[i]["samples_cost"], scale) for i in held]
    with Pool(args["workers"]) as pool:
        replayed = pool.map(replay, tasks, chunksize=1)
    print(f"replayed {len(replayed)} held-out problems in {time.perf_counter() - started:.0f}s")
    device = torch.device(args["device"])
    net = PPOAgent.load(args["agent"], device).net
    net.eval()
    cells = fc.stage_names()
    groups: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for i, (store, spent, planner, _) in zip(held, replayed, strict=True):
        if not len(store):
            continue
        _, value = fc.predict(net, store, device, args["max_chunk_rows"])
        stage = cells.get(lines[i]["cell"], "small")
        for key in (stage, "s4-s5") if stage in ("s4_targets", "s5_storage") else (stage,):
            g = groups[key]
            offset = sum(len(x) for x in g["spent"])
            g["parent"].append(np.array([p + offset if p >= 0 else -1 for p in store.parent]))
            g["spent"].append(np.array(spent))
            g["target"].append(-np.array(store.value, dtype=np.float64) / scale)
            g["planner"].append(np.array(planner))
            g["pred"].append(np.maximum(0.0, -value / scale))
    table = {
        key: metrics({k: np.concatenate(v) for k, v in g.items()})
        for key, g in sorted(groups.items())
    }
    cols = list(next(iter(table.values()))) if table else []
    print("| held-out | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, row in table.items():
        print(f"| {name} | " + " | ".join(fc._fmt(row[c]) for c in cols) + " |")
    if "s4-s5" in table:
        print(f"Reading (fixed in advance, D-52): {reading(table['s4-s5'])}")
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
        if not args["demos"] or not args["out"]:
            raise SystemExit("part=label needs demos= and out=")
        label(args)
    elif args["part"] == "check":
        if not args["agent"] or not args["labels"]:
            raise SystemExit("part=check needs agent= and labels=")
        check(args)
    else:
        raise SystemExit("part= must be label or check")


if __name__ == "__main__":
    main(sys.argv[1:])
