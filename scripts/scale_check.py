"""Scale check (paper log D-58): does one planner finish get too slow to search with?

A network costs the same at any size; the search's planner finish (`share.finish_from`, one per
leaf) grows with the problem. On set D one finish took about 1 s and the network still added
nothing (E-51). Here, per size, 10 problems; per problem one finish from the start and one from
the midpoint of that finish's plan, each timed:

- `s5`: `val_s5_v2`'s many-target problems, spread evenly over their cells (inside SPEC 1.3);
- `t20`, `t40`: probes outside 1.3, drawn from s5's many-target cells of `track2_v1` with
  `targets_per_experiment` set to 20 or 40 and demand 2-8 (settings here; configs unchanged).

A finish is capped at `cap` seconds (default 1800) and recorded as over the cap, never dropped
(rule 9); the midpoint is skipped when the start's finish is over the cap.

Reading, fixed in advance (D-58): a size makes the finish unaffordable if its median finish
(start and midpoint together; over the cap, or a midpoint skipped for it, counts as over) is
>= 10 s; then step 2 races the network there (user's go-ahead). No size reaches it: the finish
stays affordable as far as we probe, and the network has no role at these sizes.

    uv run python scripts/scale_check.py workers=30 out=outputs/scale/records.jsonl
    uv run python scripts/scale_check.py summary=outputs/scale/records.jsonl

Arguments are key=value: `sizes` (default s5,t20,t40), `n` problems per size (default 10),
`cap` seconds per finish, `seed`, `workers`, `out`.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from collections import defaultdict
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from meda.core.problem import Instance
from meda.core.rules import replay
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.generators.sampler import GeneratorSource
from meda.generators.stages import StageDesign
from meda.solvers.baselines.share import finish_from

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom  # noqa: E402

WEIGHTS = headroom.WEIGHTS
PROBES = {"t20": 20, "t40": 40}  # targets per experiment in the probes
PROBE_DEMAND = [2, 8]
# Fixed in advance (D-58): unaffordable if the median finish takes at least this long.
SLOW = 10.0


class OverCap(Exception):
    pass


def _alarm(signum: int, frame: Any) -> None:
    raise OverCap


def timed_finish(state: PrepState, inst: Instance, cap: float) -> dict[str, Any]:
    """One `finish_from`, timed; over `cap` seconds it is stopped and marked."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, cap)
    t0 = time.perf_counter()
    try:
        found = finish_from(state, inst, WEIGHTS)
    except OverCap:
        return {"seconds": round(time.perf_counter() - t0, 3), "over_cap": True}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    out: dict[str, Any] = {"seconds": round(time.perf_counter() - t0, 3), "over_cap": False}
    if found is None:
        raise RuntimeError(f"{inst.instance_id}: no target order finishes from this state")
    out["plan"], out["cost"] = found[0], float(found[1])
    return out


def probe_design(targets: int) -> StageDesign:
    data = yaml.safe_load((ROOT / "configs/stages/track2_v1.yaml").read_text())
    data["axes"]["targets"]["many"]["targets_per_experiment"] = targets
    data["axes"]["demand"]["multi"]["demand"] = PROBE_DEMAND
    return StageDesign.from_dict(data)


def problems(size: str, n: int, seed: int) -> list[Instance]:
    """`n` problems of `size` spread evenly over s5's many-target cells: from val_s5_v2, or
    drawn as probes. Only those cells change in the probes, so they compare like for like."""
    if size == "s5":
        many = [
            i for i in BenchmarkSet.read("val_s5_v2").instances if "-many-" in i.metadata["cell"]
        ]
        if n > len(many):
            raise ValueError(f"val_s5_v2 has {len(many)} many-target problems, not {n}")
        return [many[i * len(many) // n] for i in range(n)]
    cells = [c for c in probe_design(PROBES[size]).stage("s5_storage").cells if "many" in c.values]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cell = cells[i % len(cells)]
        inst = GeneratorSource(cell.config, headroom.CHIP, WEIGHTS)(rng)
        meta = {**inst.metadata, "cell": cell.name}
        out.append(replace(inst, instance_id=f"{size}-{i:02d}", metadata=meta))
    return out


def run_task(task: tuple[str, Instance, float]) -> dict[str, Any]:
    size, inst, cap = task
    rec: dict[str, Any] = {
        "size": size,
        "id": inst.instance_id,
        "cell": inst.metadata.get("cell"),
        "targets": len(inst.targets),
        "demand": sum(t.demand for t in inst.targets),
        "max_den": max(t.composition.den for t in inst.targets),
        "storage": inst.chip.storage_capacity,
    }
    start = PrepState.initial(inst)
    root = timed_finish(start, inst, cap)
    plan = root.pop("plan", None)
    rec["root"] = root
    if plan is None:
        rec["mid"] = {"skipped": "start over the cap"}
        return rec
    rec["plan_len"] = len(plan)
    state, _, _ = replay(inst, plan[: len(plan) // 2])
    mid = timed_finish(state, inst, cap)
    mid.pop("plan", None)
    rec["mid"] = mid
    return rec


# --------------------------------------------------------------------------- summary


def finishes(rs: list[dict[str, Any]]) -> list[float]:
    """Every finish of `rs`, over the cap as infinity; a midpoint skipped because the start
    was over the cap counts as over too (the start took > `cap`, far above `SLOW`)."""
    out = []
    for r in rs:
        for k in ("root", "mid"):
            f = r[k]
            out.append(float("inf") if "skipped" in f or f["over_cap"] else f["seconds"])
    return out


def size_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    fs = finishes(rs)
    root = [r["root"]["seconds"] for r in rs if not r["root"]["over_cap"]]
    mid = [r["mid"]["seconds"] for r in rs if "seconds" in r["mid"] and not r["mid"]["over_cap"]]
    return {
        "n": len(rs),
        "targets": float(np.median([r["targets"] for r in rs])),
        "median": float(np.median(fs)) if fs else None,
        "root_median": float(np.median(root)) if root else None,
        "mid_median": float(np.median(mid)) if mid else None,
        "max": max(fs) if fs else None,
        "over_cap": sum(1 for r in rs for k in ("root", "mid") if r[k].get("over_cap")),
    }


def reading(stats: dict[str, dict[str, Any]]) -> str:
    slow = [s for s, st in stats.items() if st["median"] is not None and st["median"] >= SLOW]
    if slow:
        return (
            f"the finish is unaffordable at {', '.join(slow)} (median >= {SLOW:g} s): "
            "step 2 races the network there"
        )
    return (
        f"no size reaches a median of {SLOW:g} s: the finish stays affordable as far as we "
        "probe, and the network has no role at these sizes"
    )


def _s(x: float | None) -> str:
    return "-" if x is None else ("over cap" if x == float("inf") else f"{x:.2f}")


def summarise(path: Path) -> str:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        by[r["size"]].append(r)
    stats = {s: size_stats(by[s]) for s in ("s5", *PROBES) if s in by}
    lines = [
        "Seconds per planner finish (`finish_from`); median over the start and the midpoint.",
        "",
        "| size | n | targets (median) | median | start median | midpoint median | max "
        "| over cap |",
        "|---" * 8 + "|",
    ]
    for s, st in stats.items():
        lines.append(
            f"| {s} | {st['n']} | {st['targets']:g} | {_s(st['median'])} "
            f"| {_s(st['root_median'])} | {_s(st['mid_median'])} | {_s(st['max'])} "
            f"| {st['over_cap']} |"
        )
    lines += ["", f"Reading (fixed in advance, D-58): {reading(stats)}"]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    sizes = args.get("sizes", ",".join(("s5", *PROBES))).split(",")
    n, cap, seed = int(args.get("n", 10)), float(args.get("cap", 1800)), int(args.get("seed", 0))
    todo = [(s, inst, cap) for s in sizes for inst in problems(s, n, seed)]
    out = Path(args.get("out", "outputs/scale/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problems, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with out.open("w") as fh, Pool(workers) as pool:
        for done, rec in enumerate(pool.imap_unordered(run_task, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{done}/{len(todo)}  {rec['id']}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {len(todo)} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
