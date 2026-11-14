"""Plan 8.4 step 6 / D-88: sharing on the grid (`scripts/grid_time.py`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from meda.generators.benchmark_io import BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("grid_time", ROOT / "scripts/grid_time.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GT = _module()
V = "60x60_u5"


def _prod(makespan: int | None, cost: float = 1.0, small: int | None = 60) -> dict[str, Any]:
    p: dict[str, Any] = {"cost": cost}
    for size in (30, 60, 100):
        ok = makespan is not None and small is not None and size >= small
        p[f"{size}x{size}_u5"] = (
            {"makespan": makespan, "schedule_makespan": 10, "storage_cell_ticks": 0}
            if ok
            else {"deadlock": "stuck"}
        )
    return p


def test_compare_counts_runs_and_pairs_times_on_problems_both_finish() -> None:
    recs = [
        {"producers": {"share": _prod(12), "construction": _prod(10)}},
        {"producers": {"share": _prod(None), "construction": _prod(10)}},
        {"producers": {"share": _prod(10), "construction": {"none": "no plan"}}},
    ]
    c = GT.compare(recs, "share", "construction", V)
    assert c["n"] == 2 and c["runs_ours"] == 0.5 and c["runs_other"] == 1.0
    assert c["feasibility"] == "costs feasibility"
    assert c["n_both_run"] == 1 and c["time_ratio"] == 1.2 and c["travel_overhead_ours"] == 1.2


def test_smallest_chip_pairs_ours_against_the_comparator() -> None:
    recs = [
        {"producers": {"a": _prod(5, small=100), "b": _prod(5, small=60)}},
        {"producers": {"a": _prod(5, small=30), "b": _prod(5, small=60)}},
        {"producers": {"a": _prod(5, small=60), "b": _prod(5, small=60)}},
        {"producers": {"a": _prod(None), "b": _prod(None)}},
    ]
    assert GT.chip_sizes(recs, "a", "b") == {"bigger": 1, "same": 1, "smaller": 1, "neither": 1}


def test_run_plays_every_plan_on_every_variant() -> None:
    inst = BenchmarkSet.read("test_s3_v1").instances[1]
    task = {"set": "test_s3_v1", "cell": "", "instance": inst.to_json()}
    opts = {
        "methods": ["mrcm"],
        "method": False,
        "seconds": 1.0,
        "sims": 4,
        "sizes": (30, 60),
        "units": (5,),
        "check": True,
    }
    rec = GT.run(task, opts)
    share = rec["producers"]["share"]
    assert {"30x30_u5", "60x60_u5"} <= set(share)
    m = share["60x60_u5"]
    assert m["makespan"] >= m["schedule_makespan"] > 0
    summary = GT.summarise([rec])
    assert summary["ours"] == "share_mrcm_look3" and summary["variants"] == ["30x30_u5", V]
