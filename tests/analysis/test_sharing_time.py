"""Plan 8.3 / D-86: does sharing cost run time (`scripts/sharing_time.py`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from meda.generators.benchmark_io import BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("sharing_time", ROOT / "scripts/sharing_time.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ST = _module()


def test_verdict_follows_d86() -> None:
    assert ST.verdict(1.10, 1.02, 1.18) == "costs time"
    assert ST.verdict(1.00, 0.97, 1.03) == "no cost"
    assert ST.verdict(0.90, 0.85, 0.97) == "saves time"
    assert ST.verdict(1.06, 0.99, 1.13) == "unclear"  # CI touches 1


def test_compare_pairs_only_problems_both_solve() -> None:
    def prod(makespan: int, cost: float) -> dict[str, Any]:
        return {
            "cost": cost,
            "default": {"makespan": makespan, "storage_ticks": 0},
        }

    recs = [
        {"producers": {"share": prod(12, 50.0), "construction": prod(10, 100.0)}},
        {"producers": {"share": prod(10, 50.0), "construction": prod(10, 100.0)}},
        {"producers": {"share": prod(10, 50.0), "construction": {"none": "no plan"}}},
    ]
    c = ST.compare(recs, "share", "construction")
    assert c["n"] == 2 and c["time_ratio"] == 1.1 and c["cost_ratio"] == 0.5
    assert (c["slower"], c["faster"]) == (1, 0)


def test_run_schedules_every_plan_on_every_platform() -> None:
    inst = BenchmarkSet.read("test_s3_v1").instances[1]
    task = {"set": "test_s3_v1", "cell": "", "instance": inst.to_json()}
    rec = ST.run(task, {"methods": ["mrcm"], "method": False, "seconds": 1.0, "sims": 4})
    assert {"construction", "share", "share_mrcm_look3", "mrcm"} <= set(rec["producers"])
    share = rec["producers"]["share"]
    assert set(ST.PLATFORMS) <= set(share)
    assert share["default"]["makespan"] > 0 and share["cost"] > 0
    # a slower mix never makes a schedule shorter
    assert share["slow_mix"]["makespan"] >= share["default"]["makespan"]
