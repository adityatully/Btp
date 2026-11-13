"""Plan 8.2 / D-85: the scheduling-gap script (`scripts/schedule_gap.py`)."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.pipeline.platform import Platform

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("schedule_gap", ROOT / "scripts/schedule_gap.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SG = _module()


def test_record_has_both_priorities_and_an_optimum_at_or_below_them() -> None:
    inst = BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[0]
    platform = replace(Platform.load(), changeover_ticks=0)
    r = SG.record("val_s3_v2", inst, SG.weights(), platform, ilp_seconds=30)
    assert r["status"] == "ok" and r["optimal"]
    assert r["list"] == min(r["critical_path"], r["plan_order"])
    assert r["ilp"] <= r["list"]


def test_summary_applies_the_d85_bar() -> None:
    base = {"status": "ok", "optimal": True, "ilp_seconds": 0.1, "set": "a"}
    good = [{**base, "critical_path": 10, "plan_order": 11, "list": 10, "ilp": 10}] * 9
    off = {**base, "critical_path": 13, "plan_order": 12, "list": 12, "ilp": 10}
    s = SG.summarise([*good, off])
    o = s["overall"]
    assert o["n"] == 10 and o["at_optimum"] == 9 and o["max_ratio"] == 1.2
    assert o["mean_ratio"] == 1.02 and o["within_10pct"] == 0.9
    assert (o["critical_path_better"], o["plan_order_better"]) == (9, 1)
    assert s["scheduling_done"]
    assert not SG.summarise([*good[:4], off])["scheduling_done"]  # 4 of 5 within 10%
