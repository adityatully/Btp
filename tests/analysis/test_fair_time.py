"""`scripts/fair_time.py` (D-42): a randomised run of the planner is a real plan at the cost it
reports, the answer is never dearer than the planner, and the reading follows the rule fixed in
advance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def fair_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("fair_time", ROOT / "scripts/fair_time.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fair_time"] = module
    spec.loader.exec_module(module)
    return module


FT = fair_module()


@pytest.mark.parametrize("instance", TINY.instances[:4], ids=lambda i: i.instance_id)
def test_random_runs_are_valid_plans_at_the_cost_they_report(instance: Any) -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        b = FT.random_plan(instance, rng, 4)
        if b is None:
            continue
        stats = FT.hs.plan_stats(instance, "share_random", tuple(b.actions), 0.0)
        assert stats["cost"] == pytest.approx(float(b.cost))


@pytest.mark.parametrize("instance", TINY.instances[:2], ids=lambda i: i.instance_id)
def test_answer_is_never_dearer_than_the_planner(instance: Any) -> None:
    plan = share.share_plan(instance, FT.WEIGHTS)
    assert plan is not None
    planner = FT.hs.plan_stats(instance, "share", plan, 0.0)["cost"]
    task = {"set": "C3", "id": instance.instance_id, "instance": instance.to_json()}
    rec = FT.run_task(task | {"planner": planner}, {"seconds": 0.5, "top": 4})
    assert rec["tries"] > 0 and rec["cost"] <= planner + 1e-9
    assert all(v <= planner + 1e-9 for v in rec["at"].values())


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    planner = {str(i): 100.0 for i in range(60)}
    real = {i: 90.0 for i in planner}
    assert FT.verdict(real, planner).endswith("lookahead is real, not time")
    few = {i: (90.0 if int(i) < 40 else 100.0) for i in planner}  # 0.933 but 40 cheaper
    assert FT.verdict(few, planner).endswith("partly lookahead, partly time")
    time_only = {i: 99.5 for i in planner}
    assert FT.verdict(time_only, planner).endswith("the gain was time, not lookahead")
