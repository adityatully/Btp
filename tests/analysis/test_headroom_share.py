"""`scripts/headroom_share.py` (plan 2.0): seeding exact search with a plan must not change the
optimum, the extra target orders never make the sharing pass dearer, and the headroom verdict
follows the rule fixed in advance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.base import SolveBudget
from meda.solvers.baselines.share import share_plan
from meda.solvers.exact_search import ExactSearch

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def headroom_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "headroom_share", ROOT / "scripts" / "headroom_share.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HS = headroom_module()


@pytest.mark.parametrize("instance", TINY.instances[:6], ids=lambda i: i.instance_id)
def test_seeding_with_the_sharing_pass_keeps_the_optimum(instance: Any) -> None:
    plan = share_plan(instance, HS.WEIGHTS)
    assert plan is not None
    budget = SolveBudget(max_seconds=60)
    seeded, cost, stats = HS.SeededSearch(plan, HS.cost_of(instance, plan)).search(instance, budget)
    plain = ExactSearch(HS.WEIGHTS).solve(instance, budget)
    assert not stats.stopped and plain.status == "optimal"
    assert cost == plain.cost
    assert seeded is not None and HS.cost_of(instance, seeded) == cost


@pytest.mark.parametrize("instance", TINY.instances[:6], ids=lambda i: i.instance_id)
def test_more_target_orders_never_cost_more(instance: Any) -> None:
    base = share_plan(instance, HS.WEIGHTS)
    more = HS.share_orders(instance, 8)
    assert base is not None and more is not None
    assert HS.cost_of(instance, more) <= HS.cost_of(instance, base)
    HS.plan_stats(instance, "share_orders", more, 0.0)  # replays and validates the DAG


def record(share: float, best: float, optimum: float | None) -> dict[str, Any]:
    by = "share" if best == share else "share_look3"
    return {
        "share": {"cost": share, "sample": 2, "waste": 1},
        "share_look3": {"cost": best, "sample": 1},
        "best": {"cost": best, "by": by},
        "optimum": optimum,
        "bound": optimum or 50.0,
        "target_sample": 1.0,
    }


def test_verdict_follows_the_rule_fixed_in_advance() -> None:
    at_opt = [record(100, 100, 100) for _ in range(8)]
    assert not HS.cell_summary(at_opt)["headroom"]
    # a quarter of the proved problems more than 2% above the optimum
    some_gap = at_opt[:6] + [record(103, 100, 100) for _ in range(2)]
    assert HS.cell_summary(some_gap)["headroom"]
    # nothing proved, but the best plan found is >= 5% cheaper on the median problem
    found = [record(100, 94, None) for _ in range(5)]
    s = HS.cell_summary(found)
    assert s["headroom"] and s["improved"] == 5 and s["sample_saved_median"] == 1
    # cheaper plans on a minority only: no verdict from them
    few = [record(100, 100, None) for _ in range(4)] + [record(100, 80, None)]
    assert not HS.cell_summary(few)["headroom"]
