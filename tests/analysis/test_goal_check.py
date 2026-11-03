"""`scripts/goal_check.py` (D-57): a goal sets the weights everywhere the planners, exact search
and search read them; plans are costed under the goal and checked to be lexicographic; the
search runs where the planner misses the optimum; the reading follows the rule fixed in advance."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from meda.core.composition import Rational

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    import search_check  # noqa: F401  (registers the dataclass module first)

    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GC = load("goal_check")
OPTS = {"seconds": 30.0, "nodes": 200_000, "orders": 4, "sims": 8, "search_seconds": 60.0}


@pytest.fixture(autouse=True)
def back_to_fluid() -> Iterator[None]:
    yield
    GC.use_goal("fluid")


def task(problem_id: str) -> dict[str, Any]:
    return next(
        t
        for t in GC.hs.tasks({"e2"}, None)
        if f"{t['rec']['cell']}-{t['rec']['index']}" == problem_id
    )


def test_a_goal_sets_every_module_and_the_env() -> None:
    w = GC.use_goal("steps")
    assert w.w_mix == Rational.of(65536) and w.w_waste == Rational.of(1)
    assert all(m.WEIGHTS is w for m in (GC.headroom, GC.ts, GC.hs, GC.sc))
    assert GC.sc.COSTS == "costs/steps_first.yaml"
    assert GC.use_goal("waste").w_waste == Rational.of(65536)


def test_plans_must_be_lexicographic() -> None:
    w = GC.use_goal("steps")
    GC.check_first("ok", "steps", w, {"cost": 1.0, "exact": str(3 * 65536 + 900), "mixes": 3})
    with pytest.raises(ValueError, match="not lexicographic"):
        GC.check_first(
            "bad", "steps", w, {"cost": 1.0, "exact": str(3 * 65536 + 65536), "mixes": 3}
        )
    GC.check_first("other goal", "fluid", w, {"cost": 1.0, "exact": "99999999", "mixes": 0})


def test_costs_follow_the_goal() -> None:
    t = GC.hs.tasks({"e2"}, 1)[0]
    fluid = GC.run_task(t, "fluid", OPTS)
    steps = GC.run_task(t, "steps", OPTS)
    assert "exact" not in fluid  # today's goal uses the optima on record
    assert fluid["optimum"] is not None and fluid["optimum"] <= fluid["planner"]
    for name in ("share", "share_look3", "share_orders"):
        st = steps[name]
        assert st["cost"] >= 65536 * st["mixes"]
        assert st["cost"] - 65536 * st["mixes"] == pytest.approx(fluid[name]["cost"] - st["mixes"])
    assert steps["optimum"] is not None and steps["optimum"] <= steps["planner"]


def test_the_search_runs_where_the_planner_misses() -> None:
    rec = GC.run_task(task("dil-one-d1-tight-1"), "waste", OPTS)
    assert rec["planner"] > 1.02 * rec["optimum"]
    assert "search" in rec and rec["search"]["cost"] <= rec["planner"]
    assert rec["optimum"] <= GC.after(rec) <= rec["planner"]


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    def recs(gaps: list[float]) -> list[dict[str, Any]]:
        return [{"planner": 100 * (1 + g), "optimum": 100.0} for g in gaps]

    assert GC.reading(GC.goal_stats(recs([0.0] * 19))).startswith("inconclusive")
    assert GC.reading(GC.goal_stats(recs([0.0] * 15 + [0.03] * 5))).startswith("room")  # 1/4
    assert GC.reading(GC.goal_stats(recs([0.0] * 16 + [0.03] * 4))).startswith("no room")
    assert GC.reading(GC.goal_stats(recs([0.0] * 19 + [1.0]))).startswith("room")  # mean 5%
    found = recs([0.5] * 20)
    for r in found:
        r["search"] = {"cost": 100.0}  # the search reaches the optimum
    assert GC.reading(GC.goal_stats(found)).startswith("no room")
