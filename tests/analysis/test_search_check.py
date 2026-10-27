"""`scripts/search_check.py` (D-37): the search's answer is a valid whole plan, never dearer
than the planner's finish from the start, never below a proved optimum; the reading follows
the rule fixed in advance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from meda.core.state import PrepState
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.baselines.share import finish_from

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def search_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("search_check", ROOT / "scripts/search_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["search_check"] = module  # its dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


SC = search_module()
OPTS = {"agent": None, "seconds": 60.0, "hints": "planner", "reward_scale": 0.01}


@pytest.mark.parametrize("instance", TINY.instances[:4], ids=lambda i: i.instance_id)
def test_flat_prior_search_returns_a_valid_plan_no_dearer_than_the_planner(instance: Any) -> None:
    start = finish_from(PrepState.initial(instance), instance, SC.WEIGHTS)
    assert start is not None
    task = {
        "set": "A",
        "id": instance.instance_id,
        "cell": "",
        "instance": instance.to_json(),
        "optimum": instance.metadata.get("optimum"),
        "planner": float(start[1]),
        "mrcm": None,
        "greedy": None,
    }
    rec = SC.run_task(task, "uniform", 8, OPTS)  # replays and validates the plan
    assert rec["search_cost"] is not None
    assert rec["search_cost"] <= float(start[1]) + 1e-9
    assert not rec["capped"]


def stats(n: int, below: int, ratio_opt: float = 1.1) -> dict[str, Any]:
    return {"n": n, "below": below, "ratio_opt": ratio_opt}


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    assert SC.verdict(stats(9, 3), stats(9, 2), stats(9, 3)).endswith("without the planner")
    found = SC.verdict(stats(9, 3), stats(9, 3, 1.05), stats(9, 2))
    assert found.startswith("(1) yes, (2) no, (3) no") and found.endswith("not from learning")
    assert SC.verdict(stats(9, 2), stats(9, 2), stats(9, 0)).endswith("at this budget")
    # ties on "below" go to the lower mean ratio to the optimum
    assert "(2) yes" in SC.verdict(stats(9, 3, 1.02), stats(9, 3, 1.05), None)
