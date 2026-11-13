"""D-87: leftover time, search again with more simulations (`search_check`, `grow_check.py`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GC = _module("grow_check")
import search_check as SC  # noqa: E402


def _run(grow: int, max_sims: int | None) -> dict[str, Any]:
    inst = BenchmarkSet.read("val_s1_v2", BENCHMARKS_DIR).instances[0]
    task = {
        "set": "val_s1_v2",
        "id": inst.instance_id,
        "cell": "",
        "instance": inst.to_json(),
        "optimum": None,
        "planner": 1e9,
        "mrcm": None,
        "greedy": None,
    }
    opts = {
        "agent": None,
        "seconds": 120.0,
        "hints": "planner",
        "reward_scale": 0.01,
        "first_plan": True,
        "grow": grow,
        "max_sims": max_sims,
    }
    return SC.run_task(task, "uniform", 4, opts, "switch")


def test_later_passes_start_from_method_v3_and_never_get_dearer() -> None:
    plain = _run(1, None)
    grown = _run(4, 64)
    assert len(plain["rounds"]) == 1
    assert [s for s, _, _ in grown["rounds"]] == [4, 16, 64]
    # the first pass is method-v3 itself (deterministic)
    assert grown["rounds"][0][1] == plain["rounds"][0][1] == plain["search_cost"]
    costs = [c for _, c, _ in grown["rounds"]]
    assert costs == sorted(costs, reverse=True)
    assert grown["search_cost"] == costs[-1]


def _rec(first: float, last: float, opt: float | None = None) -> dict[str, Any]:
    return {
        "set": "val_s1_v2",
        "goal": "fluid",
        "optimum": opt,
        "published": {"nfsp": {"by_goal": {"fluid": first}}},
        "ours": {"rounds": [[256, first, 1.0], [1024, last, 9.0]]},
    }


def test_reading_joins_on_enough_cheaper_problems_or_a_closed_gap() -> None:
    many = [_rec(100, 99)] * 2 + [_rec(100, 100)] * 8
    assert GC.summarise(many)[-1].endswith("leftover time joins the method")
    few = [_rec(100, 99)] + [_rec(100, 100)] * 19
    assert GC.summarise(few)[-1].endswith("dropped, method-v3 stays")
    gap = [_rec(106, 102, 100)] + [_rec(100, 100, 100)] * 19
    assert GC.summarise(gap)[-1].endswith("leftover time joins the method")
