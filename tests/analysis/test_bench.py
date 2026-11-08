"""`scripts/bench.py` (benchmark plan B1, D-69): the answer key holds nothing of ours, every
producer's plan is costed under every goal, and the reading follows the rule fixed in advance."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def bench_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/bench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = module
    spec.loader.exec_module(module)
    return module


B = bench_module()


@pytest.fixture(autouse=True)
def back_to_fluid() -> Iterator[None]:
    """`use_goal` sets the script modules' weights; later tests expect today's goal."""
    yield
    B.gc.use_goal("fluid")


OPTS = {
    "exact_seconds": 2.0,
    "exact_nodes": 10_000,
    "nfsp_seconds": 10.0,
    "orders": 2,
    "seconds": 5.0,
    "sims": 2,
    "commit": "test",
}


def test_one_problem_under_each_goal_gives_a_whole_record() -> None:
    task = B.tasks(["test_s1_v1"], ["fluid"], 1)[0]
    rec = B.run_task(task, OPTS)
    assert set(rec["published"]) == set(B.PUBLISHED)
    assert rec["bound"] <= rec["construction"]
    ours = rec["ours"]
    assert set(ours["by_goal"]) == set(B.GOALS)
    assert rec["bound"] - 1e-9 <= ours["by_goal"]["fluid"] <= rec["construction"] + 1e-9
    assert B.ours_at(rec, 1e9) == min([c for _, c in ours["trace"]] + [rec["best_planner"]])
    for st in rec["planners"].values():
        assert st["sample"] >= 1 and st["mixes"] >= 1
    steps = B.run_task(B.tasks(["test_s1_v1"], ["steps"], 1)[0], OPTS)
    assert "published" not in steps  # stored once, with the fluid record


def test_verdict_follows_the_rule_fixed_in_advance() -> None:
    base = {"n": 20, "wins": 15, "losses": 2, "mean": 0.9, "median": 0.9}
    assert B.verdict(base | {"ci": (0.85, 0.95)}) == "beats"
    assert B.verdict(base | {"losses": 3, "ci": (0.85, 0.95)}) == "level"  # 15% losses
    assert B.verdict(base | {"ci": (0.95, 1.02)}) == "level"
    assert B.verdict(base | {"ci": (1.01, 1.1)}) == "loses"


def test_paired_counts_and_bootstrap_interval() -> None:
    p = B.paired([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0])
    assert (p["n"], p["wins"], p["losses"]) == (4, 1, 2)
    assert p["mean"] == (0.5 + 1 + 1.5 + 2) / 4
    assert p["ci"][0] <= p["mean"] <= p["ci"][1]


def test_ours_at_reads_the_trace() -> None:
    rec = {"ours": {"trace": [(0.5, 100.0), (30.0, 90.0), (200.0, 80.0)]}}
    assert B.ours_at(rec, 0.1) is None
    assert B.ours_at(rec, 60.0) == 90.0
    assert B.ours_at(rec, 900.0) == 80.0


def test_ours_at_takes_the_planner_with_its_time() -> None:
    """The cheaper of planner and search (Q-140); the search starts after the planners' 10 s."""
    rec = {
        "best_planner": 85.0,
        "planner_seconds": 10.0,
        "ours": {"trace": [(0.5, 100.0), (30.0, 90.0), (200.0, 80.0)]},
    }
    assert B.ours_at(rec, 5.0) is None
    assert B.ours_at(rec, 11.0) == 85.0
    assert B.ours_at(rec, 60.0) == 85.0
    assert B.ours_at(rec, 205.0) == 85.0  # the 80 plan comes at 10 + 200 s
    assert B.ours_at(rec, 900.0) == 80.0


def test_emrcm_table_puts_the_papers_numbers_beside_ours() -> None:
    emrcm = {"by_goal": {"fluid": 200.0}, "mixes": 10, "waste": 9}
    rec = {
        "set": "test_emrcm_v1",
        "goal": "fluid",
        "cell": "N2-T2",
        "bound": 100.0,
        "published": {"emrcm": emrcm},  # only the methods run (`methods=`)
        "ours": {"trace": [(5.0, 150.0)], "mixes": 5, "waste": 3},
    }
    rows = B.emrcm_table([rec])
    assert rows[-1] == "| N2-T2 | 1 | 1 | 0.75 | 5.00 / 10.00 / 10.52 | 3.00 / 9.00 / 5.93 | 1.50 |"


def test_run_task_runs_only_the_methods_asked_for() -> None:
    task = B.tasks(["test_s1_v1"], ["fluid"], 1)[0]
    rec = B.run_task(task, OPTS | {"methods": ["mrcm"]})
    assert set(rec["published"]) == {"mrcm"}


def test_dmfb_checks_flag_a_plan_below_nfsp() -> None:
    """B3 (D-71): NFSP is fluid-optimal among 1:1 graphs, so a plan of ours with less fluid cost
    is a bug, and the check names it."""

    def st(sample: int, buffer: int, cost: float) -> dict[str, Any]:
        return {"by_goal": {"fluid": cost}, "sample": sample, "reagent": 0, "buffer": buffer}

    rec = {
        "set": "dmfb_sweep_v1",
        "goal": "fluid",
        "id": "dmfb_sweep-007",
        "cell": "d4",
        "optimum": None,
        "published": {"nfsp": st(2, 2, 140.0)},
        "bitscan": st(3, 2, 210.0),
        "planners": {"share": st(2, 2, 141.0)},
        "ours": {**st(1, 2, 70.0), "trace": [(1.0, 70.0)]},
    }
    text = "\n".join(B.dmfb_checks([rec]))
    assert "less fluid cost than NFSP: 1 (must be 0)" in text
    assert "dmfb_sweep-007 method 66 < NFSP 130" in text
    assert "dearer than NFSP: 0" in text
