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
    assert B.ours_at(rec, 1e9) == min(c for _, c in ours["trace"])
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
