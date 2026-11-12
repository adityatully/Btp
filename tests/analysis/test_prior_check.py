"""D-82: can the network's policy pick the search's moves on big problems (`prior_check.py`)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("prior_check", ROOT / "scripts/prior_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PC = _module()


def test_rank_counts_ties_against_the_move() -> None:
    p = np.array([0.1, 0.4, 0.4, 0.1])
    assert PC.rank_of(p, 1) == 2 and PC.rank_of(p, 2) == 2
    assert PC.rank_of(p, 0) == 4


def test_best_runs_reads_a3s_fluid_reference(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "set": "val_s4_v2", "goal": "fluid", "best_planner": 10},
        {"id": "b", "set": "val_s4_v2", "goal": "steps", "best_planner": 3},
        {"id": "c", "set": "val_emrcm_v1", "goal": "fluid", "best_planner": 5},
    ]
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert PC.best_runs(p) == {"a": 10.0}


def test_run_task_ranks_every_tree_move_of_a_winning_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[0]
    # a fake network: favours the last legal move
    monkeypatch.setattr(PC, "policy", lambda agent, obs, n: np.arange(1, n + 1, dtype=float))
    monkeypatch.setitem(PC._NET, "fake", object())
    task = {"set": "val_s3_v2", "id": inst.instance_id, "instance": inst.to_json(), "best_run": 1e9}
    r = PC.run_task(task, {"seconds": 1.0, "agent": "fake"})
    assert r["wins"] is True
    for m in r["moves"]:
        assert 1 <= m["rank"] <= m["legal"]


def _rec(s: str, ranks: list[tuple[int, int]]) -> dict:
    return {
        "set": s,
        "id": f"{s}-{ranks}",
        "wins": True,
        "moves": [{"depth": d, "legal": n, "rank": k} for d, (n, k) in enumerate(ranks)],
    }


def test_reading_on_s4_s5_needs_the_moves_near_the_top() -> None:
    good = [_rec("val_s4_v2", [(60, 2), (60, 4)]), _rec("val_s5_v2", [(30, 1)])]
    assert PC.summarise(good)[-1].endswith("the network can narrow the choices")
    flat = [_rec("val_s4_v2", [(60, 30), (60, 20)]), _rec("val_s5_v2", [(30, 15)])]
    assert PC.summarise(flat)[-1].endswith("cannot pick the search's moves")
