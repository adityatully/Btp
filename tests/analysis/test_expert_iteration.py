"""`scripts/expert_iteration.py` (D-38): the search arm's plan is never dearer than the
planner's, both replay and validate, and `load_demos` returns each arm's plan."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from meda.core.rules import replay
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def module(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EI = module("expert_iteration")


@pytest.mark.parametrize("instance", TINY.instances[:3], ids=lambda i: i.instance_id)
def test_search_arm_never_dearer_and_demos_round_trip(instance: Any, tmp_path: Path) -> None:
    rec = EI.run_task((instance.to_json(), 8, 30.0))
    assert rec["search_cost"] <= rec["planner_cost"] + 1e-9
    assert rec["improved"] == (rec["search_cost"] < rec["planner_cost"] - 1e-9)
    path = tmp_path / "demos.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    for arm in ("search", "planner"):
        insts, plans = EI.load_demos(path, arm)
        assert [i.instance_id for i in insts] == [instance.instance_id]
        replay(insts[0], plans[instance.instance_id])  # raises on an illegal plan
    with pytest.raises(ValueError, match="arm"):
        EI.load_demos(path, "oracle")


def test_practice_problems_are_new() -> None:
    small = EI.small_problems(1)
    assert small and all(i.instance_id.startswith("ei-") for i in small)
    held = EI.search_test_hashes()
    assert len(held) == 36
