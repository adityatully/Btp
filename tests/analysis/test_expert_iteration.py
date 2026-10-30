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


def test_final_weights_draw_mostly_stages_3_to_5() -> None:
    """Round 3 (D-45): s5 (limited storage) takes about its 0.35 weight, s1-s2 cells about 0.1."""
    for weights, s5_share in (("equal", 0.2), ("final", 0.35)):
        cells = [str(i.metadata["cell"]) for i in EI.stage_problems(200, 3, 5, weights)]
        assert abs(sum("limited" in c for c in cells) / 200 - s5_share) < 0.08
    early = sum(c.endswith("one-single-loose") for c in cells)  # s1 and s2 cells, final weights
    assert early / 200 < 0.17
    with pytest.raises(ValueError, match="weights"):
        EI.stage_problems(1, 3, 5, "even")


ROUND2 = {"agent": "", "seed": 0, "samples": True, "min_visits": 4, "value_cap": 8}


@pytest.mark.parametrize("instance", TINY.instances[:2], ids=lambda i: i.instance_id)
def test_round2_samples_are_what_the_search_saw(instance: Any, tmp_path: Path) -> None:
    rec = EI.run_task((instance.to_json(), 8, 30.0, ROUND2))
    assert rec["n_value"] >= rec["n_policy"] >= 1
    path = tmp_path / "demos.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    [(inst, samples)] = EI.load_targets(path)
    assert inst.instance_id == instance.instance_id
    root = [s for s in samples if s[0] == ()]
    assert len(root) == 1 and root[0][1] is not None  # the root always has its visit counts
    optimum = instance.metadata.get("optimum")
    if optimum is not None:
        assert root[0][2] >= float(optimum) - 1e-9
    assert root[0][2] <= rec["planner_cost"] + 1e-9 or not rec["improved"]
    for _actions, visits, remaining in samples:
        assert remaining >= -1e-9
        assert visits is None or sum(visits) >= ROUND2["min_visits"]
    # the same seed gives the same samples (rule 6)
    again = EI.run_task((instance.to_json(), 8, 30.0, ROUND2))
    assert again["samples"] == rec["samples"]


def test_load_targets_needs_samples(tmp_path: Path) -> None:
    rec = EI.run_task((TINY.instances[0].to_json(), 4, 30.0))
    path = tmp_path / "demos.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="samples=1"):
        EI.load_targets(path)
