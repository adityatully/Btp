"""`scripts/hint_coverage.py` (plan 3.2): the hint columns it decodes are the ones the env
writes, and on single-target problems the teacher's own mixes are always on its recipe map."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from meda.generators.benchmark_io import BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]


def coverage_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "hint_coverage", ROOT / "scripts" / "hint_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HC = coverage_module()


def test_every_teacher_mix_on_a_single_target_problem_has_a_recipe_hint() -> None:
    """The recipe map is drawn from the sharing pass's own single-target plan, so the teacher
    never leaves it on stage 1; a decoding error would show up here."""
    for inst in BenchmarkSet.read("val_s1_v2").instances[:5]:
        rows = HC.teacher_trace(inst)
        mixes = [r for r in rows if r["move_mix"]]
        assert mixes, inst.instance_id
        assert all(r["move_recipe"] >= 0 for r in mixes), inst.instance_id
        assert [r["move_recipe"] for r in mixes][-1] == 0  # the last mix makes the target


def test_verdict_follows_the_rule_fixed_in_advance() -> None:
    base = {"planner_last": 0.9, "planner_hinted": 0.9, "p_planner": 0.8}
    assert HC.verdict(base, base) == "neither rule met"
    assert HC.verdict(base, {**base, "planner_last": 0.3}) == "blind off the map"
    assert HC.verdict(base, {**base, "p_planner": 0.1}) == "will not commit"
