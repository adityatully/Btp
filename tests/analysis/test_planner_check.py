"""`scripts/planner_check.py` (D-61): both planners run per problem; the reading follows the
rule fixed in advance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "planner_check", ROOT / "scripts/planner_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["planner_check"] = module
    spec.loader.exec_module(module)
    return module


PC = load()


def test_a_record_holds_both_planners() -> None:
    (task,) = PC.problems(["val_s1_v2"], 1)
    rec = PC.run_task(task)
    assert rec["share"]["cost"] > 0 and rec["share_mrcm"]["cost"] <= rec["share"]["cost"]


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    def rec(source: str, i: int, plain: float, new: float) -> dict[str, Any]:
        return {
            "source": source,
            "id": f"p{i}",
            "share": {"cost": plain, "seconds": 1.0},
            "share_mrcm": {"cost": new, "seconds": 1.0},
        }

    lit = {("val_s2_v2", f"p{i}"): {"mrcm": 50.0, "best_planner": 100.0} for i in range(13)}
    b = [rec("val_s2_v2", i, 100.0, 50.0) for i in range(13)]

    def full(gain: int, worse: int) -> list[dict[str, Any]]:
        return [
            rec(
                "val_s3_v2",
                100 + i,
                100.0,
                90.0 if i < gain else 110.0 if i < gain + worse else 100.0,
            )
            for i in range(162)
        ]

    assert PC.reading(b + full(41, 0), lit).endswith("joins the planner (user's go-ahead)")
    assert PC.reading(b + full(10, 0), lit).endswith("no cost elsewhere: it joins too")
    assert PC.reading(b + full(41, 20), lit).endswith("a separate variant only")
    assert PC.reading(b + full(10, 20), lit).endswith("drop it")
    missed = [rec("val_s2_v2", i, 100.0, 100.0) for i in range(13)]
    with pytest.raises(RuntimeError, match="dearer than MRCM"):
        PC.reading(missed, lit)


def test_reading_on_recipe_choice_follows_the_rule_fixed_in_advance() -> None:
    def rec(i: int, plain: float, greedy: float, look: float) -> dict[str, Any]:
        def cell(c: float) -> dict[str, float]:
            return {"cost": c, "seconds": 1.0}

        return {
            "source": "val_s4_v2",
            "id": f"p{i}",
            "share": cell(plain),
            "share_mrcm": cell(greedy),
            "share_look3": cell(plain),
            "share_mrcm_look3": cell(look),
        }

    fixed = [
        rec(i, 100.0, 110.0 if i < 17 else 90.0, 100.0 if i < 17 else 90.0) for i in range(242)
    ]
    lines = PC.look_reading(fixed)
    assert "fixed by lookahead: 17 of 17" in lines[-3]
    assert lines[-1].endswith("joins the planner")
    still = [rec(i, 100.0, 110.0, 110.0 if i < 13 else 90.0) for i in range(242)]
    assert PC.look_reading(still)[-1].endswith("stop there")
    assert PC.look_reading([rec(0, 1.0, 1.0, 1.0) | {"share_mrcm_look3": None}][:0]) == []


def test_a_record_holds_the_planners_asked_for() -> None:
    (task,) = PC.problems(["val_s1_v2"], 1)
    rec = PC.run_task((*task, ["share", "share_mrcm", "share_mrcm_look3"]))
    assert rec["share_mrcm_look3"]["cost"] <= rec["share_mrcm"]["cost"]


def test_search_recheck_follows_the_rule_fixed_in_advance() -> None:
    def search(i: int, kind: str, found: float | None) -> dict[str, Any]:
        return {
            "set": "C4",
            "id": f"p{i}",
            "kind": kind,
            "sims": 256,
            "planner": 100.0,
            "search_cost": found,
        }

    look = [
        {"source": "val_s4_v2", "id": f"p{i}", "share_mrcm_look3": {"cost": 90.0}} for i in range(9)
    ]
    # the search at 85 is below the new planner (90) on 3 of 9: >= 1/3
    still = [search(i, "uniform", 85.0 if i < 3 else 95.0) for i in range(9)]
    assert PC.search_recheck(still, look)[-1].endswith(
        "still earns its place over the stronger planner"
    )
    gone = [search(i, "uniform", 85.0 if i < 2 else None) for i in range(9)]
    out = PC.search_recheck(gone, look)
    assert out[-1].endswith("most of what the search was fixing")
    assert "| C | uniform | 9 | 2, 0.967 | 2, 1.074 | 0.900 |" in out
    with pytest.raises(ValueError, match="no share_mrcm_look3 cost"):
        PC.search_recheck(still, look[:3])


def test_strong_finisher_reading_follows_the_rule_fixed_in_advance() -> None:
    def rows(plain: float, strong: float, n_strong: int) -> dict[Any, Any]:
        return {
            ("C", "uniform"): [(plain, 100.0, 100.0)] * 5 + [(100.0, 100.0, 100.0)] * 5,
            ("C", "uniform/strong"): [(strong, 100.0, 100.0)] * n_strong
            + [(100.0, 100.0, 100.0)] * (10 - n_strong),
        }

    assert PC.strong_reading(rows(96.0, 90.0, 6))[-1].endswith("makes the search add more")
    assert PC.strong_reading(rows(96.0, 95.0, 6))[-1].endswith("its gain is what E-59 shows")
    assert PC.strong_reading(rows(96.0, 90.0, 5))[-1].endswith("its gain is what E-59 shows")
    assert PC.strong_reading({("C", "uniform"): []}) == []


def test_strong_records_count_only_plans_found_by_the_cap() -> None:
    look = [{"source": "val_s4_v2", "id": "p0", "share_mrcm_look3": {"cost": 90.0}}]
    rec = {
        "set": "C4",
        "id": "p0",
        "kind": "uniform",
        "sims": 256,
        "planner": 100.0,
        "finisher": "strong",
        "search_cost": 70.0,
        "trace": [(10.0, 95.0), (1200.0, 70.0)],
    }
    out = PC.search_recheck([rec], look)
    assert "| C | uniform/strong | 1 | 1, 0.950 | 0, 1.056 | 0.900 |" in out
