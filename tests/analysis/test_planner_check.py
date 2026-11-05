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
