"""`scripts/scale_check.py` (D-58): probes are s5's many-target cells at 20 and 40 targets; a
finish over the cap is recorded, not dropped; the midpoint is a real state of the start's plan;
the reading follows the rule fixed in advance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from meda.core.state import PrepState

ROOT = Path(__file__).resolve().parents[2]


def load() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("scale_check", ROOT / "scripts/scale_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scale_check"] = module
    spec.loader.exec_module(module)
    return module


SC = load()


def test_problems_are_s5_many_target_cells_and_probes_scale_targets() -> None:
    s5 = SC.problems("s5", 4, 0)
    assert len({i.instance_id for i in s5}) == 4
    assert all("-many-" in i.metadata["cell"] for i in s5)
    probes = SC.problems("t20", 3, 0)
    assert [len(i.targets) for i in probes] == [20, 20, 20]
    assert {i.metadata["cell"] for i in probes} == {c for c in (i.metadata["cell"] for i in s5)}
    assert [i.instance_id for i in probes] == ["t20-00", "t20-01", "t20-02"]


def test_a_finish_over_the_cap_is_recorded_and_the_midpoint_skipped() -> None:
    inst = SC.problems("t20", 1, 0)[0]
    found = SC.timed_finish(PrepState.initial(inst), inst, 0.05)
    assert found["over_cap"] and "cost" not in found
    rec = SC.run_task(("t20", inst, 0.05))
    assert rec["root"]["over_cap"] and rec["mid"] == {"skipped": "start over the cap"}


def test_the_midpoint_finish_is_from_halfway_along_the_plan() -> None:
    inst = SC.problems("s5", 1, 0)[0]
    rec = SC.run_task(("s5", inst, 600.0))
    assert not rec["root"]["over_cap"] and not rec["mid"]["over_cap"]
    assert rec["plan_len"] > 0 and "plan" not in rec["root"]
    assert rec["mid"]["cost"] > 0


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    def recs(root: float, mid: float, n: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "targets": 20,
                "root": {"seconds": root, "over_cap": False},
                "mid": {"seconds": mid, "over_cap": False},
            }
            for _ in range(n)
        ]

    fast = SC.size_stats(recs(2.0, 1.0))
    assert SC.reading({"s5": fast, "t20": fast}).startswith("no size reaches")
    slow = SC.size_stats(recs(12.0, 10.0))
    assert SC.reading({"s5": fast, "t40": slow}).startswith("the finish is unaffordable at t40")
    capped = recs(5.0, 1.0)
    for r in capped[:6]:
        r["root"] = {"seconds": 1800.0, "over_cap": True}
        r["mid"] = {"skipped": "start over the cap"}
    st = SC.size_stats(capped)
    assert st["over_cap"] == 6 and st["median"] == float("inf")
    assert "unaffordable" in SC.reading({"t40": st})
