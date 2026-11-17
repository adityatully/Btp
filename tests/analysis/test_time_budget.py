"""`time_budget.py`: CPU time per part of a bench record."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("time_budget", ROOT / "scripts/time_budget.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TB = _module()

REC = {
    "set": "test_s4_v1",
    "exact": {"status": "timeout", "seconds": 60.0},
    "published": {"mrcm": {"cost": 5.0, "seconds": 12.0}, "nfsp": {"cost": None, "na": "x"}},
    "planners": {"share": {"seconds": 1.0}, "share_mrcm_look3": {"seconds": 300.0}},
    "ours": {"seconds": 900.0, "capped": True, "trace": [[0.1, 9.0], [40.0, 8.0]]},
}


def test_parts_and_after_best() -> None:
    p = TB.parts(REC)
    assert p == {
        "answer key: exact search": 60.0,
        "published: mrcm": 12.0,
        "planner: share": 1.0,
        "planner: share_mrcm_look3": 300.0,
        "ours: search": 900.0,
    }
    assert TB.after_best(REC) == 860.0


def test_summary(tmp_path: Path) -> None:
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps(REC) + "\n" + json.dumps({**REC, "set": "test_s5_v1"}) + "\n")
    text = TB.summarise([f])
    assert "2 records" in text and "test_s5_v1" in text
    assert "2 of 2 hit the time cap; final plan found by 60 s on 2" in text
