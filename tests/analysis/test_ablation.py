"""`scripts/ablation.py` (D-98): the frozen subset, the run groups, the reading rule, and one
short search run that records the moves the tree committed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def ablation_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("ablation", ROOT / "scripts/ablation.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ablation"] = module
    spec.loader.exec_module(module)
    return module


A = ablation_module()


def test_subset_is_frozen_and_rebuilds_the_same() -> None:
    frozen = json.loads(A.SUBSET.read_text())
    assert frozen == A.build_subset()
    assert len(frozen) == 110
    assert len({e["id"] for e in frozen}) == 110
    counts = {name: sum(e["set"] == name for e in frozen) for name, _ in A.SUBSET_SETS}
    assert counts == dict(A.SUBSET_SETS)
    assert not any(e["set"].startswith("test_") for e in frozen)


def test_groups() -> None:
    assert A.expand("J1") == [("v3", 0), ("v3", 1), ("v3", 2), ("seed1", 0), ("seed2", 0)]
    j2 = A.expand("J2")
    assert len(j2) == 12  # the 4 x 3 grid, method-v3's cell is v3 repeat 0
    assert ("v3", 0) in j2 and ("sims256_c1.5", 0) not in j2
    assert A.expand("J1,J2")[:5] == A.expand("J1")  # no duplicates


def _rec(i: str, cost: float) -> dict[str, Any]:
    return {"cost": cost, "group": "s3"}


def test_reading_rule() -> None:
    base = {str(i): _rec(str(i), 100.0) for i in range(20)}
    same = {str(i): _rec(str(i), 100.0) for i in range(20)}
    worse = {str(i): _rec(str(i), 110.0) for i in range(20)}
    c = A.compare(base, same)
    assert c["mean"] == 1.0 and A.verdict(c, None) == "level"
    c = A.compare(base, worse)
    assert A.verdict(c, 0.01) == "worse" and c["worse"] == 20
    assert A.verdict(c, 0.2) == "level"  # inside the noise band


def test_one_short_run_records_depth() -> None:
    entry = json.loads(A.SUBSET.read_text())[0]
    rec = A.run_one({**entry, "config": "sims16_c0.5", "repeat": 0, "commit": "t", "seconds": 2})
    assert rec["cost"] is not None and rec["cpuct"] == 0.5 and rec["sims"] == 16
    assert rec["depth"] >= 0 and rec["leaves"] >= 1 and rec["trace"]
