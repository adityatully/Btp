"""`scripts/ablation.py` (D-98): the frozen subset, the run groups, the reading rule, and one
short search run that records the moves the tree committed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from meda.core.composition import Rational

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


def test_new_groups_and_switch_filter() -> None:
    assert len(A.expand("J3")) == 10 and len(A.expand("J5")) == 8
    assert len(A.expand("J4")) == 4 and len(A.expand("J8")) == 9
    assert len(A.expand("J7")) == 5 and len(A.expand("J7o")) == 4
    assert A.config("backup_max+sims64_c0.5") == {"backup": "max", "sims": 64, "cpuct": 0.5}
    entries = json.loads(A.SUBSET.read_text())
    picked = [e for e in entries if A.switch_problem(e)]
    assert {e["set"] for e in picked} == {"val_s5_v2", "val_emrcm_v1"}
    assert all(e["cell"][:2] in ("N4", "N5") for e in picked if e["set"] == "val_emrcm_v1")


def test_problem_settings_are_put_back() -> None:
    base = (A.sc.WEIGHTS, A.sc.COSTS, A.share.named_orders, A.share.CHAIN_DEPTH)
    w = A._set_problem(
        {"costs": "costs/sensitivity/mix4.yaml", "planner": "chain2", "orders": ("dear",)}
    )
    assert w.w_mix == Rational.of(4) and A.sc.hs.WEIGHTS is w and A.share.CHAIN_DEPTH == 2
    A._set_problem({})
    assert (A.sc.WEIGHTS, A.sc.COSTS, A.share.named_orders, A.share.CHAIN_DEPTH) == base


def test_beam_and_j8_runs() -> None:
    entry = json.loads(A.SUBSET.read_text())[0]
    rec = A.run_one({**entry, "config": "beam1", "repeat": 0, "commit": "t", "seconds": 2})
    assert rec["cost"] is not None and rec["baselines"] is None
    rec = A.run_one({**entry, "config": "j8_step_x2", "repeat": 0, "commit": "t", "seconds": 2})
    inst = A.BenchmarkSet.read(entry["set"]).by_id(entry["id"])
    assert rec["step_limit"] == 2 * inst.step_limit
    assert set(rec["baselines"]) == {"share", "share_look3", "share_orders", "share_mrcm_look3"}
    assert A._BASE["costs"] == A.sc.COSTS


def _rows(config: str, repeat: int, factor: float) -> list[str]:
    entries = json.loads(A.SUBSET.read_text())[:20]
    return [
        json.dumps(
            {
                "config": config,
                "repeat": repeat,
                "id": e["id"],
                "cost": 100.0 * factor,
                "depth": 0,
                "leaves": 1,
                "capped": True,
            }
        )
        for e in entries
    ]


def test_summary_reads_own_reference_and_j1_noise(tmp_path: Any) -> None:
    j1 = tmp_path / "j1.jsonl"
    j1.write_text("\n".join(_rows("v3", 0, 1.0) + _rows("v3", 1, 1.02)) + "\n")
    j2 = tmp_path / "j2.jsonl"
    # J2's own method-v3 run is dearer than J1's (other caches); a config 1% below it is
    # inside J1's 2% band: level, read against J2's own v3, not J1's
    j2.write_text("\n".join(_rows("v3", 0, 1.10) + _rows("sims64_c1.5", 0, 1.089)) + "\n")
    text = A.summarise([j2], noise=j1)
    row = next(x for x in text.splitlines() if x.startswith("| sims64_c1.5 | 0 | all |"))
    assert "| 0.9900 [" in row and "| level |" in row
    try:
        A.summarise([j1, j1])
    except ValueError as e:
        assert "twice" in str(e)
    else:
        raise AssertionError("two files with the same run must be refused")
