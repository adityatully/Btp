"""D-96: the order check (`order_check.py`) and `share.named_orders`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from meda.core.state import PrepState
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("order_check", ROOT / "scripts/order_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OC = _module()


def _multi() -> Any:
    insts = BenchmarkSet.read("val_s4_v2", BENCHMARKS_DIR).instances
    return min(
        (i for i in insts if len(share.orders(i, OC.WEIGHTS)) >= 3),
        key=lambda i: len(i.targets),
    )


def test_named_orders_cover_every_rule_once_and_match_orders() -> None:
    inst = _multi()
    named = share.named_orders(inst, OC.WEIGHTS)
    assert [o for _, o in named] == share.orders(inst, OC.WEIGHTS)
    rules = [r for rs, _ in named for r in rs]
    assert sorted(rules) == sorted(share.ORDER_RULES)


def test_per_order_minimum_is_finish_from() -> None:
    inst = _multi()
    state = PrepState.initial(inst)
    runs = OC.per_order(state, inst, None)
    found = share.finish_from(state, inst, OC.WEIGHTS)
    assert found is not None
    assert min(r["cost"] for r in runs if r["cost"] is not None) == float(found[1])


def test_repeats_and_reading() -> None:
    runs = [
        {"rules": ["given"], "open": [1, 2], "cost": 10.0, "sec": 1.0},
        {"rules": ["dear", "near"], "open": [2, 1], "cost": 9.0, "sec": 2.0},
        {"rules": ["cheap"], "open": [1, 2], "cost": 10.0, "sec": 3.0},
    ]
    assert OC.repeats(runs) == (6.0, 3.0, 3, 2)
    assert OC.choosy(runs)
    fc = OC.rule_costs(runs)
    assert fc == {"given": 10.0, "dear": 9.0, "near": 9.0, "cheap": 10.0}
    rd = OC.reading([fc])
    assert rd["wins"] == {"given": 0, "dear": 1, "cheap": 0, "near": 1}
    assert rd["best"][1]["hold"] == 1.0 and rd["best"][1]["excess"] == 0.0


def test_run_task_end_to_end(tmp_path: Path) -> None:
    task = OC.tasks(("val_s4_v2",), 1)[0]
    rec = OC.run_task(task, {"seconds": 3.0, "sims": 2, "states": 3, "cuts": 2, "mrcm": False})
    assert rec["leaves"] >= 1 and len(rec["finishes"]["share"]) == len(rec["kinds"])
    assert rec["kinds"][0] == "root" and rec["kinds"].count("cut") <= 2
    out = tmp_path / "o.jsonl"
    import json

    out.write_text(json.dumps(rec) + "\n")
    assert "Order check" in OC.summarise(out)
