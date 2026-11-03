"""`scripts/lit_bench_full.py` (L-8, Q-118): each method's plan is replayed or its reason
recorded, and the summary groups the reasons."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def full_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts/lit_bench_full.py"
    spec = importlib.util.spec_from_file_location("lit_bench_full", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lit_bench_full"] = module
    spec.loader.exec_module(module)
    return module


LB = full_module()


def test_records_and_summary(tmp_path: Path) -> None:
    todo = [t for t in LB.tasks(2) if t["source"] != "val_s3_v2"]  # s3's first two are slow
    recs = [LB.run_task(t, 30.0) for t in todo]
    for r in recs:
        for name in LB.METHODS:
            v = r[name]
            assert v["cost"] is not None or "na" in v, (r["id"], name, v)
            if v["cost"] is not None:
                assert v["cost"] >= r["bound"] - 1e-9
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    text = LB.summarise(path, None)
    assert "### s4 (2 problems)" in text and "### s5 (2 problems)" in text
    assert "target off the 1/2^d grid" in text or "denominator not M-smooth" in text
