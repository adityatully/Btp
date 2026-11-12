"""D-81: is the greedy leaf finisher why the search alone loses on s4-s5 (`finisher_check.py`)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "finisher_check", ROOT / "scripts/finisher_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FC = _module()


def _bench(rid: str, s: str, best: float, trace: list[float]) -> dict:
    return {
        "id": rid,
        "set": s,
        "goal": "fluid",
        "best_planner": best,
        "planners": {
            "share": {"by_goal": {"fluid": best + 5}, "seconds": 1.0},
            "share_mrcm_look3": {"by_goal": {"fluid": best}, "seconds": 9.0},
        },
        "ours": {"trace": [[i, c] for i, c in enumerate(trace)]},
    }


def test_losers_are_the_s4_s5_problems_where_the_search_alone_was_dearer(tmp_path: Path) -> None:
    recs = [
        _bench("a", "val_s4_v2", 100, [120, 105]),  # dearer: a loser
        _bench("b", "val_s5_v2", 100, [120, 100]),  # level: not a loser
        _bench("c", "val_s3_v2", 100, [120]),  # other set
        _bench("d", "val_s5_v2", 100, [130]) | {"goal": "steps"},
    ]
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    got = FC.losers(p)
    assert set(got) == {"a"}
    assert got["a"]["a3_alone"] == 105 and got["a"]["won_by"] == "share_mrcm_look3"


def test_run_task_finishes_the_searchs_prefix_three_ways() -> None:
    inst = BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[0]
    task = {
        "set": "val_s3_v2",
        "id": inst.instance_id,
        "instance": inst.to_json(),
        "best_run": 1e9,
        "a3_alone": 1e9,
        "won_by": "share",
        "run_seconds": {},
    }
    r = FC.run_task(task, {"seconds": 1.0})
    assert r["plan_steps"] is not None and 0 <= r["prefix_steps"] <= r["plan_steps"]
    assert set(r["finishes"]) == {"plain", "mrcm_greedy", "mrcm_look3"}
    # the search's own finisher (greedy MRCM, before any switch) gives back its plan's cost
    if r["finish_mode"] == "share_mrcm":
        assert r["finishes"]["mrcm_greedy"]["cost"] <= r["search_cost"] + 1e-9


def _rec(best: float, look3: float) -> dict:
    return {
        "id": f"x-{best}-{look3}",
        "best_run": best,
        "won_by": "share_mrcm_look3",
        "search_cost": best + 10,
        "prefix_steps": 3,
        "plan_steps": 30,
        "finishes": {
            "plain": {"cost": best + 20, "seconds": 0.1},
            "mrcm_greedy": {"cost": best + 10, "seconds": 0.2},
            "mrcm_look3": {"cost": look3, "seconds": 2.0},
        },
    }


def test_reading_names_the_finisher_only_when_lookahead_reaches_the_best_run() -> None:
    reach = [_rec(100, 99 + i % 2) for i in range(10)]
    assert FC.summarise(reach)[-1].endswith("the greedy finisher is the root cause")
    short = [_rec(100, 99) for _ in range(7)] + [_rec(100, 110) for _ in range(3)]
    assert FC.summarise(short)[-1].endswith("not only the finisher")
