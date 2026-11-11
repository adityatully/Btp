"""Q-127 / D-76: the novel-intermediates check (`scripts/novel_check.py`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from meda.core.actions import Mix, Output
from meda.core.composition import Composition
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("novel_check", ROOT / "scripts/novel_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NC = _module()


def _inst() -> Any:
    return BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[0]


def test_intermediates_are_the_mixes_no_output_delivers_with_their_later_uses() -> None:
    inst = _inst()
    w = NC.gc.use_goal("fluid")
    plan = share.share_plan(inst, w)
    assert plan is not None
    got = NC.intermediates(inst, plan)
    mixes = [a for a in plan if isinstance(a, Mix)]
    delivered = {a.src.index for a in plan if isinstance(a, Output) and not a.src.is_reservoir}
    assert 0 < len(got) <= len(mixes)
    assert all(isinstance(m["comp"], Composition) and m["volume"] >= 1 for m in got)
    # every take from a droplet is a later use of an intermediate or of a delivered droplet
    takes = sum(not s.is_reservoir for a in mixes for s in (a.a, a.b))
    assert sum(m["uses"] for m in got) <= takes
    # every droplet on chip comes from a mix: it is delivered or it is an intermediate
    assert len(got) == len(mixes) - len(delivered)
    assert [m["step"] for m in got] == sorted(m["step"] for m in got)


def test_novel_keeps_only_reused_compositions_no_reference_plan_makes() -> None:
    a, b, c = (Composition.from_ints(n, 4) for n in ((1, 3), (2, 2), (3, 1)))
    search = [
        {"comp": a, "uses": 1},  # made by a reference plan
        {"comp": b, "uses": 0},  # new but never used again
        {"comp": c, "uses": 2},  # new and reused
    ]
    ref = [[{"comp": a, "uses": 0}], []]
    assert NC.novel(search, ref) == [{"comp": c, "uses": 2}]


def _rec(s: str, ref: float, search: float, novel: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "set": s,
        "id": f"{s}-{ref}-{search}",
        "cell": "",
        "best_planner": ref,
        "finisher_alone": ref,
        "reference": ref,
        "search_cost": search,
        "novel": novel,
    }


def test_summary_reads_the_story_from_better_against_tied_problems() -> None:
    n = [{"comp": "(1, 3)/4", "volume": 2, "step": 1, "uses": 2}]
    holds = [_rec("x", 100, 80, n), _rec("x", 100, 90, n), _rec("x", 100, 100, [])]
    assert any(x.endswith("the unexpected-moves story holds") for x in NC.summarise(holds))
    assert "| all | 3 | 2 | 2 (2) | 1 | 0 |" in "\n".join(NC.summarise(holds))
    not_holds = [_rec("x", 100, 80, []), _rec("x", 100, 90, n), _rec("x", 100, 100, n)]
    lines = NC.summarise(not_holds)
    assert any(x.endswith("ordinary choices made better") for x in lines)


def test_run_task_records_costs_and_novel_intermediates_against_the_reference() -> None:
    inst = _inst()
    task = {"set": "val_s3_v2", "id": inst.instance_id, "cell": "", "instance": inst.to_json()}
    rec = NC.run_task(task, {"seconds": 5.0, "sims": 4, "orders": 2})
    assert rec["reference"] <= rec["best_planner"] and rec["reference"] <= rec["finisher_alone"]
    assert rec["search_cost"] <= rec["finisher_alone"] + 1e-9
    assert rec["novel"] is not None and len(rec["novel"]) <= rec["search_reused"]
    assert set(rec["planner_intermediates"]) <= {
        "share",
        "share_look3",
        "share_orders",
        "share_mrcm_look3",
    }
