"""D-94: novel intermediates against the published methods (`published_novel_check.py`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "published_novel_check", ROOT / "scripts/published_novel_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PN = _module()


def _inst() -> Any:
    return BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[0]


def test_mixed_holds_every_intermediate_and_the_delivered_droplets() -> None:
    inst = _inst()
    plan = share.share_plan(inst, PN.bench.gc.use_goal("fluid"))
    assert plan is not None
    made = PN.mixed(inst, plan)
    assert {PN.nc.show(m["comp"]) for m in PN.nc.intermediates(inst, plan)} <= made
    assert len(made) >= 1


def test_single_target_method_runs_per_target_and_declines_the_whole_problem() -> None:
    inst = _inst()
    assert len(inst.targets) > 1 or inst.targets[0].demand > 1
    whole = PN.run_method(inst, "dfm")
    assert "na" in whole and "comps" not in whole
    one = PN.replace(inst, targets=(PN.replace(inst.targets[0], id=0, demand=1),))
    alone = PN.run_method(one, "dfm")
    assert alone.get("comps"), alone


def test_run_task_reads_every_method_whole_and_per_target() -> None:
    inst = _inst()
    task = {"set": "val_s3_v2", "id": inst.instance_id, "instance": inst.to_json()}
    rec = PN.run_task(task, {"methods": ["dfm", "mrcm"], "nfsp_seconds": 5, "per_target": True})
    assert set(rec["whole"]) == {"dfm", "mrcm"}
    assert len(rec["per_target"]) == len(inst.targets)
    assert "instance" not in rec


def _rec(pid: str, goal: str, cost: float, best: float, novel: list[str]) -> dict[str, Any]:
    return {
        "set": "s",
        "id": pid,
        "goal": goal,
        "best_planner": best,
        "ours": {"by_goal": {goal: cost}, "novel": [{"comp": c} for c in novel]},
    }


def test_summary_counts_a_droplet_any_published_method_makes_as_not_novel() -> None:
    results = [
        {
            "set": "s",
            "id": "a",
            "whole": {"mrcm": {"na": "one target"}},
            "per_target": {"0": {"mrcm": {"comps": ["(1, 1)/2"]}}},
        },
        {"set": "s", "id": "b", "whole": {"mrcm": {"comps": []}}, "per_target": {}},
    ]
    recs = [
        _rec("a", "fluid", 9.0, 10.0, ["(1, 1)/2", "(1, 3)/4"]),  # a win, one made by MRCM
        _rec("b", "fluid", 8.0, 10.0, ["(3, 5)/8"]),  # a win, stays novel
        _rec("b", "waste", 10.0, 10.0, ["(3, 5)/8"]),  # a tie: not read
    ]
    text = "\n".join(PN.summarise(results, recs))
    assert "2 of 3 novel intermediates (67%) stay novel; 2 of 2 wins (100%) keep one" in text
    assert "published methods make a share of them" in text
    assert "- mrcm/t: 1" in text


def test_tasks_take_only_problems_with_novel_intermediates() -> None:
    inst = _inst()
    recs = [
        {"set": "val_s3_v2", "id": inst.instance_id, "goal": "fluid", "ours": {"novel": [{}]}},
        {"set": "val_s3_v2", "id": inst.instance_id, "goal": "waste", "ours": {"novel": [{}]}},
        {"set": "val_s3_v2", "id": "other", "goal": "fluid", "ours": {"novel": []}},
    ]
    got = PN.tasks(recs, None)
    assert [t["id"] for t in got] == [inst.instance_id]
