"""D-80: the causal check of the novel intermediates (`scripts/causal_check.py`)."""

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
    spec = importlib.util.spec_from_file_location("causal_check", ROOT / "scripts/causal_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CC = _module()


def test_finish_cost_of_the_empty_prefix_is_the_finishers_whole_plan() -> None:
    inst = BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[0]
    w = CC.gc.use_goal("fluid")

    def finish(state: PrepState) -> Any:
        return share.finish_from(state, inst, w)

    whole = finish(PrepState.initial(inst))
    assert whole is not None
    assert CC.finish_cost(inst, (), finish) == float(whole[1])
    # one step in: the step's cost plus the finish from there
    plan = whole[0]
    assert CC.finish_cost(inst, plan[:1], finish) is not None


def test_single_target_recipes_hold_each_targets_own_intermediates() -> None:
    inst = BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[0]
    w = CC.gc.use_goal("fluid")
    comps = CC.single_target_comps(inst, w)
    t = inst.targets[0]
    one = CC.replace(inst, targets=(CC.replace(t, id=0),))
    plan = share.share_plan(one, w)
    assert plan is not None
    assert {m["comp"] for m in CC.nc.intermediates(one, plan)} <= comps


def _rec(ref: float, search: float, moves: list[tuple[int, float, float, bool]]) -> dict:
    return {
        "set": "x",
        "id": f"x-{ref}-{search}",
        "reference": ref,
        "search_cost": search,
        "moves": [
            {"comp": "c", "step": s, "uses": 1, "single_target": st, "before": b, "after": a}
            for s, b, a, st in moves
        ],
    }


def test_shares_explained_by_the_best_move_and_by_all_novel_moves() -> None:
    r = _rec(463, 338, [(1, 463, 402, False), (2, 402, 338, False)])
    assert abs(CC.explained(r) - 64 / 125) < 1e-9
    assert CC.together(r) == 1.0
    assert CC.explained(_rec(100, 100, [(1, 100, 90, False)])) is None  # no gain to explain


def test_summary_reads_whether_the_novel_moves_cause_the_gain() -> None:
    holds = [_rec(100, 80, [(1, 100, 80, False)]), _rec(100, 90, [(2, 100, 92, False)])]
    lines = CC.summarise(holds)
    assert lines[-1].endswith("the novel intermediates cause the gains")
    weak = [_rec(100, 80, [(1, 100, 99, True)]), _rec(100, 90, [(2, 100, 99, True)])]
    assert CC.summarise(weak)[-1].endswith("do not alone explain the gains")
