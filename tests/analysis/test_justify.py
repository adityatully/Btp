"""The justification sweeps' other scripts (D-98): `justify_offline.py` (J0, J9) reads bench
records as the reading says, and `justify_planner.py` (J6) changes one setting of the sharing
pass at a time and always puts the pass back."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OFF = load("justify_offline")
JP = load("justify_planner")


def _rec(i: int, theirs: float) -> dict[str, Any]:
    return {
        "set": "test_s1_v1",
        "id": f"p{i}",
        "goal": "fluid",
        "method": {"portfolio": False},
        "ours": {"trace": [[1.0, 110.0], [50.0, 100.0]], "seconds": 900.0, "switched_at": None},
        "best_planner": 105.0,
        "published": {"facda": {"by_goal": {"fluid": theirs, "steps": 1.0, "waste": 1.0}}},
    }


def test_j0_reads_convergence_and_final_time() -> None:
    recs = [_rec(i, 120.0) for i in range(12)]
    assert OFF.final_time(recs[0]) == 50.0
    text = OFF.j0(recs)
    # at 10 s every problem has a plan 1.1x its final cost and 1.1/1.05 of the planner's
    assert "1.100 [1.100, 1.100]; 1.048; 0%" in text
    assert "1.000 [1.000, 1.000]; 0.952; 100%" in text  # at 60 s: converged, all final


def test_j9_verdicts_and_thresholds() -> None:
    # ours 100 vs 120 on 10 problems, but 2 of 12 lost: beats at 20%, level at 5% and 10%
    recs = [_rec(i, 120.0) for i in range(10)] + [_rec(10, 90.0), _rec(11, 90.0)]
    text = OFF.j9(recs)
    row = next(line for line in text.splitlines() if line.startswith("| facda |"))
    assert "| 12 | level / level / beats (changes) |" in row


def test_planner_settings_restore_the_pass(tmp_path: Path) -> None:
    before = (JP.share.CHAIN_DEPTH, JP.share._chain, JP.share._candidates, JP.tree.aim)
    for name in ("chain1", "last_any", "drop_chain", "aim_finest", "global_greedy"):
        JP.apply(JP.configs()[name])
        JP.apply({})
        assert (JP.share.CHAIN_DEPTH, JP.share._chain, JP.share._candidates, JP.tree.aim) == (
            before
        )


def test_planner_runs_and_base_matches_share_plan() -> None:
    from meda.generators.benchmark_io import BenchmarkSet

    inst = BenchmarkSet.read("val_s1_v2").instances[0]
    task = {"set": "val_s1_v2", "id": inst.instance_id, "commit": "t", "cap": 60.0}
    base = JP.run_one({**task, "config": "base", "base": "plain"})
    plan = JP.share.share_plan(inst, JP.WEIGHTS)
    _, total, _ = JP.replay(inst, plan)
    assert base["cost"] == float(JP.WEIGHTS.scalar(total, inst.fluids))
    for name in ("chain1", "blend0", "drop_tree", "evict_oldest", "greedy_total", "counts_all"):
        rec = JP.run_one({**task, "config": name, "base": "plain"})
        assert rec["cost"] is not None and not rec["capped"], name
    json.dumps(base)


def test_finish_cost_is_the_engines() -> None:
    """F8 (D-98): a finish's reported cost is what the engine charges for its moves, from the
    start and after a few random moves (210 of 210 on the dev subset's s1-s3 and emrcm)."""
    import random
    from dataclasses import replace

    from meda.core.rules import apply, legal_actions
    from meda.core.state import PrepState
    from meda.generators.benchmark_io import BenchmarkSet
    from meda.solvers.fallback import RELAXED_STEP_LIMIT

    for name in ("val_s1_v2", "val_s3_v2"):
        inst = BenchmarkSet.read(name).instances[0]
        relaxed = replace(inst, step_limit=RELAXED_STEP_LIMIT)
        rng = random.Random(0)
        for walk in (0, 3):
            state, prefix = PrepState.initial(inst), []
            for _ in range(walk):
                action = rng.choice(legal_actions(state, inst))
                state, _, _ = apply(state, action, inst)
                prefix.append(action)
            found = JP.share.finish_from(state, inst, JP.WEIGHTS)
            assert found is not None
            _, total, _ = JP.replay(relaxed, [*prefix, *found[0]])
            _, pre, _ = JP.replay(relaxed, prefix)
            w = JP.WEIGHTS
            assert w.scalar(total, inst.fluids) - w.scalar(pre, inst.fluids) == found[1]
