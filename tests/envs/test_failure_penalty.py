"""SPEC 5.8: with the shipped `failure_penalty_factor`, giving up on purpose never pays better
than a finished plan near the best known one.

The cheapest failure we know of: every step, take the legal action with the lowest immediate
cost that does not finish the episode, until the step limit (or a dead end) adds the penalty.
At factor 1 that failure cost only 1.03-1.33x the best known plan on the stage validation
sets, so any finished plan worse than that earned less than failing (the 2026-10-23 pilot).

That bar holds for `cost_to_finish: construction`, which ignores what is on chip. The shipped
`share` penalty prices the real completion from the state reached, so a failed episode that
built useful intermediates is rightly charged little (paper log E-19: on val_s1-008 the cheap
moves plus the planner's finish cost 168 against a best known 475). For it the guarantee is
per state: the penalty is the factor times the cost of a completion that really exists from
that state, so finishing yourself pays whenever your finish is under factor x the planner's.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from meda.analysis.evaluate import build_env
from meda.core.composition import ZERO
from meda.core.costs import CostWeights
from meda.core.rules import apply
from meda.envs.sample_prep_env import SamplePrepEnv
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines.share import finish_from
from meda.solvers.fallback import RELAXED_STEP_LIMIT, ConstructiveCostToFinish
from tests.core.helpers import load_yaml

# Failing must cost at least this multiple of the best known plan on every problem.
MIN_FAILURE_MULTIPLE = 1.8


def cheapest_failure_cost(env: SamplePrepEnv, weights: CostWeights) -> float:
    """Total cost (penalty included) of the greedy cheapest non-finishing episode."""
    total, done, info = 0.0, False, {}
    while not done:
        state, inst = env.state, env.instance
        options = [apply(state, a, inst) for a in env.legal]
        pick = min(
            range(len(options)),
            key=lambda i: (
                options[i][0].is_success,
                float(weights.scalar(options[i][1], inst.fluids)),
            ),
        )
        _, r, term, trunc, info = env.step(pick)
        total -= r
        done = term or trunc
    assert not info["success"]
    return total


@pytest.mark.slow
@pytest.mark.parametrize("name", ["val_s1_v2", "val_s2_v2", "val_s3_v2"])
def test_failing_on_purpose_costs_well_above_the_best_known_plan(name: str) -> None:
    """Under the construction penalty (wastes the chip, rebuilds)."""
    env_cfg = {**load_yaml("env/sample_prep.yaml"), "cost_to_finish": "construction"}
    cfg = {"costs": load_yaml("costs/geometric.yaml"), "env": env_cfg}
    weights = CostWeights.from_dict(cfg["costs"])
    bench = BenchmarkSet.read(name)
    env = build_env(cfg, list(bench.instances))
    for inst in bench.instances:
        env.reset(options={"instance": inst})
        best_known = float(bench.reference[inst.instance_id].cost)
        failure = cheapest_failure_cost(env, weights)
        assert failure >= MIN_FAILURE_MULTIPLE * best_known, (
            f"{inst.instance_id}: failing costs {failure:.0f}, only "
            f"{failure / best_known:.2f}x the best known {best_known:.0f}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["val_s1_v2", "val_s2_v2", "val_s3_v2"])
def test_share_penalty_is_the_factor_times_a_real_completion(name: str) -> None:
    """Under the shipped `share` penalty, the charge at truncation is `failure_penalty_factor`
    (>= 2) times the cost of a completion that replays from the final state to success."""
    cfg = {"costs": load_yaml("costs/geometric.yaml"), "env": load_yaml("env/sample_prep.yaml")}
    assert cfg["env"]["cost_to_finish"] == "share"
    factor = int(cfg["env"]["failure_penalty_factor"])
    assert factor >= 2
    weights = CostWeights.from_dict(cfg["costs"])
    bench = BenchmarkSet.read(name)
    env = build_env(cfg, list(bench.instances))
    for inst in bench.instances[:15]:
        env.reset(options={"instance": inst})
        cheapest_failure_cost(env, weights)
        state = env.state
        charged = env.cost_to_finish(state, inst)
        relaxed = replace(inst, step_limit=RELAXED_STEP_LIMIT)
        found = finish_from(state, inst, weights)
        completions = [ConstructiveCostToFinish(weights)(state, inst)]
        if found is not None:
            end, total = state, ZERO
            for action in found[0]:
                end, step, _ = apply(end, action, relaxed)
                total = total + weights.scalar(step, inst.fluids)
            assert end.is_success and total == found[1]
            completions.append(found[1])
        assert charged == min(completions), inst.instance_id
