"""Hint tables (`solvers.hints`) and the env's hint columns (SPEC 9.3): exact facts about which
intermediates lead to a target, the information the copy test lacked (paper log E-16)."""

from __future__ import annotations

import numpy as np
import pytest

from meda.analysis.evaluate import build_env
from meda.core.actions import Mix
from meda.core.costs import CostWeights
from meda.core.rules import apply
from meda.core.state import PrepState
from meda.envs.encoding import CHAIN_MAX, HINT_COL, RECIPE_MAX
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines.share import share_plan
from meda.solvers.hints import CandidateHints, key_of
from tests.core.helpers import chip, dilution, load_yaml, mixture
from tests.envs.helpers import make_env

GEOMETRIC = CostWeights.from_dict(load_yaml("costs/geometric.yaml"))
CFG = {"costs": load_yaml("costs/geometric.yaml"), "env": load_yaml("env/sample_prep.yaml")}


def test_chain_names_the_unexpected_intermediate_of_33_40() -> None:
    """33/40 S at M = 9: one mix of 1 unit of 1/8 S with 4 units of S makes it (40 = 8 x 5),
    though 1/8 is far from 33/40 by distance. 3/4 S, much closer, is two or more mixes away."""
    inst = dilution([(33, 40)], chip("meda_m8", max_mix_volume=9))
    tables = CandidateHints(GEOMETRIC).tables(inst, 0)
    eighth, quarter3 = (1, 7, 8), (3, 1, 4)
    assert tables.chain_left(eighth, volume=1) == 1
    assert tables.chain_left(quarter3, volume=4) != 1
    assert tables.recipe_left(key_of(inst.targets[0].composition), volume=1) == 0


def test_chain_needs_enough_volume() -> None:
    inst = dilution([(33, 40)], chip("meda_m8", max_mix_volume=9), droplet_volume=5)
    tables = CandidateHints(GEOMETRIC).tables(inst, 0)
    # A 5-unit droplet of 33/40 needs all 5 units of the last mix: 1 of 1/8 + 4 of S only
    # yields 5 units when the 1/8 droplet holds at least 1 unit.
    assert tables.chain_left((1, 7, 8), volume=1) == 1
    assert tables.chain.get((1, 7, 8)) is not None


@pytest.mark.parametrize("name", ["val_s1_v2", "val_s2_v2"])
def test_every_recipe_entry_is_made_by_a_teacher_plan(name: str) -> None:
    """Recipe entries are compositions the sharing pass or the tree really makes, with the
    delivered composition at 0; on stages 1-2 every mix of the sharing pass is in the table."""
    hints = CandidateHints(GEOMETRIC)
    for inst in BenchmarkSet.read(name).instances[:10]:
        tables = hints.tables(inst, 0)
        assert 0 in tables.recipe.values()
        plan = share_plan(inst, GEOMETRIC)
        assert plan is not None
        state = PrepState.initial(inst)
        for action in plan:
            if isinstance(action, Mix):
                uid = state.next_uid
                state, _, _ = apply(state, action, inst)
                if state.has_droplet(uid):
                    drop = state.droplet(uid)
                    assert tables.recipe_left(key_of(drop.composition), drop.volume) is not None
            else:
                state, _, _ = apply(state, action, inst)


def test_hint_columns_are_one_hot_on_mix_rows_and_zero_elsewhere() -> None:
    inst = mixture([(1, 1, 2), (1, 2, 1)], demand=2, step_limit=14)
    env = build_env(CFG, [inst])
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4):
        n = len(env.legal)
        hints = obs["candidates"][:n, HINT_COL:]
        recipe, chain = hints[:, : RECIPE_MAX + 2], hints[:, RECIPE_MAX + 2 :]
        assert chain.shape[1] == CHAIN_MAX + 2
        is_mix = np.array([isinstance(a, Mix) for a in env.legal])
        assert (recipe[is_mix].sum(axis=1) == 1).all()
        assert (chain[is_mix].sum(axis=1) == 1).all()
        assert (hints[~is_mix] == 0).all()
        assert (obs["candidates"][n:, HINT_COL:] == 0).all()
        mixes = np.flatnonzero(is_mix)
        obs, *_ = env.step(int(rng.choice(mixes)))


def test_hint_columns_stay_zero_without_a_hint_source() -> None:
    env = make_env([mixture([(1, 1, 2)], step_limit=10)])
    obs, _ = env.reset(seed=0)
    assert (obs["candidates"][:, HINT_COL:] == 0).all()


def test_problem_hints_call_no_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source "problem" (Q-105): the recipe columns come from the problem's arithmetic alone;
    the tree and the sharing pass are never called."""
    import meda.solvers.hints as hints_mod

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("planner called")

    monkeypatch.setattr(hints_mod, "share_plan", refuse)
    monkeypatch.setattr(hints_mod, "build", refuse)
    inst = dilution([(33, 40)], chip("meda_m8", max_mix_volume=9))
    tables = CandidateHints(GEOMETRIC, "problem").tables(inst, 0)
    assert tables.recipe == {}
    assert tables.recipe_left(key_of(inst.targets[0].composition), volume=1) == 0
    assert tables.recipe_left((1, 7, 8), volume=1) == 1  # 1/8 S, one mix from 33/40
    assert tables.chain_left((1, 7, 8), volume=1) == 1  # the chain columns are unchanged


def test_problem_hints_reach_deeper_than_the_chain() -> None:
    """A droplet three pure-fluid mixes from the target has no chain hint (depth 2) but a
    recipe-column value of 3 under source "problem"."""
    inst = dilution([(1, 16)], chip("meda_m8", max_mix_volume=2))
    tables = CandidateHints(GEOMETRIC, "problem").tables(inst, 0)
    half = (1, 1, 2)  # 1/2 S -> 1/4 -> 1/8 -> 1/16, one buffer unit each time
    assert tables.chain_left(half, volume=1) is None
    assert tables.recipe_left(half, volume=1) == 3


def test_problem_hint_source_reaches_the_env() -> None:
    cfg = {"costs": CFG["costs"], "env": {**CFG["env"], "hint_source": "problem"}}
    env = build_env(cfg, [dilution([(33, 40)], chip("meda_m8", max_mix_volume=9))])
    assert env.encoder.hints is not None
    assert env.encoder.hints.source == "problem"
    with pytest.raises(ValueError, match="hint_source"):
        build_env({**cfg, "env": {**CFG["env"], "hint_source": "oracle"}}, [])
