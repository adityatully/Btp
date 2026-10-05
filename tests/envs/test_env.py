"""SamplePrepEnv, encoding, wrappers and the random agent (SPEC 9, 17.2; milestone M3)."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from meda.agents.base import run_episode
from meda.agents.random_agent import RandomAgent
from meda.core.actions import Action, Mix, Output, Source
from meda.core.composition import Composition, Rational
from meda.core.dag import validate
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError, TerminalStatus, apply, legal_actions
from meda.envs.encoding import OP_MIX
from meda.envs.sample_prep_env import FixedInstances, SamplePrepEnv
from meda.envs.wrappers import CurriculumWrapper, EpisodeStats, PreferenceWrapper
from tests.core.helpers import chip, default_weights, dilution, mixture
from tests.envs.helpers import AnyIndexToLegal, make_env, make_tiny_env, tiny_grid

S, B = Source.reservoir(0), Source.reservoir(1)
CASE3 = dilution([(3, 10)])
CASE3_PLAN: list[Action] = [Mix(S, 2, B, 3), Mix(Source.droplet(2), 3, B, 1)]


def index_of(env: SamplePrepEnv, action: Action) -> int:
    key = action.canonical_key(env.state)
    return [a.canonical_key(env.state) for a in env.legal].index(key)


def random_episode(env: SamplePrepEnv, rng: np.random.Generator, **reset: Any) -> list[Any]:
    """Steps (obs, reward, terminated, truncated, info) of one random legal episode."""
    obs, _ = env.reset(**reset)
    steps = [(obs, 0.0, False, False, {})]
    while True:
        out = env.step(np.int64(rng.integers(len(env.legal))))
        steps.append(out)
        if out[2] or out[3]:
            return steps


# --------------------------------------------------------------------------- API


def test_check_env() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "env is wrapped": the wrapper is for check_env only
        check_env(AnyIndexToLegal(make_tiny_env()), skip_render_check=True)


def test_mask_and_padding_follow_legal_actions() -> None:
    env = make_env([dilution([(1, 4), (3, 8)], demand=2, step_limit=12)])
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        while True:
            assert env.observation_space.contains(obs)
            n_legal = len(env.legal)
            assert env.legal == (() if done else legal_actions(env.state, env.instance))
            expected = np.arange(env.config.sizes.l_max) < n_legal
            assert (env.action_masks() == expected).all()
            assert (obs["action_mask"].astype(bool) == expected).all()
            assert (obs["candidates"][n_legal:] == 0).all()
            assert (obs["cand_index"][n_legal:] == -1).all()
            n_src = env.instance.num_fluids + len(env.state.droplets)
            assert obs["source_mask"].sum() == n_src
            assert (obs["sources"][n_src:] == 0).all()
            assert obs["target_mask"].sum() == len(env.instance.targets)
            if done:
                break
            obs, _, term, trunc, _ = env.step(np.int64(rng.integers(n_legal)))
            done = term or trunc


def test_masked_index_raises() -> None:
    env = make_env([CASE3])
    env.reset(seed=0)
    for bad in (len(env.legal), env.config.sizes.l_max - 1, -1):
        with pytest.raises(IllegalActionError, match="masked"):
            env.step(np.int64(bad))


def test_step_after_end_raises() -> None:
    env = make_env([dilution([(1, 2)])])
    env.reset(seed=0)
    env.step(np.int64(index_of(env, Mix(S, 1, B, 1))))
    _, _, term, _, _ = env.step(np.int64(index_of(env, Output(Source.droplet(2), 0))))
    assert term
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.int64(0))


def test_success_episode_reward_cost_and_dag() -> None:
    env = make_env([CASE3])
    env.reset(seed=0)
    rewards = []
    for act in [*CASE3_PLAN, Output(Source.droplet(3), 0)]:
        _, r, term, trunc, info = env.step(np.int64(index_of(env, act)))
        rewards.append(r)
    assert term and not trunc
    assert info["status"] is TerminalStatus.SUCCESS and info["success"]
    weights = default_weights()
    assert info["plan_cost"] == weights.scalar(info["episode_totals"], CASE3.fluids)
    assert sum(rewards) == pytest.approx(-float(info["plan_cost"]))
    assert info["penalty"] == Rational.of(0)
    validate(info["dag"], CASE3).raise_if_invalid()


def test_truncation_adds_penalty() -> None:
    inst = dilution([(3, 10)], step_limit=1)
    env = make_env([inst], penalty=7)
    env.reset(seed=0)
    _, r, term, trunc, info = env.step(np.int64(0))
    assert trunc and not term
    assert info["status"] is TerminalStatus.TRUNCATED
    step_cost = default_weights().scalar(info["cost_breakdown"], inst.fluids)
    assert r == pytest.approx(-float(step_cost) - 7)
    assert info["penalty"] == Rational.of(7)
    validate(info["dag"], inst, complete=False).raise_if_invalid()


def test_dead_end_adds_penalty() -> None:
    # No mix is legal (max_denominator 1). Target 0 is pure S (served from the reservoir);
    # target 1 (1/2) can never be made, so after the output nothing is legal.
    base = dilution([(1, 1), (1, 2)], chip_limits=chip(max_denominator=1))
    env = make_env([base], penalty=5)
    env.reset(seed=0)
    assert env.legal == (Output(S, 0),)
    _, r, term, trunc, info = env.step(np.int64(0))
    assert term and not trunc
    assert info["dead_end"] and info["status"] is TerminalStatus.DEAD_END
    assert r == pytest.approx(-10 - 5)  # one unit of sample (weight 10) + penalty


def test_reset_rejects_instances_that_do_not_fit() -> None:
    big = dilution([(1, 4)])  # M = 8: 28 legal mixes at the start
    with pytest.raises(ValueError, match=r"test-dilution.*L_max 10"):
        make_env([big], L_max=10).reset(seed=0)
    with pytest.raises(ValueError, match=r"test-mixture.*F_max 2"):
        make_env([mixture([(1, 1, 1)])], F_max=2).reset(seed=0)
    with pytest.raises(ValueError, match="no legal action at the start"):
        make_env([dilution([(1, 2)], chip_limits=chip(max_denominator=1))]).reset(seed=0)
    wide = dilution([(1, 4)], droplet_volume=9)
    with pytest.raises(ValueError, match="droplet_volume 9 > max_mix_volume"):
        make_env([wide]).reset(seed=0)


def test_reset_options_choose_instance_and_seed_is_deterministic() -> None:
    grid = tiny_grid()
    env = make_env(grid, L_max=128)
    _, info = env.reset(options={"instance": grid[3]})
    assert info["instance_id"] == grid[3].instance_id
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    a = random_episode(make_tiny_env(), rng_a, seed=11)
    b = random_episode(make_tiny_env(), rng_b, seed=11)
    assert len(a) == len(b)
    for (oa, ra, *_), (ob, rb, *_) in zip(a, b, strict=True):
        assert ra == rb
        assert all((oa[k] == ob[k]).all() for k in oa)


# --------------------------------------------------------------------------- encoding


def test_encoding_of_known_state() -> None:
    env = make_env([CASE3])
    env.reset(seed=0)
    obs, *_ = env.step(np.int64(index_of(env, CASE3_PLAN[0])))  # droplet 2: 2/5, volume 5
    z = env.config.sizes
    droplet_row = obs["sources"][2]
    assert droplet_row[:2] == pytest.approx([0.4, 0.6])
    assert droplet_row[z.f_max] == pytest.approx(5 / 8)  # volume / M
    assert droplet_row[z.f_max + 1] == 0  # not a reservoir
    reservoir_row = obs["sources"][1]
    assert reservoir_row[:2] == pytest.approx([0.0, 1.0])
    assert reservoir_row[z.f_max + 1] == 1

    j = index_of(env, CASE3_PLAN[1])  # 3 units of 2/5 + 1 unit of B -> 3/10, volume 4
    cand = obs["candidates"][j]
    assert list(obs["cand_index"][j]) == [OP_MIX, 2, 1]
    assert cand[OP_MIX] == 1
    assert cand[3:6] == pytest.approx([3 / 8, 1 / 8, 4 / 8])
    assert cand[6:8] == pytest.approx([0.3, 0.7])  # exact result, as float
    c = 6 + z.f_max
    assert cand[c] == pytest.approx(1 - 1 / 10)  # denominator feature
    assert cand[c + 1] == pytest.approx(0.0)  # L-inf to target 0
    assert cand[c + 1 + z.t_max] == 1  # satisfies an open target
    reagent = cand[c + 2 + z.t_max : c + 5 + z.t_max]
    assert reagent == pytest.approx([0, 0, 1 / 8])  # 1 unit of buffer
    assert cand[-1] == pytest.approx(2 / 10)  # 2 droplets after (2/5 keeps 2 units)
    # The flag is exact: it is set iff applying the candidate yields exactly 3/10.
    sat = obs["candidates"][: len(env.legal), c + 1 + z.t_max]
    for act, flag in zip(env.legal, sat, strict=True):
        if isinstance(act, Mix):
            _, _, delta = apply(env.state, act, env.instance)
            assert bool(flag) == (delta.nodes[0].composition == Composition.from_ints((3, 7), 10))


def test_age_feature_grows() -> None:
    env = make_env([dilution([(1, 3)], step_limit=10)])
    env.reset(seed=0)
    env.step(np.int64(index_of(env, Mix(S, 1, B, 2))))
    obs, *_ = env.step(np.int64(index_of(env, Mix(S, 1, B, 1))))
    z = env.config.sizes
    ages = sorted(obs["sources"][2:4, z.d_src - 1].tolist())
    assert ages == pytest.approx([0.0, 0.1])


# --------------------------------------------------------------------------- agents, wrappers


def test_random_agent_episodes_are_legal_and_dags_validate() -> None:
    grid = tiny_grid()
    env = make_tiny_env()
    agent = RandomAgent(seed=1)
    for inst in grid:
        result = run_episode(agent, env, inst)
        assert result.dag is not None
        assert result.dag.producer == "random"
        validate(result.dag, inst, complete=result.status == "feasible").raise_if_invalid()


def test_random_agent_save_load(tmp_path: Any) -> None:
    agent = RandomAgent(seed=42)
    agent.save(tmp_path / "a.json")
    assert RandomAgent.load(tmp_path / "a.json").seed == 42


def test_episode_stats_wrapper() -> None:
    env = EpisodeStats(make_env([CASE3]))
    env.reset(seed=0)
    base = env.unwrapped
    assert isinstance(base, SamplePrepEnv)
    total = 0.0
    for act in [*CASE3_PLAN, Output(Source.droplet(3), 0)]:
        _, r, _, _, info = env.step(np.int64(index_of(base, act)))
        total += float(r)
    stats = info["episode_stats"]
    assert stats["return"] == pytest.approx(total)
    assert stats["length"] == 3 and stats["success"] and stats["mix_ops"] == 2


class TwoStageCurriculum:
    def __init__(self, first: Instance, second: Instance) -> None:
        self.stages = [FixedInstances([first]), FixedInstances([second])]
        self.finished = 0

    def source(self) -> FixedInstances:
        return self.stages[min(self.finished, 1)]

    def record(self, episode_info: dict[str, Any]) -> None:
        self.finished += 1


def test_curriculum_wrapper_switches_source() -> None:
    a, b = dilution([(1, 2)]), replace(dilution([(1, 4)]), instance_id="second")
    cur = TwoStageCurriculum(a, b)
    env = CurriculumWrapper(make_env([a]), cur)
    _, info = env.reset(seed=0)
    assert info["instance_id"] == a.instance_id
    rng = np.random.default_rng(0)
    base = env.unwrapped
    assert isinstance(base, SamplePrepEnv)
    while True:
        _, _, term, trunc, _ = env.step(np.int64(rng.integers(len(base.legal))))
        if term or trunc:
            break
    assert cur.finished == 1
    _, info = env.reset()
    assert info["instance_id"] == "second"


def test_preference_conditioning_is_reserved() -> None:
    with pytest.raises(NotImplementedError):
        PreferenceWrapper(make_tiny_env())
    with pytest.raises(NotImplementedError):
        make_env(tiny_grid(), preference_conditioning=True)


def test_async_vector_env() -> None:
    from gymnasium.vector import AsyncVectorEnv

    venv = AsyncVectorEnv([make_tiny_env, make_tiny_env])
    try:
        obs, _ = venv.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(30):
            actions = [
                int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in obs["action_mask"]
            ]
            obs, *_ = venv.step(np.array(actions))
    finally:
        venv.close()


# --------------------------------------------------------------------------- M3 acceptance


@pytest.mark.slow
def test_random_agent_10000_episodes_on_tiny_dilution() -> None:
    env = make_tiny_env()
    rng = np.random.default_rng(0)
    successes = 0
    for episode in range(10_000):
        steps = random_episode(env, rng, seed=episode)
        for obs, *_ in steps:
            assert env.observation_space.contains(obs)
        successes += bool(steps[-1][4]["success"])
    assert successes > 0
