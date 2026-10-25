"""SamplePrepEnv, encoding, wrappers and the random agent (SPEC 9, 17.2; milestone M3)."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from meda.agents.base import run_episode
from meda.agents.ppo.policy import trim_obs
from meda.agents.random_agent import RandomAgent
from meda.core.actions import Action, Mix, Output, Source
from meda.core.composition import Composition, Rational, linf_distance
from meda.core.dag import validate
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError, TerminalStatus, apply, legal_actions
from meda.envs.encoding import OP_MIX, TYPE_DROPLET, TYPE_FLUID, TYPE_TARGET
from meda.envs.sample_prep_env import FixedInstances, SamplePrepEnv
from meda.envs.wrappers import CurriculumWrapper, EpisodeStats, PreferenceWrapper
from meda.solvers.fallback import ConstructiveCostToFinish
from tests.core.helpers import chip, default_weights, dilution, mixture
from tests.envs.helpers import AnyIndexToLegal, env_config, make_env, make_tiny_env, tiny_grid

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
            expected = np.arange(env.config.sizes.l_pad) < n_legal
            assert (env.action_masks() == expected).all()
            assert (obs["action_mask"].astype(bool) == expected).all()
            assert (obs["candidates"][n_legal:] == 0).all()
            assert (obs["cand_index"][n_legal:] == -1).all()
            n_drop = len(env.state.droplets)
            assert obs["fluid_mask"].sum() == env.instance.num_fluids
            assert obs["droplet_mask"].sum() == n_drop
            assert (obs["droplets"][n_drop:] == 0).all()
            assert (obs["droplet_comp"][n_drop:] == 0).all()
            assert obs["target_mask"].sum() == len(env.instance.targets)
            if done:
                break
            obs, _, term, trunc, _ = env.step(np.int64(rng.integers(n_legal)))
            done = term or trunc


def test_masked_index_raises() -> None:
    env = make_env([CASE3])
    env.reset(seed=0)
    for bad in (len(env.legal), env.config.sizes.l_pad - 1, -1):
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
    env = make_env([inst], penalty=7, failure_penalty_factor=1)
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
    env = make_env([base], penalty=5, failure_penalty_factor=1)
    env.reset(seed=0)
    assert env.legal == (Output(S, 0),)
    _, r, term, trunc, info = env.step(np.int64(0))
    assert term and not trunc
    assert info["dead_end"] and info["status"] is TerminalStatus.DEAD_END
    assert r == pytest.approx(-10 - 5)  # one unit of sample (weight 10) + penalty


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_failure_penalty_is_scaled_by_the_factor(factor: int) -> None:
    """SPEC 5.8: the penalty is `failure_penalty_factor` x cost_to_finish, on truncation and on
    a dead end alike."""
    trunc_env = make_env(
        [dilution([(3, 10)], step_limit=1)], penalty=7, failure_penalty_factor=factor
    )
    trunc_env.reset(seed=0)
    *_, info = trunc_env.step(np.int64(0))
    assert info["status"] is TerminalStatus.TRUNCATED
    assert info["penalty"] == Rational.of(7 * factor)

    base = dilution([(1, 1), (1, 2)], chip_limits=chip(max_denominator=1))
    dead_env = make_env([base], penalty=5, failure_penalty_factor=factor)
    dead_env.reset(seed=0)
    _, r, *_, info = dead_env.step(np.int64(0))
    assert info["dead_end"] and info["penalty"] == Rational.of(5 * factor)
    assert r == pytest.approx(-10 - 5 * factor)


def test_failure_penalty_factor_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="failure_penalty_factor"):
        make_env(tiny_grid(), failure_penalty_factor=0)


def test_reset_rejects_instances_that_do_not_fit() -> None:
    big = dilution([(1, 4)])  # M = 8: 28 legal mixes at the start
    with pytest.raises(ValueError, match=r"test-dilution.*L_pad 10"):
        make_env([big], L_pad=10).reset(seed=0)
    with pytest.raises(ValueError, match=r"test-mixture.*F_pad 2"):
        make_env([mixture([(1, 1, 1)])], F_pad=2).reset(seed=0)
    with pytest.raises(ValueError, match=r"storage_capacity 10 > env N_pad 4"):
        make_env([big], N_pad=4).reset(seed=0)
    with pytest.raises(ValueError, match="no legal action at the start"):
        make_env([dilution([(1, 2)], chip_limits=chip(max_denominator=1))]).reset(seed=0)
    wide = dilution([(1, 4)], droplet_volume=9)
    with pytest.raises(ValueError, match="droplet_volume 9 > max_mix_volume"):
        make_env([wide]).reset(seed=0)


def test_reset_options_choose_instance_and_seed_is_deterministic() -> None:
    grid = tiny_grid()
    env = make_env(grid, L_pad=128)
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
    vref, cref = 16, 16
    assert obs["fluids"][:2].tolist() == [[1, 0, 0], [0, 0, 1]]  # sample, buffer
    assert obs["droplet_comp"][0, :2] == pytest.approx([0.4, 0.6])
    drop = obs["droplets"][0]
    assert drop[0] == pytest.approx(5 / 8)  # volume / M
    assert drop[1] == pytest.approx(5 / vref)  # volume / volume_ref
    assert drop[2] == pytest.approx(1 - 1 / 5)  # denominator feature
    assert drop[4] == pytest.approx(0.1)  # L-inf from 2/5 to the target 3/10
    assert drop[5] == 0
    assert obs["droplet_target"][0, 0] == pytest.approx([0.1, 0.0])
    assert obs["target_comp"][0, :2] == pytest.approx([0.3, 0.7])
    assert obs["targets"][0] == pytest.approx([1 / 8, 1 / vref, 0, 1, 1 / cref, 0, 1 - 1 / 10])

    j = index_of(env, CASE3_PLAN[1])  # 3 units of 2/5 + 1 unit of B -> 3/10, volume 4
    cand = obs["candidates"][j]
    # (op, a = droplet row 0, b = fluid row 1)
    assert list(obs["cand_index"][j]) == [OP_MIX, TYPE_DROPLET, 0, TYPE_FLUID, 1]
    assert cand[OP_MIX] == 1
    assert cand[3:9] == pytest.approx([3 / 8, 1 / 8, 4 / 8, 3 / vref, 1 / vref, 4 / vref])
    assert obs["cand_comp"][j, :2] == pytest.approx([0.3, 0.7])  # exact result, as float
    assert cand[9] == pytest.approx(1 - 1 / 10)  # denominator feature
    assert cand[10] == pytest.approx(0.0)  # L-inf to the nearest open target
    assert cand[11] == 1  # satisfies an open target
    assert obs["cand_target"][j, 0] == pytest.approx([0.0, 1.0])
    assert cand[12:15] == pytest.approx([0, 0, 1 / vref])  # 1 unit of buffer
    assert cand[16] == pytest.approx(2 / 10)  # 2 droplets after (2/5 keeps 2 units)
    glob = obs["globals"]
    limit = CASE3.step_limit
    assert glob == pytest.approx(
        [1 / limit, (limit - 1) / cref, 1 / 10, 1, 1 / cref, 8 / vref, 10 / cref]
    )
    # The flag is exact: it is set iff applying the candidate yields exactly 3/10.
    for act, flag in zip(env.legal, obs["candidates"][: len(env.legal), 11], strict=True):
        if isinstance(act, Mix):
            _, _, delta = apply(env.state, act, env.instance)
            assert bool(flag) == (delta.nodes[0].composition == Composition.from_ints((3, 7), 10))


def _exact_pair(comp: Composition, inst: Instance, remaining: tuple[int, ...]) -> list[Any]:
    """(distance, satisfies) per target, from `meda.core`, with the met-target convention."""
    out = []
    for t in inst.targets:
        if remaining[t.id] == 0:
            out.append((1.0, 0.0))
        else:
            dist = linf_distance(comp, t.composition)
            out.append((float(dist), float(dist <= t.tolerance)))
    return out


def test_v2_compositions_pairs_and_operands_are_exact() -> None:
    """SPEC 17.2: rows sum to 1, every pair equals the core's exact value, operands point at
    the rows the action names."""
    inst = mixture([(1, 1, 2), (1, 2, 1)], demand=2, step_limit=14)
    env = make_env([inst], L_pad=2048)
    rng = np.random.default_rng(3)
    obs, _ = env.reset(seed=0)
    for _ in range(8):
        state, nf, nt = env.state, inst.num_fluids, len(inst.targets)
        rem = state.remaining_demand
        for name in ("droplet_comp", "target_comp", "cand_comp"):
            real = obs[
                {"droplet_comp": "droplet_mask", "target_comp": "target_mask"}.get(
                    name, "action_mask"
                )
            ].astype(bool)
            assert obs[name][real].sum(axis=1) == pytest.approx(1.0)
        for i, d in enumerate(state.droplets):
            want = _exact_pair(d.composition, inst, rem)
            assert np.allclose(obs["droplet_target"][i, :nt], want)
        row_of = {d.uid: i for i, d in enumerate(state.droplets)}
        for j, act in enumerate(env.legal):
            _, _, delta = apply(state, act, inst)
            idx = obs["cand_index"][j]
            if isinstance(act, Mix):
                result = delta.nodes[0].composition
                for src, (typ, row) in ((act.a, idx[1:3]), (act.b, idx[3:5])):
                    if src.is_reservoir:
                        assert (typ, row) == (TYPE_FLUID, src.index)
                    else:
                        assert (typ, row) == (TYPE_DROPLET, row_of[src.index])
                assert np.allclose(obs["cand_target"][j, :nt], _exact_pair(result, inst, rem))
            elif isinstance(act, Output):
                assert idx[3:5].tolist() == [TYPE_TARGET, act.target]
        if not env.legal:
            break
        obs, _, term, trunc, _ = env.step(np.int64(rng.integers(len(env.legal))))
        if term or trunc:
            break
    assert obs["fluid_mask"].sum() == nf


def test_encoding_is_the_same_at_two_padding_sizes() -> None:
    inst = mixture([(1, 1, 2), (1, 2, 1)], demand=2, step_limit=14)
    small = make_env([inst], L_pad=1200)
    big = make_env([inst], F_pad=7, T_pad=20, N_pad=15, L_pad=1600)
    rng = np.random.default_rng(1)
    a, _ = small.reset(seed=0)
    b, _ = big.reset(seed=0)
    for _ in range(6):
        assert _obs_equal(trim_obs(_batched(a)), trim_obs(_batched(b)))
        i = np.int64(rng.integers(len(small.legal)))
        a, *rest_a = small.step(i)
        b, *_ = big.step(i)
        if rest_a[1] or rest_a[2]:
            break


def test_reset_refuses_an_instance_the_fallback_cannot_price() -> None:
    """An instance with no truncation penalty is refused at reset, not mid-run."""
    unpriceable = mixture([(5, 4, 2)])  # denominator 11, a prime above M = 8: unreachable
    env = SamplePrepEnv(
        env_config(),
        FixedInstances([unpriceable]),
        default_weights(),
        ConstructiveCostToFinish(default_weights()),
    )
    with pytest.raises(ValueError, match=r"test-mixture.*cannot price"):
        env.reset(seed=0)


def _batched(obs: dict[str, Any]) -> dict[str, Any]:
    return {k: v[None] for k, v in obs.items()}


def test_age_feature_grows() -> None:
    env = make_env([dilution([(1, 3)], step_limit=10)])
    env.reset(seed=0)
    env.step(np.int64(index_of(env, Mix(S, 1, B, 2))))
    obs, *_ = env.step(np.int64(index_of(env, Mix(S, 1, B, 1))))
    ages = sorted(obs["droplets"][:2, 3].tolist())
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


# --------------------------------------------------------------------------- snapshot / restore


def _obs_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _play(env: SamplePrepEnv, rng: np.random.Generator, steps: int) -> list[Any]:
    """`steps` random legal actions, resetting from the env's own source after each episode."""
    out = []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(np.int64(rng.integers(len(env.legal))))
        out.append((obs, reward, terminated, truncated, info.get("status"), info.get("plan_cost")))
        if terminated or truncated:
            obs, info = env.reset()
            out.append((obs, info["instance_id"]))
    return out


def _same_trace(a: list[Any], b: list[Any]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if not _obs_equal(x[0], y[0]) or x[1:] != y[1:]:
            return False
    return True


def test_restore_continues_the_run_exactly() -> None:
    """Snapshot mid-episode; a fresh env restored from it (through pickle, as a checkpoint
    would) plays the next steps, across several episode boundaries, exactly like the original."""
    import pickle

    env = make_tiny_env()
    env.reset(seed=3)
    _play(env, np.random.default_rng(1), 23)
    assert not env._done and env.state.step > 0  # really mid-episode
    snap = pickle.loads(pickle.dumps(env.snapshot()))
    obs_at_snap = env.observe()

    straight = _play(env, np.random.default_rng(2), 60)

    resumed_env = make_tiny_env()
    resumed_env.reset(seed=999)  # a different history: restore must overwrite all of it
    resumed_env.restore(snap)
    assert _obs_equal(resumed_env.observe(), obs_at_snap)
    resumed = _play(resumed_env, np.random.default_rng(2), 60)

    assert sum(1 for x in straight if len(x) == 2) >= 3  # crossed several resets
    assert _same_trace(straight, resumed)


def test_snapshot_is_independent_of_the_env() -> None:
    env = make_tiny_env()
    env.reset(seed=0)
    _play(env, np.random.default_rng(0), 3)
    snap = env.snapshot()
    before = (snap.state, snap.born, snap.deltas, snap.rng.bit_generator.state)
    _play(env, np.random.default_rng(1), 20)
    assert (snap.state, snap.born, snap.deltas, snap.rng.bit_generator.state) == before


def test_restore_rejects_an_instance_that_does_not_fit() -> None:
    big = make_env([dilution([(1, 4), (3, 8)], demand=2, step_limit=12)])
    big.reset()
    small = make_env(tiny_grid(), T_pad=1)
    with pytest.raises(ValueError, match="targets > env T_pad"):
        small.restore(big.snapshot())


@pytest.mark.parametrize("vector", ["sync", "async"])
def test_vector_env_resume_is_exact(vector: str) -> None:
    """A trainer's checkpoint: per-worker snapshots via `get_attr`, restored into new workers
    via `set_attr`, observations rebuilt via `call("observe")` (SAME_STEP autoreset)."""
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    def make() -> Any:
        cls = SyncVectorEnv if vector == "sync" else AsyncVectorEnv
        return cls([make_tiny_env] * 3, autoreset_mode=AutoresetMode.SAME_STEP)

    def run(venv: Any, obs: dict[str, Any], rng: np.random.Generator, steps: int) -> list[Any]:
        trace = []
        for _ in range(steps):
            masks = obs["action_mask"].astype(bool)
            actions = np.array([int(rng.choice(np.flatnonzero(m))) for m in masks])
            obs, reward, terminated, truncated, _ = venv.step(actions)
            trace.append((obs, reward.tolist(), terminated.tolist(), truncated.tolist()))
        return trace

    def stack(obs_list: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

    a, b = make(), make()
    try:
        obs, _ = a.reset(seed=11)
        rng = np.random.default_rng(5)
        obs = run(a, obs, rng, 17)[-1][0]
        snaps = a.get_attr("env_snapshot")
        straight = run(a, obs, np.random.default_rng(6), 40)

        b.reset(seed=12)
        b.set_attr("env_snapshot", list(snaps))
        assert _obs_equal(stack(b.call("observe")), obs)
        resumed = run(b, obs, np.random.default_rng(6), 40)
        assert _same_trace(straight, resumed)
    finally:
        a.close()
        b.close()


def test_unknown_cost_to_finish_is_refused() -> None:
    with pytest.raises(ValueError, match="cost_to_finish"):
        make_env(tiny_grid(), cost_to_finish="optimal")
