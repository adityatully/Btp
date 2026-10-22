"""The workers collector (`agents.ppo.collector.ActorPool`, `train.collector=workers`)."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from meda.agents.ppo.collector import ActorPool
from meda.agents.ppo.ppo import PPOConfig, PPOTrainer
from meda.agents.ppo.rollout import compute_gae, split_obs
from meda.analysis.evaluate import build_env
from meda.core.problem import Instance
from meda.envs.sample_prep_env import FixedInstances, SamplePrepEnv
from meda.utils.checkpoint import load_checkpoint, save_checkpoint
from tests.agents.test_ppo import TINY, cfg, ppo_cfg

INSTANCES = [TINY.by_id("tiny-sharing-m2"), TINY.by_id("tiny-grid-m2-3of8")]


def pool_trainer(instances: list[Instance], pc: PPOConfig, seed: int = 0) -> PPOTrainer:
    c = cfg()
    pool = ActorPool([functools.partial(build_env, c, instances)] * pc.num_envs, pc.net)
    return PPOTrainer(pc, build_env(c, instances).config.sizes, pool, "cpu", seed=seed)


def strip(m: dict[str, float]) -> dict[str, Any]:
    """Metrics without wall-clock ones; NaN as None so equal runs compare equal."""
    return {
        k: None if math.isnan(v) else v
        for k, v in m.items()
        if k != "env_steps_per_second" and not k.startswith("seconds_")
    }


def broken_env() -> SamplePrepEnv:
    raise ValueError("boom: this env cannot be built")


def test_same_seed_same_run_and_same_weights() -> None:
    pc = ppo_cfg(num_envs=2, total_env_steps=2 * 32 * 2)
    runs, nets = [], []
    for _ in range(2):
        torch.manual_seed(5)
        trainer = pool_trainer(INSTANCES, pc, seed=3)
        try:
            trainer.start()
            runs.append([strip(trainer.iteration()) for _ in range(2)])
            nets.append(trainer.net.state_dict())
        finally:
            trainer.envs.close()
    assert runs[0] == runs[1]
    for key in nets[0]:
        assert torch.equal(nets[0][key], nets[1][key]), key


def test_rollout_matches_the_lockstep_format_and_first_observations() -> None:
    """Env i starts from the same reset (seed + i) under both collectors, so the first
    observations kept are identical; every kept observation is trimmed to its legal count."""
    pc = ppo_cfg(num_envs=2)
    c = cfg()
    lockstep = SyncVectorEnv(
        [functools.partial(build_env, c, INSTANCES)] * 2, autoreset_mode=AutoresetMode.SAME_STEP
    )
    first, _ = lockstep.reset(seed=3)
    expected = split_obs(first)
    trainer = pool_trainer(INSTANCES, pc, seed=3)
    try:
        trainer.start()
        buf, last, episodes, legal, _ = trainer._collect_workers()
    finally:
        trainer.envs.close()
    for got, want in zip(buf.obs[:2], expected, strict=True):
        assert got.keys() == want.keys()
        for k in want:
            assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), k
    for o in buf.obs:
        assert o["action_mask"].all() and o["action_mask"].shape[1] == o["candidates"].shape[1]
    assert buf.steps_added == pc.rollout_steps and last.shape == (2,)
    assert trainer.global_step == pc.batch_size and len(legal) == pc.rollout_steps
    assert len(episodes) == int(buf.dones.sum())
    assert np.all(buf.logprobs <= 0) and np.all(np.isfinite(buf.values))


def test_resume_from_checkpoint_repeats_the_run_exactly(tmp_path: Path) -> None:
    """As the lockstep acceptance (c): weights, env snapshots and every env's sampling
    generator come back, so 2 + 2 updates equal 4 straight."""
    pc = ppo_cfg(num_envs=2, total_env_steps=2 * 32 * 4)
    torch.manual_seed(7)
    straight = pool_trainer(INSTANCES, pc, seed=3)
    straight.start()
    run_a = [strip(straight.iteration()) for _ in range(4)]
    straight.envs.close()

    torch.manual_seed(7)
    first = pool_trainer(INSTANCES, pc, seed=3)
    first.start()
    run_b = [strip(first.iteration()) for _ in range(2)]
    save_checkpoint(tmp_path / "last.pt", first)
    first.envs.close()

    torch.manual_seed(12345)
    resumed = pool_trainer(INSTANCES, pc, seed=99)
    try:
        load_checkpoint(tmp_path / "last.pt", resumed)
        run_b += [strip(resumed.iteration()) for _ in range(2)]
    finally:
        resumed.envs.close()
    assert run_a == run_b
    for (name, wa), wb in zip(
        straight.net.state_dict().items(), resumed.net.state_dict().values(), strict=True
    ):
        assert torch.equal(wa, wb), name


def test_a_checkpoint_resumes_only_with_its_own_collector(tmp_path: Path) -> None:
    pc = ppo_cfg(num_envs=2)
    trainer = pool_trainer(INSTANCES, pc)
    trainer.start()
    save_checkpoint(tmp_path / "last.pt", trainer)
    trainer.envs.close()
    lockstep = PPOTrainer(
        pc,
        build_env(cfg(), INSTANCES).config.sizes,
        SyncVectorEnv(
            [functools.partial(build_env, cfg(), INSTANCES)] * 2,
            autoreset_mode=AutoresetMode.SAME_STEP,
        ),
        "cpu",
    )
    with pytest.raises(ValueError, match=r"train\.collector=workers"):
        load_checkpoint(tmp_path / "last.pt", lockstep)


def test_the_learner_refuses_a_rollout_from_other_weights() -> None:
    """The first minibatch is checked against what the workers collected: with untouched
    weights they agree to rounding; weights changed after collecting are caught."""
    pc = ppo_cfg(num_envs=2)
    trainer = pool_trainer(INSTANCES, pc)
    try:
        trainer.start()
        metrics = trainer.iteration()
        assert metrics["collected_logprob_max_diff"] < 1e-5
        assert metrics["collected_value_max_diff"] < 1e-5
        buf, last, _, _, _ = trainer._collect_workers()
        with torch.no_grad():
            for p in trainer.net.parameters():
                p.add_(0.5)
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, last, pc.gamma, pc.gae_lambda)
        with pytest.raises(RuntimeError, match="not the one being updated"):
            trainer._optimise(buf, adv, ret)

        # A rollout stamped with another version than the one sent is refused.
        pool = trainer.envs
        assert isinstance(pool, ActorPool)
        real = pool._all

        def stale(cmd: str, args: Any) -> list[Any]:
            out = real(cmd, args)
            for r in out:
                r.version -= 1
            return out

        pool._all = stale  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match=r"returned version 6 .* expected version 7"):
            pool.rollout(trainer.net.state_dict(), 7, 4)
        pool._all = real  # type: ignore[method-assign]
    finally:
        trainer.envs.close()


def test_a_failing_worker_stops_the_pool_with_its_error() -> None:
    with pytest.raises(RuntimeError, match="boom: this env cannot be built"):
        ActorPool([broken_env, functools.partial(build_env, cfg(), INSTANCES)], ppo_cfg().net)


def test_close_leaves_no_worker_processes() -> None:
    pc = ppo_cfg(num_envs=2)
    for terminate in (False, True):
        trainer = pool_trainer(INSTANCES, pc)
        trainer.start()
        pool = trainer.envs
        assert isinstance(pool, ActorPool)
        procs = list(pool._procs)
        pool.close(terminate=terminate)
        assert not any(p.is_alive() for p in procs)
        with pytest.raises(RuntimeError, match="closed"):
            pool.reset(0)


def test_learns_like_the_lockstep_collector() -> None:
    """The smoke test of the lockstep trainer, with workers: finite losses, falling value
    loss, finished episodes."""
    torch.manual_seed(0)
    pc = ppo_cfg(total_env_steps=2048)
    trainer = pool_trainer([TINY.by_id("tiny-quarter-m2")], pc)
    try:
        trainer.start()
        history = []
        while not trainer.done:
            history.append(trainer.iteration())
    finally:
        trainer.envs.close()
    assert trainer.global_step == 2048
    for m in history:
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            assert math.isfinite(m[key]), (key, m)
    assert history[-1]["value_loss"] < history[0]["value_loss"]
    assert history[-1]["episodes"] > 0


def test_a_new_instance_source_reaches_every_worker() -> None:
    """What the regime does on a promotion: episodes started after the call draw from the new
    source. Only calls that leave the episode alone are allowed through the pool."""
    other = TINY.by_id("tiny-quarter-m2")
    pc = ppo_cfg(num_envs=2)
    trainer = pool_trainer(INSTANCES, pc)
    try:
        trainer.start()
        pool = trainer.envs
        assert isinstance(pool, ActorPool)
        pool.call("set_instance_source", FixedInstances([other]))
        _, _, episodes, _, _ = trainer._collect_workers()
        assert episodes  # every env finished at least one episode and reset
        assert {i.instance_id for i in pool.get_attr("instance")} == {other.instance_id}
        with pytest.raises(RuntimeError, match="cannot be called through the pool"):
            pool.call("reset")
    finally:
        trainer.envs.close()
