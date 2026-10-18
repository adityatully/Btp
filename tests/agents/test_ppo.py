"""PPO with a set encoder (SPEC 11.4, 17.5; milestone M8): network, masking, GAE, trainer,
checkpoint resume (acceptance (c)) and overfitting one tiny instance (acceptance (a))."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from meda.agents.ppo.networks import NetConfig, SetEncoderNet
from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution
from meda.agents.ppo.ppo import PPOConfig, PPOTrainer
from meda.agents.ppo.rollout import RolloutBuffer, compute_gae
from meda.analysis.evaluate import build_env
from meda.core.problem import Instance
from meda.generators.benchmark_io import BenchmarkSet
from meda.train import train_ppo
from meda.utils.checkpoint import load_checkpoint, save_checkpoint
from tests.core.helpers import dilution, load_yaml
from tests.envs.helpers import make_env

TINY = BenchmarkSet.read("tiny_exact_v1")
SMALL_NET = NetConfig(d_model=32, n_layers=2, n_heads=4)


def cfg(l_max: int = 128) -> dict[str, Any]:
    return {
        "env": {**load_yaml("env/sample_prep.yaml"), "L_max": l_max},
        "costs": load_yaml("costs/default.yaml"),
    }


def ppo_cfg(**overrides: Any) -> PPOConfig:
    base = {
        **load_yaml("agent/ppo.yaml"),
        "d_model": 32,
        "n_layers": 2,
        "num_envs": 4,
        "rollout_steps": 32,
        "num_minibatches": 4,
        "total_env_steps": 4 * 32 * 4,
    }
    return PPOConfig.from_dict({**base, **overrides})


def trainer_for(instances: list[Instance], pc: PPOConfig, seed: int = 0) -> PPOTrainer:
    c = cfg()
    envs = SyncVectorEnv(
        [functools.partial(build_env, c, instances)] * pc.num_envs,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    sizes = build_env(c, instances).config.sizes
    return PPOTrainer(pc, sizes, envs, "cpu", seed=seed)


def mid_episode_obs() -> tuple[dict[str, Any], Any]:
    """An observation with droplets on chip and two open targets (M = 8)."""
    env = make_env([dilution([(1, 4), (3, 8)], demand=2, step_limit=12)])
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(2):
        mixes = [i for i, a in enumerate(env.legal) if type(a).__name__ == "Mix"]
        obs, *_ = env.step(np.int64(rng.choice(mixes)))
    return obs, env.config.sizes


# --------------------------------------------------------------------------- network


def test_network_shapes_and_permutation_invariance() -> None:
    torch.manual_seed(0)
    obs, sizes = mid_episode_obs()
    net = SetEncoderNet(sizes, SMALL_NET).eval()
    s, t = int(obs["source_mask"].sum()), int(obs["target_mask"].sum())
    n_cand = int(obs["action_mask"].sum())
    assert s > 2 and t == 2 and n_cand > 10

    p = np.random.default_rng(1).permutation(s)  # new row i holds old row p[i]
    q = np.array([1, 0])
    inv_p, inv_q = np.argsort(p), np.argsort(q)
    shuffled = {k: v.copy() for k, v in obs.items()}
    shuffled["sources"][:s] = obs["sources"][p]
    shuffled["targets"][:t] = obs["targets"][q]
    ci = obs["cand_index"][:n_cand]
    new_ci = shuffled["cand_index"][:n_cand]
    new_ci[:, 1] = inv_p[ci[:, 1]]
    mix, out = ci[:, 0] == 0, ci[:, 0] == 1
    new_ci[mix, 2] = inv_p[ci[mix, 2]]
    new_ci[out, 2] = inv_q[ci[out, 2]]

    with torch.no_grad():
        a = batch_obs({k: v[None] for k, v in obs.items()}, torch.device("cpu"))
        b = batch_obs({k: v[None] for k, v in shuffled.items()}, torch.device("cpu"))
        la, va = net(a)
        lb, vb = net(b)
    assert la.shape == (1, n_cand)  # trimmed to the real candidates
    torch.testing.assert_close(la, lb, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(va, vb, rtol=1e-5, atol=1e-5)


def test_masked_candidates_have_zero_probability() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 6) * 50
    mask = torch.tensor([[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]])
    dist = masked_distribution(logits, mask)
    assert torch.all(dist.probs[mask == 0] == 0)
    torch.testing.assert_close(dist.probs.sum(-1), torch.ones(3))
    samples = dist.sample((2000,))
    assert torch.all(mask.gather(1, samples.T) == 1)
    assert torch.isfinite(dist.entropy()).all()


def test_agent_acts_legally_and_save_load_round_trips(tmp_path: Path) -> None:
    torch.manual_seed(0)
    obs, sizes = mid_episode_obs()
    agent = PPOAgent(SetEncoderNet(sizes, SMALL_NET))
    mask = obs["action_mask"].astype(bool)
    greedy = agent.act(obs, mask, deterministic=True)
    assert mask[greedy]
    agent.save(tmp_path / "agent.pt")
    loaded = PPOAgent.load(tmp_path / "agent.pt")
    assert loaded.act(obs, mask, deterministic=True) == greedy
    assert all(mask[agent.act(obs, mask, deterministic=False)] for _ in range(50))


# --------------------------------------------------------------------------- GAE and buffer


def test_gae_hand_case_with_episode_end_and_cut_off() -> None:
    """One env, 3 steps, gamma = 1, lambda = 0.5. The episode ends after step 1 (no bootstrap
    through it); step 2 starts a new episode that the rollout cuts off (bootstrap V = -4)."""
    rewards = np.array([[-1.0], [-2.0], [-3.0]])
    values = np.array([[-2.5], [-1.5], [-5.0]])
    dones = np.array([[False], [True], [False]])
    adv, ret = compute_gae(rewards, values, dones, np.array([-4.0]), gamma=1.0, lam=0.5)
    d2 = -3.0 + -4.0 - -5.0  # cut off: bootstrap from last value
    d1 = -2.0 - -1.5  # episode ended: no V(s')
    d0 = -1.0 + -1.5 - -2.5
    expected = np.array([[d0 + 0.5 * d1], [d1], [d2]])
    np.testing.assert_allclose(adv, expected)
    np.testing.assert_allclose(ret, expected + values)


def test_rollout_buffer_pads_candidates_to_the_widest_step() -> None:
    buf = RolloutBuffer(num_steps=2, num_envs=1)
    for width in (3, 5):
        mask = np.zeros((1, 10), dtype=np.int8)
        mask[0, :width] = 1
        obs = {
            "action_mask": mask,
            "cand_index": np.where(mask[..., None] == 1, 0, -1).repeat(3, -1).astype(np.int32),
            "candidates": mask[..., None].astype(np.float32).repeat(2, -1),
            "globals": np.zeros((1, 4), dtype=np.float32),
        }
        buf.add(obs, np.array([0]), np.zeros(1), np.zeros(1), np.zeros(1), np.array([False]))
    flat = buf.flat_obs()
    assert flat["action_mask"].shape == (2, 5) and flat["candidates"].shape == (2, 5, 2)
    assert flat["action_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert flat["cand_index"][0, 4].tolist() == [-1, -1, -1]
    assert flat["globals"].shape == (2, 4)


# --------------------------------------------------------------------------- trainer


def test_config_rejects_unknown_keys_and_bad_minibatches() -> None:
    with pytest.raises(ValueError, match="unknown"):
        PPOConfig.from_dict({**load_yaml("agent/ppo.yaml"), "momentum": 0.9})
    with pytest.raises(ValueError, match="divisible"):
        ppo_cfg(num_minibatches=5)


def test_trainer_requires_same_step_autoreset() -> None:
    inst = TINY.by_id("tiny-quarter-m2")
    envs = SyncVectorEnv([functools.partial(build_env, cfg(), [inst])] * 4)  # NEXT_STEP
    with pytest.raises(ValueError, match="SAME_STEP"):
        PPOTrainer(ppo_cfg(), build_env(cfg(), [inst]).config.sizes, envs)


def test_smoke_2000_steps_no_nans_and_value_loss_falls() -> None:
    torch.manual_seed(0)
    trainer = trainer_for([TINY.by_id("tiny-quarter-m2")], ppo_cfg(total_env_steps=2048))
    trainer.start()
    history = []
    while not trainer.done:
        history.append(trainer.iteration())
    assert trainer.global_step == 2048
    for m in history:
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            assert math.isfinite(m[key]), (key, m)
    assert history[-1]["value_loss"] < history[0]["value_loss"]
    assert history[-1]["episodes"] > 0


def test_resume_from_checkpoint_repeats_the_run_exactly(tmp_path: Path) -> None:
    """Acceptance (c): 4 updates straight == 2 updates, save, load into a fresh trainer and
    fresh envs, 2 more updates. Every metric (except wall-clock speed) and every weight."""
    instances = [TINY.by_id("tiny-sharing-m2"), TINY.by_id("tiny-grid-m2-3of8")]
    pc = ppo_cfg()

    def strip(m: dict[str, float]) -> dict[str, float]:
        return {k: v for k, v in m.items() if k != "env_steps_per_second"}

    torch.manual_seed(7)
    straight = trainer_for(instances, pc, seed=3)
    straight.start()
    run_a = [strip(straight.iteration()) for _ in range(4)]

    torch.manual_seed(7)
    first = trainer_for(instances, pc, seed=3)
    first.start()
    run_b = [strip(first.iteration()) for _ in range(2)]
    save_checkpoint(tmp_path / "last.pt", first)
    first.envs.close()

    torch.manual_seed(12345)  # a different global state: the checkpoint must overwrite it
    resumed = trainer_for(instances, pc, seed=99)
    load_checkpoint(tmp_path / "last.pt", resumed)
    assert resumed.update == 2
    run_b += [strip(resumed.iteration()) for _ in range(2)]

    assert run_a == run_b
    for (name, wa), wb in zip(
        straight.net.state_dict().items(), resumed.net.state_dict().values(), strict=True
    ):
        assert torch.equal(wa, wb), name


def test_load_rejects_a_different_config(tmp_path: Path) -> None:
    inst = [TINY.by_id("tiny-quarter-m2")]
    trainer = trainer_for(inst, ppo_cfg())
    trainer.start()
    save_checkpoint(tmp_path / "last.pt", trainer)
    other = trainer_for(inst, ppo_cfg(lr=1e-3))
    with pytest.raises(ValueError, match="lr"):
        load_checkpoint(tmp_path / "last.pt", other)


# --------------------------------------------------------------------------- acceptance (a)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("instance_id", "steps"), [("tiny-quarter-m2", 10_240), ("tiny-sharing-m2", 20_480)]
)
def test_overfits_one_tiny_instance_to_its_optimum(
    instance_id: str, steps: int, tmp_path: Path
) -> None:
    """Acceptance (a): trained on one instance, the greedy plan costs the proved optimum."""
    torch.manual_seed(0)
    config = {
        **cfg(),
        "seed": 0,
        "agent": {
            **load_yaml("agent/ppo.yaml"),
            "num_envs": 8,
            "rollout_steps": 64,
            "num_minibatches": 4,
            "d_model": 64,
            "n_layers": 2,
            "total_env_steps": steps,
        },
        "train": {
            **load_yaml("train/default.yaml"),
            "device": "cpu",
            "instances": [instance_id],
            "checkpoint_every_updates": 0,
            "eval_every_updates": 0,
        },
        "eval": load_yaml("eval/default.yaml"),
    }
    df = train_ppo(config, tmp_path)
    row = df.iloc[0]
    assert row["success"], row["status"]
    assert row["gap_to_optimal"] == 0, (row["cost_exact"], row["reference_cost"])
