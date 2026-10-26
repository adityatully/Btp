"""Imitation of a teacher, the sharing pass by default (SPEC 11.4, *Imitation warm start*):
demonstrations replay the teacher's plan exactly, and the network can copy it."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from meda.agents.base import run_episode
from meda.agents.ppo.imitation import TEACHERS, Demonstrations, imitation_loss, teacher_index
from meda.agents.ppo.networks import NetConfig, SetEncoderNet
from meda.agents.ppo.policy import PPOAgent, batch_obs
from meda.agents.ppo.ppo import PPOConfig, PPOTrainer
from meda.analysis.evaluate import build_env
from meda.core.actions import Mix, Source
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.train import load_initial_weights
from tests.core.helpers import load_yaml

CFG = {
    "env": {**load_yaml("env/sample_prep.yaml"), "L_pad": 2048},
    "costs": load_yaml("costs/geometric.yaml"),
}
WEIGHTS = CostWeights.from_dict(CFG["costs"])


def test_teacher_index_matches_a_mix_up_to_operand_order() -> None:
    r0, d3 = Source.reservoir(0), Source.droplet(3)
    legal = (Mix(d3, 1, r0, 2), Mix(d3, 1, r0, 1))
    assert teacher_index(legal, Mix(r0, 1, d3, 1), "x") == 1
    assert teacher_index(legal, Mix(d3, 1, r0, 2), "x") == 0
    with pytest.raises(ValueError, match=r"'x'.*not a candidate"):
        teacher_index(legal, Mix(r0, 2, d3, 2), "x")


@pytest.mark.parametrize("teacher", sorted(TEACHERS))
@pytest.mark.parametrize("name", ["val_s1_v2", "val_s2_v2"])
def test_demonstrations_replay_the_teachers_plan_at_its_cost(name: str, teacher: str) -> None:
    bench = BenchmarkSet.read(name)
    env = build_env(CFG, list(bench.instances))
    for inst in bench.instances[:10]:
        demos = Demonstrations()
        recorded = demos.add_episode(env, inst, WEIGHTS, TEACHERS[teacher])
        plan = TEACHERS[teacher](inst, WEIGHTS)
        assert plan is not None and len(demos) == len(plan)
        _, totals, _ = replay(inst, plan)
        cost = WEIGHTS.scalar(totals, inst.fluids)
        assert recorded == cost
        assert cost <= bench.reference[inst.instance_id].cost  # never dearer than best known
        env.reset(options={"instance": inst})
        for index in demos.actions:
            *_, info = env.step(index)
        assert info["success"] and info["plan_cost"] == cost


def test_network_copies_the_teacher_on_its_training_problems() -> None:
    """A small network trained by cross-entropy agrees with every teacher move and its greedy
    plan succeeds at the teacher's cost."""
    torch.manual_seed(0)
    bench = BenchmarkSet.read("val_s1_v2")
    instances = list(bench.instances[:3])
    env = build_env(CFG, instances)
    demos = Demonstrations()
    costs = [demos.add_episode(env, inst, WEIGHTS) for inst in instances]
    net = SetEncoderNet(NetConfig(d_model=32, n_layers=2, n_heads=4))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    obs, actions = demos.minibatch(np.arange(len(demos)))
    t, a = batch_obs(obs, torch.device("cpu")), torch.as_tensor(actions)
    for _ in range(1000):
        optimizer.zero_grad()
        loss, agree = imitation_loss(net, t, a)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        if agree.item() == 1.0 and loss.item() < 0.05:
            break
    assert agree.item() == 1.0
    agent = PPOAgent(net)
    for inst, cost in zip(instances, costs, strict=True):
        result = run_episode(agent, env, inst, deterministic=True)
        assert result.cost == cost


# ---------------------------------------------------------------- warm start inside PPO


def small_ppo(**overrides: object) -> PPOConfig:
    base = {
        **load_yaml("agent/ppo.yaml"),
        "d_model": 32,
        "n_layers": 2,
        "num_envs": 2,
        "rollout_steps": 16,
        "num_minibatches": 2,
        "total_env_steps": 2 * 16 * 3,
    }
    return PPOConfig.from_dict({**base, **overrides})


def small_trainer(pc: PPOConfig, instances: list[Instance]) -> PPOTrainer:
    envs = SyncVectorEnv(
        [functools.partial(build_env, CFG, instances)] * pc.num_envs,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return PPOTrainer(pc, build_env(CFG, instances).config.sizes, envs, "cpu", seed=0)


def test_imitation_config_is_checked() -> None:
    with pytest.raises(ValueError, match="imitation_coef must be >= 0"):
        small_ppo(imitation_coef=-1.0)
    with pytest.raises(ValueError, match="imitation_steps"):
        small_ppo(imitation_coef=1.0, imitation_steps=0)


def test_imitation_weight_falls_linearly_to_zero() -> None:
    instances = list(BenchmarkSet.read("val_s1_v2").instances[:2])
    trainer = small_trainer(small_ppo(imitation_coef=2.0, imitation_steps=100), instances)
    for step, expected in ((0, 2.0), (25, 1.5), (100, 0.0), (500, 0.0)):
        trainer.global_step = step
        assert trainer.imitation_weight() == pytest.approx(expected)


def test_imitation_term_is_logged_and_needs_demonstrations() -> None:
    torch.manual_seed(0)
    instances = list(BenchmarkSet.read("val_s1_v2").instances[:2])
    trainer = small_trainer(small_ppo(imitation_coef=1.0, imitation_steps=10_000), instances)
    trainer.start()
    with pytest.raises(RuntimeError, match="no demonstrations"):
        trainer.iteration()
    demos = Demonstrations()
    env = build_env(CFG, instances)
    for inst in instances:
        demos.add_episode(env, inst, WEIGHTS)
    trainer.set_demonstrations(demos)
    metrics = trainer.iteration()
    assert 0 < metrics["imitation_coef"] <= 1
    assert np.isfinite(metrics["imitation_loss"]) and 0 <= metrics["imitation_agree"] <= 1
    off = small_trainer(small_ppo(), instances)
    off.start()
    stats = off.iteration()
    assert stats["imitation_coef"] == 0 and "imitation_loss" not in stats


def test_init_from_loads_the_imitation_network(tmp_path: Path) -> None:
    instances = list(BenchmarkSet.read("val_s1_v2").instances[:2])
    pc = small_ppo()
    source = SetEncoderNet(pc.net)
    PPOAgent(source).save(tmp_path / "imitation.pt")
    trainer = small_trainer(pc, instances)
    load_initial_weights(trainer, tmp_path / "imitation.pt")
    for a, b in zip(source.state_dict().values(), trainer.net.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    other = small_trainer(small_ppo(d_model=64), instances)
    with pytest.raises(ValueError, match="differs from this run"):
        load_initial_weights(other, tmp_path / "imitation.pt")
