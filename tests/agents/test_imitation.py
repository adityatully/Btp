"""Imitation of a teacher, the sharing pass by default (SPEC 11.4, *Imitation warm start*):
demonstrations replay the teacher's plan exactly, and the network can copy it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from meda.agents.base import run_episode
from meda.agents.ppo.imitation import TEACHERS, Demonstrations, imitation_loss, teacher_index
from meda.agents.ppo.networks import NetConfig, SetEncoderNet
from meda.agents.ppo.policy import PPOAgent, batch_obs
from meda.analysis.evaluate import build_env
from meda.core.actions import Mix, Source
from meda.core.costs import CostWeights
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet
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
