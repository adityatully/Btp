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
from meda.core.composition import ZERO, Composition
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.envs.encoding import TYPE_DROPLET, TYPE_FLUID, TYPE_TARGET
from meda.generators.benchmark_io import BenchmarkSet
from meda.train import train_ppo
from meda.utils.checkpoint import load_checkpoint, save_checkpoint
from tests.core.helpers import SRB, chip, dilution, load_yaml, mixture
from tests.envs.helpers import make_env

TINY = BenchmarkSet.read("tiny_exact_v1")
SMALL_NET = NetConfig(d_model=32, n_layers=2, n_heads=4)


def cfg(l_pad: int = 128) -> dict[str, Any]:
    return {
        "env": {**load_yaml("env/sample_prep.yaml"), "L_pad": l_pad},
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


def mid_episode_obs(inst: Instance | None = None, **env_overrides: Any) -> dict[str, Any]:
    """An observation with droplets on chip and two open targets over three fluids (M = 8)."""
    inst = inst or mixture([(1, 1, 2), (1, 2, 1)], demand=2, step_limit=14)
    env = make_env([inst], **{"L_pad": 2048, **env_overrides})
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(2):
        mixes = [i for i, a in enumerate(env.legal) if type(a).__name__ == "Mix"]
        obs, *_ = env.step(np.int64(rng.choice(mixes)))
    return obs


def permuted(obs: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """`obs` with its fluid, droplet and target rows shuffled (row i holds old row p[i]) and
    every reference to them (composition columns, pairs, operands) remapped to match."""
    nf, nd, nt = (int(obs[k].sum()) for k in ("fluid_mask", "droplet_mask", "target_mask"))
    nc = int(obs["action_mask"].sum())
    pf, pd, pt = rng.permutation(nf), rng.permutation(nd), rng.permutation(nt)
    out = {k: v.copy() for k, v in obs.items()}
    out["fluids"][:nf] = obs["fluids"][pf]
    out["droplets"][:nd] = obs["droplets"][pd]
    out["targets"][:nt] = obs["targets"][pt]
    out["droplet_comp"][:nd, :nf] = obs["droplet_comp"][pd][:, pf]
    out["target_comp"][:nt, :nf] = obs["target_comp"][pt][:, pf]
    out["droplet_target"][:nd, :nt] = obs["droplet_target"][pd][:, pt]
    out["cand_comp"][:nc, :nf] = obs["cand_comp"][:nc][:, pf]
    out["cand_target"][:nc, :nt] = obs["cand_target"][:nc][:, pt]
    inverse = {
        TYPE_FLUID: np.argsort(pf),
        TYPE_DROPLET: np.argsort(pd),
        TYPE_TARGET: np.argsort(pt),
    }
    ci, new = obs["cand_index"][:nc], out["cand_index"][:nc]
    for typ_col, row_col in ((1, 2), (3, 4)):
        for typ, inv in inverse.items():
            hit = ci[:, typ_col] == typ
            new[hit, row_col] = inv[ci[hit, row_col]]
    return out


def forward(net: SetEncoderNet, *observations: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    batched = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
    with torch.no_grad():
        return net(batch_obs(batched, torch.device("cpu")))


# --------------------------------------------------------------------------- network


def test_network_shapes_and_permutation_invariance() -> None:
    """Logits and value unchanged when fluids, droplets and targets are reordered (17.5)."""
    torch.manual_seed(0)
    obs = mid_episode_obs()
    nf, nd, nt = (int(obs[k].sum()) for k in ("fluid_mask", "droplet_mask", "target_mask"))
    n_cand = int(obs["action_mask"].sum())
    assert nf == 3 and nd >= 2 and nt == 2 and n_cand > 10
    net = SetEncoderNet(SMALL_NET).eval()
    la, va = forward(net, obs)
    assert la.shape == (1, n_cand)  # trimmed to the real candidates
    for seed in range(3):
        lb, vb = forward(net, permuted(obs, np.random.default_rng(seed)))
        torch.testing.assert_close(la, lb, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(va, vb, rtol=1e-5, atol=1e-5)


def test_one_network_reads_any_fluids_targets_m_storage_and_padding() -> None:
    """M8 acceptance (d): the same weights read instances that differ in F, T, M and storage
    capacity, at any padding size, alone or batched together."""
    torch.manual_seed(0)
    four = (*SRB[:2], Fluid(2, "R2", FluidKind.REAGENT), Fluid(3, "B", FluidKind.BUFFER))
    four_targets = tuple(
        Target(i, i, Composition.from_ratio(r), ZERO, 1, 1)
        for i, r in enumerate([(1, 1, 1, 1), (2, 1, 0, 1), (1, 0, 2, 1)])
    )
    instances = [
        dilution([(1, 4)], chip_limits=chip("dmfb_m2"), step_limit=10),  # F 2, T 1, M 2
        mixture([(1, 1, 2), (1, 2, 1)], demand=2, step_limit=14),  # F 3, T 2, M 8
        Instance(
            "test-four", four, four_targets, chip(max_mix_volume=4, storage_capacity=3), 20
        ),  # F 4, T 3, M 4, S 3
    ]
    net = SetEncoderNet(SMALL_NET).eval()
    small = [mid_episode_obs(i) for i in instances]
    big = [mid_episode_obs(i, F_pad=9, T_pad=30, N_pad=16, L_pad=4000) for i in instances]
    alone = [forward(net, o) for o in small]
    for (la, va), o in zip(alone, big, strict=True):
        lb, vb = forward(net, o)  # same instance, larger padding: identical after trimming
        torch.testing.assert_close(la, lb)
        torch.testing.assert_close(va, vb)
    logits, values = forward(net, *small)  # one batch mixing all three
    for i, (la, va) in enumerate(alone):
        n = la.shape[1]
        torch.testing.assert_close(logits[i : i + 1, :n], la, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(values[i : i + 1], va, rtol=1e-5, atol=1e-5)
    # No parameter's shape mentions a count: the network never saw the padding sizes.
    assert not any(9 in p.shape or 30 in p.shape for p in net.parameters())


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
    obs = mid_episode_obs()
    agent = PPOAgent(SetEncoderNet(SMALL_NET))
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


def test_rollout_buffer_pads_every_token_axis_to_the_widest_in_the_batch() -> None:
    """Observations are stored trimmed; a batch pads each axis to its widest (operands -1)."""
    env = make_env([mixture([(1, 1, 2), (1, 2, 1)], demand=2, step_limit=14)], L_pad=2048)
    first, _ = env.reset(seed=0)  # empty chip
    later = mid_episode_obs()  # droplets on chip, more candidates
    buf = RolloutBuffer(num_steps=2, num_envs=1)
    for obs in (first, later):
        batched = {k: v[None] for k, v in obs.items()}
        buf.add(batched, np.array([0]), np.zeros(1), np.zeros(1), np.zeros(1), np.array([False]))
    flat = buf.flat_obs()
    widths = {k: int(later[k].sum()) for k in ("droplet_mask", "action_mask")}
    assert flat["droplets"].shape[1] == flat["droplet_comp"].shape[1] == widths["droplet_mask"]
    assert flat["droplet_target"].shape[1] == widths["droplet_mask"]
    assert flat["candidates"].shape[1] == flat["cand_target"].shape[1] == widths["action_mask"]
    n0 = int(first["action_mask"].sum())
    assert (flat["cand_index"][0, n0:] == -1).all() and (flat["action_mask"][0, n0:] == 0).all()
    assert flat["fluids"].shape == (2, 3, 3) and flat["globals"].shape == (2, 7)
    # A minibatch pads only to its own widest observation, not the rollout's.
    alone = buf.minibatch_obs(np.array([0]))
    assert alone["candidates"].shape[1] == n0 < widths["action_mask"]
    assert alone["droplets"].shape[1] == 1  # empty chip: one (masked) row


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

    def strip(m: dict[str, float]) -> dict[str, Any]:
        # NaN (no successful episode yet) is written as None so equal runs compare equal.
        return {
            k: None if math.isnan(v) else v
            for k, v in m.items()
            if k != "env_steps_per_second" and not k.startswith("seconds_")  # wall clock
        }

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


def test_trains_from_a_generator_and_never_draws_a_held_out_problem(tmp_path: Path) -> None:
    """`train.source=generator`: every episode is a fresh draw from the stage, none of them a
    problem of the frozen sets, and the benchmark is used only to grade the result."""
    from meda.generators.benchmark_io import BenchmarkSet, problem_hash
    from meda.train import training_source

    config = {
        **cfg(),
        "seed": 0,
        "chip": load_yaml("chip/dmfb_m2.yaml"),
        "generator": load_yaml("generator/tiny_dilution.yaml"),
        "agent": {
            **load_yaml("agent/ppo.yaml"),
            "num_envs": 2,
            "rollout_steps": 16,
            "num_minibatches": 2,
            "d_model": 32,
            "n_layers": 1,
            "total_env_steps": 64,
        },
        "train": {
            **load_yaml("train/default.yaml"),
            "source": "generator",
            "benchmark": "val_stage1_v2",
            "device": "cpu",
            "checkpoint_every_updates": 0,
            "eval_every_updates": 0,
        },
        "eval": load_yaml("eval/default.yaml"),
    }
    config["generator"].pop("defaults", None)
    bench = BenchmarkSet.read("val_stage1_v2")
    source = training_source(config, bench)
    assert source is not None
    held = {
        problem_hash(i)
        for name in ("tiny_exact_v1", "val_stage1_v2")
        for i in BenchmarkSet.read(name).instances
    }
    rng = np.random.default_rng(0)
    for _ in range(40):
        assert problem_hash(source(rng)) not in held
    df = train_ppo(config, tmp_path)
    assert len(df) == len(bench.instances) and (df["env_steps"] == 64).all()


def test_reward_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="reward_scale"):
        ppo_cfg(reward_scale=0.0)
    assert ppo_cfg(reward_scale=1 / 64).reward_scale == 1 / 64


@pytest.mark.slow  # about 50 s: four rounds of grading on the stage sets
def test_trains_under_a_regime_promotes_and_resumes_its_shares(tmp_path: Path) -> None:
    """`train.source=regime`: draws come from the design's cells at the regime's shares, each
    validation round may promote (here forced after one round), the new shares reach the env
    workers, and a resumed run restores the regime and its shares."""
    import json

    from meda.train import train_ppo_regime

    def config(out: Path, resume: Path | None = None) -> dict[str, Any]:
        return {
            **cfg(l_pad=4096),
            "costs": load_yaml("costs/geometric.yaml"),
            "seed": 0,
            "stages": load_yaml("stages/track2_v1.yaml"),
            "regime": {
                **load_yaml("regime/expanding.yaml"),
                "num_stages": 2,
                "max_stage_fraction": 0.01,
            },
            "agent": {
                **load_yaml("agent/ppo.yaml"),
                "num_envs": 2,
                "rollout_steps": 32,
                "num_minibatches": 2,
                "d_model": 32,
                "n_layers": 1,
                "reward_scale": 0.01,
                "total_env_steps": 128,
            },
            "train": {
                **load_yaml("train/default.yaml"),
                "source": "regime",
                "device": "cpu",
                "checkpoint_every_updates": 1,
                "eval_every_updates": 1,
                "resume": resume,
            },
            "eval": load_yaml("eval/default.yaml"),
        }

    first = tmp_path / "first"
    df = train_ppo_regime(config(first), first)
    assert set(df["stage"]) == {"s1_dilution", "s2_mixture", "held_out"}
    state = json.loads((first / "regime.json").read_text())
    assert [(e["stage"], e["event"], e["forced"]) for e in state["events"]] == [
        ("s1_dilution", "promoted", True),
        ("s2_mixture", "finished", True),
    ]
    shares = [
        {k: v for k, v in json.loads(line).items() if k not in ("prefix", "step")}
        for line in (first / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["prefix"] == "regime/weight"
    ]
    assert shares[0] == {"s1_dilution": 1.0}
    assert shares[1] == {"s1_dilution": 0.5, "s2_mixture": 0.5}
    assert shares[2] == pytest.approx({"s1_dilution": 0.5, "s2_mixture": 0.5})  # final 0.05:0.05

    # Resuming restores the regime (a mismatch would raise) and the run's step count.
    second = tmp_path / "second"
    df = train_ppo_regime(config(second, resume=first / "checkpoints" / "last.pt"), second)
    assert json.loads((second / "regime.json").read_text()) == state
    assert (df["env_steps"] == 128).all()


def test_chunked_update_matches_the_unchunked_one() -> None:
    """`max_chunk_rows` bounds memory only: one update with every observation in its own chunk
    gives the same weights as one with each minibatch in a single forward pass."""
    instances = [TINY.by_id("tiny-sharing-m2")]
    nets = []
    for rows in (1, 10**9):
        torch.manual_seed(0)
        trainer = trainer_for(instances, ppo_cfg(max_chunk_rows=rows, total_env_steps=4 * 32))
        trainer.start()
        metrics = trainer.iteration()
        assert metrics["seconds_update"] > 0 and metrics["seconds_env"] > 0
        nets.append(trainer.net.state_dict())
    for key in nets[0]:
        assert torch.allclose(nets[0][key], nets[1][key], atol=1e-5), key
    with pytest.raises(ValueError, match="max_chunk_rows"):
        ppo_cfg(max_chunk_rows=0)
