"""Imitation-only copy test (SPEC 11.4, *Imitation warm start*; paper log D-31). No RL.

Can the PPO network copy the teacher (`teacher=share`, the sharing pass, or
`teacher=construction`) on problems it has not seen?
Demonstrations come from fresh stage 1-2 problems of the stage design (never a validation or
held-out problem); the network is trained by cross-entropy on the teacher's candidate index.
Each round reports, per stage: move agreement and loss on the validation problems' teacher
steps (unseen states), and the greedy policy's success and gap to best known on `val_s1_v2` and
`val_s2_v2`. Best known there is `cheapest_construction`'s cost; the sharing pass is often
cheaper, so a perfect copy of it scores 100% success at a gap at or below 0 (the teacher's own
gap is printed for reference).

Usage:
    uv run python scripts/imitation_check.py problems=2000 epochs=20 out=outputs/imitation/s12

Arguments are key=value. Expert iteration (D-38): `demos=<demos.jsonl> arm=search|planner
init=<agent.pt>` trains on the practice problems of `scripts/expert_iteration.py` with that
arm's plans, starting from the given weights; validation is unchanged. Round 2 (D-40):
`arm=visits` trains on what the search observed (`samples=1` demos): the policy toward the
visit counts (cross-entropy with the normalised counts) and, with `value_coef` > 0, the value
head toward -(remaining cost of the cheapest plan seen) x `reward_scale`, the PPO value's own
scale (gamma 1). The last 5% of the practice problems are held out and report the value's mean
absolute error in cost units and how often the top move is the search's most visited.

Writes `metrics.jsonl` (one line per round) and `imitation.pt` (the network, loadable with
`PPOAgent.load`, the starting point of the PPO warm start) into `out`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import OmegaConf

from meda.agents.ppo.imitation import TEACHERS, Demonstrations, imitation_loss
from meda.agents.ppo.networks import SetEncoderNet
from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution
from meda.agents.ppo.ppo import PPOConfig, chunk_by_width
from meda.agents.ppo.rollout import pad_obs, split_obs
from meda.analysis.dag_metrics import ScheduleConfig
from meda.analysis.evaluate import AgentProducer, build_env, evaluate
from meda.core.chip import ChipLimits
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.envs.encoding import OP_MIX
from meda.generators.benchmark_io import BenchmarkSet
from meda.generators.stages import StageDesign
from meda.utils.seeding import seed_everything

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULTS: dict[str, Any] = {
    "stages": 2,  # the first n stages of the design (D-31: the teacher is an expert on 1-2)
    "problems": 2000,  # training problems, drawn evenly over the stages' cells
    "epochs": 20,
    "eval_every": 2,  # epochs between greedy gradings on the validation sets
    "batch": 64,
    "lr": 3e-4,
    "max_chunk_rows": 32768,
    "seed": 0,
    "device": "cpu",
    "threads": 1,
    "out": "outputs/imitation/run",
    "teacher": "share",
    "hints": "planner",  # the env's `hint_source`: planner or problem (planner-free, Q-105)
    "init": "",  # start from these weights (an agent.pt) instead of a fresh network
    "demos": "",  # practice problems and plans from `expert_iteration.py` instead of draws
    "arm": "search",  # with `demos`: the search's plans, the planner's (D-38) or visits (D-40)
    "value_coef": 0.0,  # arm=visits: weight of the value loss (0 = policy only, the control)
    "reward_scale": 0.01,  # the PPO reward scale the value head was trained with (E-22)
}


def load(rel: str) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(CONFIGS / rel))
    assert isinstance(data, dict)
    return data  # type: ignore[return-value]


def parse(argv: list[str]) -> dict[str, Any]:
    args = dict(DEFAULTS)
    for item in argv:
        key, _, value = item.partition("=")
        if key not in args:
            raise SystemExit(f"unknown argument '{key}'; known: {sorted(args)}")
        args[key] = type(DEFAULTS[key])(value)
    return args


@dataclass
class SearchTargets:
    """Positions the search observed (D-40): trimmed observations, the normalised visit counts
    (None where too few) and the value target (nan where none)."""

    obs: list[dict[str, Any]] = field(default_factory=list)
    pi: list[npt.NDArray[np.float64] | None] = field(default_factory=list)
    value: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def add_problem(
        self, env: Any, inst: Instance, samples: Sequence[Any], reward_scale: float
    ) -> None:
        """Replay each sample's moves from the start, along a prefix tree so every move is
        played once; the env's candidate order is deterministic, so the visit counts line up."""
        trie: dict[Any, Any] = {}
        for actions, visits, remaining in samples:
            node = trie
            for a in actions:
                node = node.setdefault(a, {})
            node[None] = (visits, remaining)
        obs, _ = env.reset(options={"instance": inst})
        stack = [(trie, env.snapshot(), obs)]
        while stack:
            node, snap, obs = stack.pop()
            if None in node:
                visits, remaining = node[None]
                if visits is not None and len(visits) != len(snap.legal):
                    raise RuntimeError(f"{inst.instance_id}: visit counts do not fit candidates")
                self.obs.extend(split_obs({k: v[None] for k, v in obs.items()}))
                v = np.asarray(visits, dtype=np.float64) if visits is not None else None
                self.pi.append(None if v is None else v / v.sum())
                self.value.append(-remaining * reward_scale)
            for action, child in node.items():
                if action is None:
                    continue
                env.restore(snap)
                nxt, _, _, _, _ = env.step(env.legal.index(action))
                stack.append((child, env.snapshot(), nxt))

    def minibatch(
        self, index: npt.NDArray[np.int64]
    ) -> tuple[dict[str, Any], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        obs = pad_obs([self.obs[int(i)] for i in index])
        pi = np.zeros(obs["action_mask"].shape, dtype=np.float64)
        for row, i in enumerate(index):
            p = self.pi[int(i)]
            if p is not None:
                pi[row, : len(p)] = p
        return obs, pi, np.array([self.value[int(i)] for i in index])

    def candidate_widths(self) -> npt.NDArray[np.int64]:
        return np.array([o["action_mask"].shape[1] for o in self.obs], dtype=np.int64)


def search_loss(
    net: SetEncoderNet, t: Any, pi: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summed policy cross-entropy over rows with visit counts, summed squared value error over
    rows with a target, top-move agreement count and absolute value error sum (target units)."""
    logits, v = net(t)
    logp = masked_distribution(logits, t["action_mask"]).logits
    has_pi = pi.sum(-1) > 0
    ce = -torch.where(pi > 0, pi * logp, torch.zeros_like(logp)).sum(-1)
    has_v = ~torch.isnan(value)
    err = torch.where(has_v, v - torch.nan_to_num(value), torch.zeros_like(v))
    agree = ((logp.argmax(-1) == pi.argmax(-1)) & has_pi).sum()
    return ce[has_pi].sum(), (err**2).sum(), agree, err.abs().sum()


def main(argv: list[str]) -> None:
    args = parse(argv)
    torch.set_num_threads(args["threads"])
    seed_everything(args["seed"])
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)

    design = StageDesign.from_dict(load("stages/track2_v1.yaml")).first(args["stages"])
    cfg = {"costs": load(f"costs/{design.costs}.yaml"), "env": load("env/sample_prep.yaml")}
    cfg["env"]["hint_source"] = args["hints"]
    weights = CostWeights.from_dict(cfg["costs"])
    chip = ChipLimits.from_dict(load(f"chip/{design.chip}.yaml"))
    val_sets = {s.name: BenchmarkSet.read(s.val) for s in design.stages}
    held = set(BenchmarkSet.read(design.held_out_val).hashes())
    for bench in val_sets.values():
        held |= bench.hashes()
    shares = {s.name: 1 / len(design.stages) for s in design.stages}
    source = design.source(shares, chip, weights, frozenset(held))

    started = time.perf_counter()
    rng = np.random.default_rng(args["seed"])
    env = build_env(cfg, [], source)
    teacher = TEACHERS[args["teacher"]]
    train = Demonstrations()
    visits = args["arm"] == "visits"
    targets, held_out = SearchTargets(), SearchTargets()
    if visits:
        if not args["demos"]:
            raise SystemExit("arm=visits needs demos=<demos.jsonl> written with samples=1")
        sys.path.insert(0, str(CONFIGS.parent / "scripts"))
        from expert_iteration import load_targets

        loaded = load_targets(args["demos"])
        cut = len(loaded) - max(1, len(loaded) // 20)
        for idx, (inst, samples) in enumerate(loaded):
            store = targets if idx < cut else held_out
            store.add_problem(env, inst, samples, args["reward_scale"])
        print(
            f"{len(targets)} search positions from {cut} problems "
            f"({sum(p is not None for p in targets.pi)} with visit counts); held out "
            f"{len(held_out)} from {len(loaded) - cut}"
        )
    elif args["demos"]:
        sys.path.insert(0, str(CONFIGS.parent / "scripts"))
        from expert_iteration import load_demos

        problems, plans = load_demos(args["demos"], args["arm"])
        for inst in problems:
            train.add_episode(env, inst, weights, lambda i, _w: plans[i.instance_id])
    else:
        for _ in range(args["problems"]):
            train.add_episode(env, source(rng), weights, teacher)
    val = {name: Demonstrations() for name in val_sets}
    for name, bench in val_sets.items():
        gaps = []
        for inst in bench.instances:
            cost = val[name].add_episode(env, inst, weights, teacher)
            best = bench.reference[inst.instance_id].cost
            gaps.append(float(cost) / float(best) - 1)
        print(f"teacher '{args['teacher']}' on {name}: mean gap to best known {np.mean(gaps):+.3f}")
    widths = targets.candidate_widths() if visits else train.candidate_widths()
    print(
        f"{len(targets) if visits else len(train)} training steps "
        f"(candidates median {int(np.median(widths))}, max {int(np.max(widths))}); "
        f"validation steps {', '.join(f'{k} {len(v)}' for k, v in val.items())}; "
        f"{time.perf_counter() - started:.0f}s"
    )

    device = torch.device(args["device"])
    if args["init"]:
        net = PPOAgent.load(args["init"], device).net
        print(f"network initialised from {args['init']}")
    else:
        net = SetEncoderNet(PPOConfig.from_dict(load("agent/ppo.yaml")).net).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"])
    eval_env = build_env(cfg, [i for b in val_sets.values() for i in b.instances])
    schedule = ScheduleConfig.from_dict(load("eval/default.yaml")["schedule"])
    order = np.arange(len(targets) if visits else len(train))

    def search_pass(
        store: SearchTargets, mb: npt.NDArray[np.int64], learn: bool
    ) -> dict[str, float]:
        """One pass over `mb` of `store`; with `learn`, one optimiser step on its mean loss."""
        n_pi = max(1, sum(store.pi[int(i)] is not None for i in mb))
        n_v = max(1, len(mb))
        sums = {"ce": 0.0, "mse": 0.0, "agree": 0.0, "mae": 0.0}
        widths_mb = store.candidate_widths()[mb]
        for chunk in chunk_by_width(np.arange(len(mb)), widths_mb, args["max_chunk_rows"]):
            obs, pi, value = store.minibatch(mb[chunk])
            t = batch_obs(obs, device)
            with torch.set_grad_enabled(learn):
                ce, mse, agree, mae = search_loss(
                    net,
                    t,
                    torch.as_tensor(pi, dtype=torch.float32, device=device),
                    torch.as_tensor(value, dtype=torch.float32, device=device),
                )
                if learn:
                    (ce / n_pi + args["value_coef"] * mse / n_v).backward()
            sums["ce"] += ce.item()
            sums["mse"] += mse.item()
            sums["agree"] += float(agree)
            sums["mae"] += mae.item()
        return {
            "loss": sums["ce"] / n_pi,
            "value_mse": sums["mse"] / n_v,
            "agree": sums["agree"] / n_pi,
            "value_mae_cost": sums["mae"] / n_v / args["reward_scale"],
        }

    def score(demos: Demonstrations) -> dict[str, float]:
        """Loss and agreement on all teacher steps, and agreement on the Mix steps alone
        (Output and Discard steps are nearly always right and flatter the total, E-16)."""
        net.eval()
        loss = agree = mix_agree = 0.0
        mixes = 0
        index = np.arange(len(demos))
        with torch.no_grad():
            for chunk in chunk_by_width(index, demos.candidate_widths(), args["max_chunk_rows"]):
                obs, actions = demos.minibatch(chunk)
                t = batch_obs(obs, device)
                a = torch.as_tensor(actions, device=device)
                lo, ag = imitation_loss(net, t, a)
                loss += lo.item() * len(chunk)
                agree += ag.item() * len(chunk)
                logits, _ = net(t)
                top = masked_distribution(logits, t["action_mask"]).probs.argmax(-1)
                rows = torch.arange(len(chunk), device=device)
                is_mix = t["candidates"][rows, a, OP_MIX] == 1
                mixes += int(is_mix.sum())
                mix_agree += float(((top == a) & is_mix).sum())
        return {
            "loss": loss / len(demos),
            "agree": agree / len(demos),
            "mix_agree": mix_agree / max(mixes, 1),
        }

    with (out / "metrics.jsonl").open("w") as log:
        for epoch in range(1, args["epochs"] + 1):
            net.train()
            rng.shuffle(order)
            totals = {"loss": 0.0, "agree": 0.0}
            for start in range(0, len(order), args["batch"]):
                mb = order[start : start + args["batch"]]
                optimizer.zero_grad()
                if visits:
                    got = search_pass(targets, mb, learn=True)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()
                    totals["loss"] += got["loss"] * len(mb)
                    totals["agree"] += got["agree"] * len(mb)
                    continue
                for chunk in chunk_by_width(mb, widths, args["max_chunk_rows"]):
                    obs, actions = train.minibatch(chunk)
                    t = batch_obs(obs, device)
                    loss, agree = imitation_loss(net, t, torch.as_tensor(actions, device=device))
                    share = len(chunk) / len(mb)
                    (share * loss).backward()
                    totals["loss"] += loss.item() * len(chunk)
                    totals["agree"] += agree.item() * len(chunk)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": totals["loss"] / len(order),
                "train_agree": totals["agree"] / len(order),
            }
            if visits:
                net.eval()
                got = search_pass(held_out, np.arange(len(held_out)), learn=False)
                row["heldout/visit_agree"] = got["agree"]
                row["heldout/value_mae_cost"] = got["value_mae_cost"]
            for name, demos in val.items():
                for k, v in score(demos).items():
                    row[f"{name}/val_{k}"] = v
            if epoch % args["eval_every"] == 0 or epoch == args["epochs"]:
                agent = PPOAgent(net, device)
                for name, bench in val_sets.items():
                    df = evaluate(AgentProducer(agent, eval_env), bench, schedule)
                    gaps = df["gap_to_best_known"].dropna()
                    row[f"{name}/success"] = float(df["success"].mean())
                    row[f"{name}/gap"] = float(gaps.mean()) if len(gaps) else float("nan")
                    row[f"{name}/cheaper"] = int((gaps < 0).sum())
            row["seconds"] = time.perf_counter() - started
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                " ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                ),
                flush=True,
            )
    PPOAgent(net, device).save(out / "imitation.pt")
    print(f"saved {out / 'imitation.pt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
