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
arm's plans, starting from the given weights; validation is unchanged.

Writes `metrics.jsonl` (one line per round) and `imitation.pt` (the network, loadable with
`PPOAgent.load`, the starting point of the PPO warm start) into `out`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from meda.agents.ppo.imitation import TEACHERS, Demonstrations, imitation_loss
from meda.agents.ppo.networks import SetEncoderNet
from meda.agents.ppo.policy import PPOAgent, batch_obs, masked_distribution
from meda.agents.ppo.ppo import PPOConfig, chunk_by_width
from meda.analysis.dag_metrics import ScheduleConfig
from meda.analysis.evaluate import AgentProducer, build_env, evaluate
from meda.core.chip import ChipLimits
from meda.core.costs import CostWeights
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
    "arm": "search",  # with `demos`: copy the search's plans or the planner's (D-38)
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
    if args["demos"]:
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
    widths = train.candidate_widths()
    print(
        f"{len(train)} teacher steps from {train.episodes} problems "
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
    order = np.arange(len(train))

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
                "train_loss": totals["loss"] / len(train),
                "train_agree": totals["agree"] / len(train),
            }
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
