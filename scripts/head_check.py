"""Two new value heads, offline (D-54, Q-126, after E-49): can a head rank sibling moves?

E-49: with 100k clean labels the value head picks the planner's cheapest sibling 0.39 of the
time (bound 0.36, chance 0.21); its error (139) is far above the sibling gap (7). Two changes
never tried, each alone and together, on E-49's own data (train on `train_100000.jsonl`,
grade on `held.jsonl`), all from round 3's network:
- `global`: today's head (MLP on the global token), squared error on the planner finish. The
  control; it should land near E-49's 0.39.
- `moves`: a head that also reads the candidate moves: attention pooling and the mean over the
  scorer's last hidden layer per candidate (each candidate carries its exact result and the
  hint columns, which the global token never sees), with the global token, into a new MLP.
  Squared error.
- `rank`: today's head, trained to rank siblings instead of to hit the cost: per sibling group,
  cross-entropy between softmax(-(spent + predicted finish) / tau) and the cheapest sibling
  (spent + planner finish). Only differences inside a group are learned, so the cost's large
  fixed part never has to be predicted.
- `moves_rank`: both.
Every arm trains the whole network (lr, clip and epochs as E-49's 100k run). The loss is
taken on all of a batch's values at once, then pushed through the network chunk by chunk, so a
group's loss sees every sibling however wide the observations.

Graded as E-49 (`scaling_check.metrics`): how often spent + predicted finish picks the
planner's cheapest sibling, against the bound's pick and chance; `top2`: the cheapest sibling
is among the head's two best (what pruning children before planner finishes needs).
Reading, fixed in advance (D-54): an arm is useful if it picks the cheapest sibling >= 0.55 of
the time on the held-out problems; none useful → the value head cannot rank siblings even
seeing the moves and trained to rank (Q-125 is the next step).

    uv run python scripts/head_check.py train=outputs/scaling/train_100000.jsonl \\
        held=outputs/scaling/held.jsonl init=outputs/expert/round3/copy/imitation.pt \\
        workers=36 device=cuda threads=16 out=outputs/heads

Arguments are key=value: `train`, `held`, `init`, `arms`, `epochs`, `batch` (positions per
step), `groups` (sibling groups per step for the ranking arms), `tau` (cost units), `lr`,
`seed`, `workers`, `threads`, `device`, `reward_scale`, `max_chunk_rows`, `out`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from meda.agents.ppo.networks import SetEncoderNet, masked_mean
from meda.agents.ppo.policy import PPOAgent, batch_obs
from meda.agents.ppo.ppo import chunk_by_width
from meda.agents.ppo.rollout import pad_obs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_check as sc  # noqa: E402

DEFAULTS: dict[str, Any] = {
    "train": "",
    "held": "",
    "init": "",
    "arms": "global,moves,rank,moves_rank",
    "epochs": 6,  # as E-49's 100k run
    "batch": 64,
    "groups": 10,  # ~60 positions a step, like `batch`
    "tau": 10.0,  # cost units; the median sibling gap is 7 (E-49)
    "lr": 3e-4,
    "seed": 0,
    "workers": 8,
    "threads": 8,
    "device": "cpu",
    "reward_scale": 0.01,
    "max_chunk_rows": 32768,
    "out": "",
}
BAR = 0.55  # fixed in advance (D-54)


@dataclass
class Positions:
    """Labelled positions of many problems: observations, the planner finish (`target`), the
    cost spent, the bound and the parent's index here (-1 at the top)."""

    obs: list[dict[str, Any]] = field(default_factory=list)
    target: list[float] = field(default_factory=list)
    spent: list[float] = field(default_factory=list)
    bound: list[float] = field(default_factory=list)
    parent: list[int] = field(default_factory=list)

    def add(self, store: Any, spent: list[float], bound: list[float], scale: float) -> None:
        offset = len(self.obs)
        self.obs += store.obs
        self.target += [-v / scale for v in store.value]
        self.spent += spent
        self.bound += bound
        self.parent += [p + offset if p >= 0 else -1 for p in store.parent]

    def widths(self) -> npt.NDArray[np.int64]:
        return np.array([o["action_mask"].shape[1] for o in self.obs], dtype=np.int64)

    def groups(self) -> list[npt.NDArray[np.int64]]:
        """Sibling groups: the labelled children of one position, two or more."""
        kids: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(self.parent):
            if p >= 0:
                kids[p].append(i)
        return [np.array(k) for k in kids.values() if len(k) >= 2]


def load(path: str, workers: int, scale: float) -> Positions:
    lines = [json.loads(x) for x in Path(path).read_text().split("\n") if x]
    with Pool(workers) as pool:
        got = pool.map(sc.replay, [(r["instance"], r["samples"], scale) for r in lines])
    data = Positions()
    for store, spent, bound in got:
        data.add(store, spent, bound, scale)
    return data


class Head(nn.Module):
    """A value head on a `SetEncoderNet` (value units: -cost x reward_scale). `moves` adds
    attention and mean pooling over the scorer's last hidden layer of the legal candidates."""

    def __init__(self, net: SetEncoderNet, moves: bool, start: float) -> None:
        super().__init__()
        self.net, self.moves = net, moves
        if moves:
            d = net.cfg.d_model
            self.attend = nn.Linear(d, 1)
            last = nn.Linear(d, 1)
            nn.init.constant_(last.bias, start)
            self.value = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), last)

    def forward(self, t: dict[str, Tensor]) -> Tensor:
        if not self.moves:
            return self.net.value_head(self.net.encode(t)[1]).squeeze(-1)
        z, g = self.net.candidate_inputs(t)
        h = self.net.scorer[:-1](z)  # [B, L, d]
        legal = t["action_mask"] == 1
        a = self.attend(h).squeeze(-1).masked_fill(~legal, torch.finfo(h.dtype).min)
        pooled = (torch.softmax(a, dim=-1).unsqueeze(-1) * h).sum(1)
        mean = masked_mean(h, legal.unsqueeze(-1), dim=1)
        return self.value(torch.cat([g, pooled, mean], dim=-1)).squeeze(-1)


def rank_loss(
    v: Tensor, spent: Tensor, target: Tensor, slices: list[slice], tau: float, scale: float
) -> Tensor:
    """Mean over groups of the cross-entropy between softmax(-(spent + predicted finish) / tau)
    and the cheapest siblings (spent + target; ties share)."""
    losses = []
    for s in slices:
        logits = (v[s] / scale - spent[s]) / tau
        total = spent[s] + target[s]
        best = torch.isclose(total, total.min()).to(v.dtype)
        losses.append(-(best / best.sum() * torch.log_softmax(logits, dim=0)).sum())
    return torch.stack(losses).mean()


def squared_error(v: Tensor, want: Tensor) -> Tensor:
    return ((v - want) ** 2).mean()


def values(
    head: Head, data: Positions, idx: npt.NDArray[np.int64], device: torch.device, rows: int
) -> Tensor:
    out = torch.zeros(len(idx), device=device)
    widths = data.widths()[idx]
    for c in chunk_by_width(np.arange(len(idx)), widths, rows):
        out[torch.as_tensor(c, device=device)] = head(
            batch_obs(pad_obs([data.obs[int(i)] for i in idx[c]]), device)
        )
    return out


def step(
    head: Head,
    opt: torch.optim.Optimizer,
    data: Positions,
    idx: npt.NDArray[np.int64],
    loss_fn: Callable[[Tensor], Tensor],
    device: torch.device,
    rows: int,
) -> float:
    """One optimiser step: the loss on the batch's values, its gradient pushed back through the
    network one chunk at a time."""
    with torch.no_grad():
        v = values(head, data, idx, device, rows)
    v.requires_grad_(True)
    loss = loss_fn(v)
    (grad,) = torch.autograd.grad(loss, v)
    opt.zero_grad()
    widths = data.widths()[idx]
    for c in chunk_by_width(np.arange(len(idx)), widths, rows):
        out = head(batch_obs(pad_obs([data.obs[int(i)] for i in idx[c]]), device))
        (out * grad[torch.as_tensor(c, device=device)]).sum().backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
    opt.step()
    return float(loss.item())


def grade(
    head: Head, held: Positions, device: torch.device, rows: int, scale: float
) -> dict[str, Any]:
    head.eval()
    with torch.no_grad():
        v = values(head, held, np.arange(len(held.obs)), device, rows).cpu().numpy()
    table = {
        "parent": np.array(held.parent),
        "spent": np.array(held.spent),
        "target": np.array(held.target),
        "pred": -v.astype(np.float64) / scale,
        "bound": np.array(held.bound),
    }
    row = sc.metrics(table)
    row["top2"] = top2(table, held.groups())
    return row


def top2(rows: dict[str, npt.NDArray[Any]], groups: list[npt.NDArray[np.int64]]) -> float | None:
    """Share of groups whose cheapest sibling is among the head's two best."""
    hits = []
    for k in groups:
        total = rows["spent"][k] + rows["target"][k]
        best = np.isclose(total, total.min())
        hits.append(bool(best[np.argsort(rows["spent"][k] + rows["pred"][k])[:2]].any()))
    return float(np.mean(hits)) if hits else None


def reading(table: dict[str, dict[str, Any]]) -> list[str]:
    """The rule fixed in advance (D-54)."""
    useful = [a for a, r in table.items() if (r["head_best"] or 0.0) >= BAR]
    if not useful:
        return [f"no arm picks the cheapest sibling >= {BAR}: the head cannot rank siblings"]
    return [f"{a}: useful (picks the cheapest sibling {table[a]['head_best']:.2f})" for a in useful]


def train_arm(
    arm: str, data: Positions, held: Positions, args: dict[str, Any], out: Path
) -> dict[str, Any]:
    torch.manual_seed(args["seed"])
    rng = np.random.default_rng(args["seed"])
    device = torch.device(args["device"])
    scale, rows = args["reward_scale"], args["max_chunk_rows"]
    start = -float(np.mean(data.target)) * scale
    head = Head(PPOAgent.load(args["init"], device).net, "moves" in arm, start).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args["lr"])
    target = torch.as_tensor(data.target, dtype=torch.float32, device=device)
    spent = torch.as_tensor(data.spent, dtype=torch.float32, device=device)
    groups = data.groups()
    started = time.perf_counter()
    row: dict[str, Any] = {}
    for epoch in range(1, args["epochs"] + 1):
        head.train()
        losses = []
        if "rank" in arm:
            order = rng.permutation(len(groups))
            for b in range(0, len(order), args["groups"]):
                chosen = [groups[i] for i in order[b : b + args["groups"]]]
                idx = np.concatenate(chosen)
                ends = np.cumsum([len(g) for g in chosen])
                slices = [slice(e - len(g), e) for g, e in zip(chosen, ends, strict=True)]
                loss_fn = partial(
                    rank_loss,
                    spent=spent[idx],
                    target=target[idx],
                    slices=slices,
                    tau=args["tau"],
                    scale=scale,
                )
                losses.append(step(head, opt, data, idx, loss_fn, device, rows))
        else:
            order = rng.permutation(len(data.obs))
            for b in range(0, len(order), args["batch"]):
                idx = order[b : b + args["batch"]]
                loss_fn = partial(squared_error, want=-target[idx] * scale)
                losses.append(step(head, opt, data, idx, loss_fn, device, rows))
        row = {"arm": arm, "epoch": epoch, "train_loss": float(np.mean(losses))}
        row |= grade(head, held, device, rows, scale)
        row["seconds"] = time.perf_counter() - started
        print(" ".join(f"{k}={sc.fc._fmt(v)}" for k, v in row.items()), flush=True)
        with (out / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
    torch.save(head.state_dict(), out / f"{arm}.pt")
    return row


def main(argv: list[str]) -> None:
    args = dict(DEFAULTS)
    for item in argv:
        key, _, value = item.partition("=")
        if key not in args:
            raise SystemExit(f"unknown argument '{key}'; known: {sorted(args)}")
        args[key] = type(DEFAULTS[key])(value)
    if not (args["train"] and args["held"] and args["init"] and args["out"]):
        raise SystemExit("needs train=, held=, init= and out=")
    arms = args["arms"].split(",")
    unknown = set(arms) - {"global", "moves", "rank", "moves_rank"}
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    torch.set_num_threads(args["threads"])
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scale = args["reward_scale"]
    data = load(args["train"], args["workers"], scale)
    held = load(args["held"], args["workers"], scale)
    print(
        f"train {len(data.obs)} positions, {len(data.groups())} sibling groups; held "
        f"{len(held.obs)}, {len(held.groups())} groups; {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    table = {arm: train_arm(arm, data, held, args, out) for arm in arms}
    names = ["head_best", "top2", "bound_best", "chance_best", "head_err", "runner_gap"]
    print("| arm | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for arm, row in table.items():
        print(f"| {arm} | " + " | ".join(sc.fc._fmt(row[c]) for c in names) + " |")
    for line in reading(table):
        print(f"Reading (fixed in advance, D-54): {line}")
    (out / "table.json").write_text(json.dumps(table, indent=1))
    print(f"{time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
