"""Search test (plan 4.2 brought forward; paper log D-37): can lookahead find the cheaper plans
that the planner and the greedy agent miss?

Problems where a cheaper plan than our best planner is known:
- **A (the test):** E2 problems with a proved optimum on which the best planner (cheapest of
  `share`, `share_look3`, `share_orders`) is more than 2% above it (`notes/headroom/records.jsonl`).
- **B (reported, no bar):** `val_s2_v2` problems on which MRCM is cheaper than the best planner
  (`notes/lit_bench/runs/e22/records.jsonl`; no proved optimum, MRCM's cost is the reference).

Searches, all from the same env and observations as the agent (AlphaZero-style MCTS, one tree
per real move, kept across moves; value (C0 - total) / C0, total = cost so far + estimated cost
to finish, C0 the root's total):
- `net`: the network's move probabilities as the prior; a leaf is scored by finishing from it
  with the sharing pass (`finish_from`). Each leaf whose finish fits the step limit is a whole
  plan, so the answer can be "a few searched moves, then the planner's finish" (E-19).
- `uniform`: the same with a flat prior: what lookahead with planner finishes does without
  the network.
- `zero`: the network's prior and its own value head as the leaf score (remaining cost =
  -value / `reward_scale`); **no planner call during play** (the env's hint columns are the
  agent's inputs, as trained). Only plans the search itself plays or reaches count.
The answer is the cheapest whole plan seen, replayed and validated; `planner_tail` says how many
of its actions came from a planner finish. A problem that hits `seconds` stops early with what
it has, and says so (`capped`).

Reading, fixed in advance (D-37), on set A at the largest budget ("below" = cheaper than the
best planner):
- **(1) lookahead with planner finishes finds them:** `net` below on >= 1/3 of the problems;
- **(2) the network helps it:** `net` below on more problems than `uniform` (ties: lower mean
  ratio to the optimum);
- **(3) learning plus search finds them without the planner:** `zero` below on >= 1/3.
(3) is the case for RL: search as the teacher of the network (expert iteration) is the next
step. (1) without (3): the gains come from planner finishes after a few searched moves, not from
what the network learned; rethink the network's role. Neither: rethink before stages 3-5.

    uv run python scripts/search_check.py agent=outputs/warm/mixed_s0/checkpoints/agent.pt \
        workers=30 out=notes/search/records.jsonl
    uv run python scripts/search_check.py summary=notes/search/records.jsonl

Arguments are key=value: `sets` (A,B), `limit=N` problems per set, `sims` (budgets, default
64,256), `uniform` (budget for the flat prior, default the largest of `sims`; 0 = off),
`seconds` (cap per problem and search, default 900), `hints` (the env's `hint_source`; must be
the one the agent was trained with, default planner), `reward_scale` (the agent's PPO reward
scale, for `zero`; default 0.01 as E-22), `workers`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from meda.analysis.evaluate import build_env
from meda.core.actions import Action
from meda.core.problem import Instance
from meda.envs.sample_prep_env import EnvSnapshot, SamplePrepEnv
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines.share import finish_from

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom_share as hs  # noqa: E402

_load = lambda rel: OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / rel))  # noqa: E731
WEIGHTS = hs.WEIGHTS
PLANNERS = ("share", "share_look3", "share_orders")
C_PUCT = 1.5
# Fixed in advance (module docstring).
FOUND_SHARE, MISS = 1 / 3, 0.02


def best_planner(rec: dict[str, Any]) -> float:
    return min(rec[k]["cost"] for k in PLANNERS if rec.get(k, {}).get("cost") is not None)


# --------------------------------------------------------------------------- problems


def tasks(sets: set[str], limit: int | None) -> list[dict[str, Any]]:
    """Each task: set, id, cell, the instance (as JSON) and its references."""
    out: list[dict[str, Any]] = []
    if "A" in sets:
        recs = [json.loads(x) for x in (ROOT / "notes/headroom/records.jsonl").open()]
        want = {
            r["id"]: r
            for r in recs
            if r.get("source") == "e2"
            and r.get("optimum") is not None
            and best_planner(r) > (1 + MISS) * r["optimum"]
        }
        found = []
        for t in hs.tasks({"e2"}, None):
            rid = f"{t['rec']['cell']}-{t['rec']['index']}"
            if rid in want:
                inst, _, _ = hs.load_task(t)
                assert inst is not None, rid
                r = want[rid]
                found.append(_task("A", inst, r["cell"], r, r["optimum"]))
        out += found[:limit]
    if "B" in sets:
        recs = [json.loads(x) for x in (ROOT / "notes/lit_bench/runs/e22/records.jsonl").open()]
        want = {
            r["id"]: r
            for r in recs
            if r.get("source") == "val_s2_v2"
            and r.get("mrcm", {}).get("cost") is not None
            and r["mrcm"]["cost"] < best_planner(r)
        }
        insts = [i for i in BenchmarkSet.read("val_s2_v2").instances if i.instance_id in want]
        out += [
            _task("B", i, want[i.instance_id]["cell"], want[i.instance_id], None)
            for i in insts[:limit]
        ]
    return out


def _task(
    name: str, inst: Instance, cell: str, rec: dict[str, Any], optimum: float | None
) -> dict[str, Any]:
    agent = rec.get("agent") or {}
    return {
        "set": name,
        "id": inst.instance_id,
        "cell": cell,
        "instance": inst.to_json(),
        "optimum": optimum,
        "planner": best_planner(rec),
        "mrcm": (rec.get("mrcm") or {}).get("cost"),
        "greedy": agent.get("cost"),
    }


# --------------------------------------------------------------------------- search


Sample = tuple[tuple[Action, ...], list[int] | None, float]  # moves, visits, remaining cost


@dataclass
class Node:
    snap: EnvSnapshot
    actions: tuple[Action, ...]  # from the start of the episode
    prior: np.ndarray
    value: float  # the leaf value when it was expanded
    terminal: bool
    best: float = math.inf  # cheapest whole plan seen through this node (fits the step limit)
    visits: np.ndarray = field(init=False)
    total: np.ndarray = field(init=False)
    children: dict[int, Node] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.visits = np.zeros(len(self.prior), dtype=np.int64)
        self.total = np.zeros(len(self.prior))


class Search:
    """MCTS over the env (deterministic, one player, minimise cost)."""

    def __init__(
        self,
        env: SamplePrepEnv,
        inst: Instance,
        agent: Any,
        deadline: float,
        planner_leaf: bool = True,
        reward_scale: float = 0.01,
        rng: np.random.Generator | None = None,
        collect: int | None = None,
    ) -> None:
        self.env, self.inst, self.agent, self.deadline = env, inst, agent, deadline
        self.planner_leaf, self.reward_scale = planner_leaf, reward_scale
        self.rng = rng  # breaks ties at random; without it ties go in candidate order (E-33)
        self.best: tuple[float, tuple[Action, ...], int] | None = None  # cost, plan, tail
        self.c0 = 1.0
        self.leaves = 0
        # Expert iteration round 2 (D-40): with `collect` (the fewest visits for a position's
        # visit counts to be kept), record what the search observed; see `samples`.
        self.collect = collect
        self._policy: list[Sample] = []
        self._other: list[Sample] = []

    def net(self, obs: dict[str, Any], n: int) -> tuple[np.ndarray, float]:
        """Move probabilities over the `n` candidates (flat without a network) and the
        network's estimate of the cost to finish (0 without one)."""
        if self.agent is None or n == 0:
            return np.full(n, 1.0 / max(n, 1)), 0.0
        import torch

        from meda.agents.ppo.policy import batch_obs, masked_distribution

        with torch.no_grad():
            t = batch_obs({k: v[None] for k, v in obs.items()}, self.agent.device)
            logits, value = self.agent.net(t)
            p = masked_distribution(logits, t["action_mask"]).probs[0].numpy()
        return np.asarray(p[:n], dtype=float), max(0.0, -float(value[0]) / self.reward_scale)

    def _offer(self, cost: float, actions: tuple[Action, ...], tail: int = 0) -> None:
        if self.best is None or cost < self.best[0]:
            self.best = (cost, actions, tail)

    def leaf(self, obs: dict[str, Any], actions: tuple[Action, ...], info: dict | None) -> Node:
        """A node for the env's current state: its prior and its value."""
        self.leaves += 1
        env, snap = self.env, self.env.snapshot()
        if info is not None and "status" in info:  # the episode ended at this node
            if info["success"]:
                cost = float(info["plan_cost"])
                self._offer(cost, actions)
                return Node(snap, actions, np.zeros(0), self.score(cost), True, cost)
            return Node(snap, actions, np.zeros(0), -1.0, True)
        prior, estimate = self.net(obs, len(env.legal))
        if not self.planner_leaf:
            return Node(snap, actions, prior, self.score(float(snap.plan_cost) + estimate), False)
        found = finish_from(env.state, self.inst, WEIGHTS)
        best = math.inf
        if found is None:
            value = -1.0
        else:
            cost = float(snap.plan_cost + found[1])
            if env.state.step + len(found[0]) <= self.inst.step_limit:
                self._offer(cost, actions + found[0], len(found[0]))
                best = cost
            value = self.score(cost)
        return Node(snap, actions, prior, value, False, best)

    def score(self, cost: float) -> float:
        return max(-1.0, (self.c0 - cost) / self.c0)

    def child(self, node: Node, i: int) -> Node:
        self.env.restore(node.snap)
        action = self.env.legal[i]
        obs, _, term, trunc, info = self.env.step(i)
        return self.leaf(obs, (*node.actions, action), info if (term or trunc) else None)

    def simulate(self, root: Node) -> None:
        path: list[tuple[Node, int]] = []
        node = root
        while True:
            n = node.visits.sum()
            q = np.where(node.visits > 0, node.total / np.maximum(node.visits, 1), node.value)
            u = C_PUCT * node.prior * math.sqrt(n + 1) / (1 + node.visits)
            score = q + u
            if self.rng is not None:
                score = score + 1e-9 * self.rng.random(len(score))
            i = int(np.argmax(score))
            path.append((node, i))
            if i not in node.children:
                node.children[i] = self.child(node, i)
                end = node.children[i]
                break
            node = node.children[i]
            if node.terminal:
                end = node
                break
        for nd, i in path:
            nd.visits[i] += 1
            nd.total[i] += end.value
            nd.best = min(nd.best, end.best)

    def run(self, sims: int) -> dict[str, Any]:
        obs, _ = self.env.reset(options={"instance": self.inst})
        if self.planner_leaf:
            root = self.leaf(obs, (), None)
            self.c0 = max(self.best[0], 1.0) if self.best else 1.0
        else:
            self.c0 = max(self.net(obs, len(self.env.legal))[1], 1.0)
            root = self.leaf(obs, (), None)
        root.value = 0.0
        past: list[Node] = []
        capped = False
        while not root.terminal:
            for _ in range(sims):
                if time.perf_counter() > self.deadline:
                    capped = True
                    break
                self.simulate(root)
            if capped or root.visits.sum() == 0:
                break
            i = int(np.argmax(root.visits))
            if self.collect is not None:  # the other branches are never visited again
                for j, other in root.children.items():
                    if j != i:
                        self._take(other)
                root.children = {i: root.children[i]}
                past.append(root)
            root = root.children[i]
        if self.collect is not None:
            self._take(root)
            later = root
            for node in reversed(past):  # plans found after the search moved on count too
                node.best = min(node.best, later.best)
                self._keep(node)
                later = node
        if self.best is None:  # `zero` reached no finished plan
            return {"plan": None, "tail": 0, "capped": capped, "played_to_end": root.terminal}
        return {
            "plan": self.best[1],
            "tail": self.best[2],
            "capped": capped,
            "played_to_end": root.terminal,
        }

    def _keep(self, node: Node) -> None:
        if node.terminal or not math.isfinite(node.best):
            return
        remaining = node.best - float(node.snap.plan_cost)
        assert self.collect is not None
        if node.visits.sum() >= self.collect:
            self._policy.append((node.actions, [int(v) for v in node.visits], remaining))
        else:
            self._other.append((node.actions, None, remaining))

    def _take(self, top: Node) -> None:
        """Record every position under `top` (its search is over)."""
        stack = [top]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            self._keep(node)

    def samples(self, value_cap: int, rng: np.random.Generator) -> list[Sample]:
        """What the search observed, for the network to learn from (expert iteration round 2,
        D-40): per position, the moves from the start, the visit counts over its candidates
        (only where it was visited at least `collect` times, else None) and the remaining cost
        of the cheapest whole plan seen through it. Every position with visit counts is kept,
        plus up to `value_cap` others drawn with `rng` (value only); positions with no whole
        plan seen through them are left out. Branches are recorded as soon as the search
        leaves them and then freed, so the tree never holds more than in round 1."""
        if self.collect is None:
            raise RuntimeError("construct the search with collect=<min visits>")
        pick = rng.permutation(len(self._other))[:value_cap]
        return self._policy + [self._other[int(i)] for i in sorted(pick)]


_AGENT: dict[str, Any] = {}


def run_task(task: dict[str, Any], kind: str, sims: int, opts: dict[str, Any]) -> dict[str, Any]:
    inst = Instance.from_json(task["instance"])
    agent = None
    if kind in ("net", "zero"):
        import torch

        from meda.agents.ppo.policy import PPOAgent

        if opts["agent"] not in _AGENT:
            torch.set_num_threads(1)
            _AGENT[opts["agent"]] = PPOAgent.load(opts["agent"])
        agent = _AGENT[opts["agent"]]
    cfg = {"costs": _load("costs/geometric.yaml"), "env": _load("env/sample_prep.yaml")}
    cfg["env"]["hint_source"] = opts["hints"]
    env = build_env(cfg, [inst])
    t0 = time.perf_counter()
    search = Search(env, inst, agent, t0 + opts["seconds"], kind != "zero", opts["reward_scale"])
    out = search.run(sims)
    stats = hs.plan_stats(inst, f"search_{kind}", out["plan"], time.perf_counter() - t0)
    opt, cost = task["optimum"], stats["cost"]
    if opt is not None and cost is not None and cost < opt - 1e-9:  # a bug (rule 9)
        raise RuntimeError(f"{inst.instance_id}: search plan {stats['cost']} < optimum")
    rec = {k: v for k, v in task.items() if k != "instance"}
    rec.update(
        kind=kind,
        sims=sims,
        leaves=search.leaves,
        capped=out["capped"],
        played_to_end=out["played_to_end"],
        planner_tail=out["tail"],
        **{f"search_{k}": v for k, v in stats.items()},
    )
    return rec


def _star(args: tuple[dict[str, Any], str, int, dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


# --------------------------------------------------------------------------- summary


def group_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [r for r in rs if r["search_cost"] is None]
    rs = [r for r in rs if r["search_cost"] is not None]
    below = [r for r in rs if r["search_cost"] < r["planner"] - 1e-9]
    opt = [r for r in rs if r["optimum"] is not None]
    at_opt = [r for r in opt if r["search_cost"] <= r["optimum"] + 1e-9]
    ratio_opt = [r["search_cost"] / r["optimum"] for r in opt]
    ratio_plan = [r["search_cost"] / r["planner"] for r in rs]
    beats_mrcm = [r for r in rs if r["mrcm"] is not None and r["search_cost"] <= r["mrcm"] + 1e-9]
    return {
        "n": len(rs) + len(failed),
        "failed": len(failed),
        "below": len(below),
        "tail": sum(1 for r in below if r["planner_tail"] > 0),
        "at_opt": len(at_opt),
        "n_opt": len(opt),
        "ratio_opt": float(np.mean(ratio_opt)) if ratio_opt else None,
        "ratio_plan": float(np.mean(ratio_plan)) if ratio_plan else None,
        "mrcm": len(beats_mrcm),
        "capped": sum(r["capped"] for r in rs),
        "seconds": float(np.median([r["search_seconds"] for r in rs])) if rs else 0.0,
    }


def greedy_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    solved = [r for r in rs if r["greedy"] is not None]
    return {
        "n": len(rs),
        "solved": len(solved),
        "below": sum(1 for r in solved if r["greedy"] < r["planner"] - 1e-9),
        "at_opt": sum(
            1 for r in solved if r["optimum"] is not None and r["greedy"] <= r["optimum"] + 1e-9
        ),
    }


def verdict(
    net: dict[str, Any] | None, uniform: dict[str, Any] | None, zero: dict[str, Any] | None
) -> str:
    out = []
    found = net is not None and net["below"] >= FOUND_SHARE * net["n"]
    helps = (
        net is not None
        and uniform is not None
        and (
            net["below"] > uniform["below"]
            or (
                net["below"] == uniform["below"]
                and net["ratio_opt"] is not None
                and uniform["ratio_opt"] is not None
                and net["ratio_opt"] < uniform["ratio_opt"]
            )
        )
    )
    alone = zero is not None and zero["below"] >= FOUND_SHARE * zero["n"]
    out.append(f"(1) {'yes' if found else 'no'}")
    out.append(f"(2) {'yes' if helps else 'no'}")
    out.append(f"(3) {'yes' if alone else 'no'}")
    if alone:
        return ", ".join(out) + ": learning plus search finds them without the planner"
    if found:
        return ", ".join(out) + ": the gains come from planner finishes, not from learning"
    return ", ".join(out) + ": search does not find them at this budget"


def _r(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        groups[(r["set"], r["kind"], r["sims"])].append(r)
    lines = [
        "below = cheaper than the best planner; ratio = mean cost / optimum (A) or / best "
        "planner; MRCM = at or below MRCM's cost (B).",
        "",
        "| set | search | sims | n | below planner | at optimum | ratio to optimum "
        "| ratio to planner | <= MRCM | below via planner finish | no plan | capped | median s |",
        "|---" * 13 + "|",
    ]
    for s in sorted({k[0] for k in groups}):
        firsts = next(rs for k, rs in groups.items() if k[0] == s)
        g = greedy_stats(firsts)
        lines.append(
            f"| {s} | greedy agent (stored) | - | {g['n']} | {g['below']} | {g['at_opt']} | - | - "
            f"| - | - | - | - | - | (solved {g['solved']})"
        )
        for k in sorted(k for k in groups if k[0] == s):
            st = group_stats(groups[k])
            lines.append(
                f"| {s} | {k[1]} | {k[2]} | {st['n']} | {st['below']} | {st['at_opt']}/"
                f"{st['n_opt']} | {_r(st['ratio_opt'])} | {_r(st['ratio_plan'])} | {st['mrcm']} "
                f"| {st['tail']} | {st['failed']} | {st['capped']} | {st['seconds']:.0f} |"
            )
    a = {k: group_stats(rs) for k, rs in groups.items() if k[0] == "A"}
    if a:
        top = max(k[2] for k in a)
        net, uni, zero = (a.get(("A", kind, top)) for kind in ("net", "uniform", "zero"))
        lines += ["", f"Reading on A at {top} sims (fixed in advance): {verdict(net, uni, zero)}"]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    sims = [int(x) for x in args.get("sims", "64,256").split(",")]
    uniform = int(args.get("uniform", max(sims)))
    opts = {
        "agent": args.get("agent"),
        "seconds": float(args.get("seconds", 900)),
        "hints": args.get("hints", "planner"),
        "reward_scale": float(args.get("reward_scale", 0.01)),
    }
    limit = int(args["limit"]) if "limit" in args else None
    found = tasks(set(args.get("sets", "A,B").split(",")), limit)
    runs = [(kind, s) for kind in ("net", "zero") for s in sims] if opts["agent"] else []
    runs += [("uniform", uniform)] if uniform else []
    todo = [(t, kind, s, opts) for t in found for kind, s in runs]
    out = Path(args.get("out", "notes/search/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(found)} problems x {runs} = {len(todo)} searches, {workers} workers", flush=True)
    t0, done = time.perf_counter(), 0
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {done} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
