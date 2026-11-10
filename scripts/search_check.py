"""Search test (plan 4.2 brought forward; paper log D-37): can lookahead find the cheaper plans
that the planner and the greedy agent miss?

Problems where a cheaper plan than our best planner is known:
- **A (the test):** E2 problems with a proved optimum on which the best planner (cheapest of
  `share`, `share_look3`, `share_orders`, `share_mrcm_look3` where recorded) is more than 2%
  above it (`notes/headroom/records.jsonl`).
- **B (reported, no bar):** `val_s2_v2` problems on which MRCM is cheaper than the best planner
  (`notes/lit_bench/runs/e22/records.jsonl`; no proved optimum, MRCM's cost is the reference).
- **C (D-41):** `val_s3_v2`, `val_s4_v2`, `val_s5_v2` at full size, without the problems where
  the best planner is within 2% of the lower bound (it is already optimal there), from
  `notes/headroom/records.jsonl`. Reported per stage as C3, C4, C5.
- **D (D-56):** B at full size: `val_s3_v2`, `val_s4_v2`, `val_s5_v2` problems on which a
  published method (e-MRCM or NFSP) is cheaper than the best planner
  (`notes/lit_bench/runs/full/records.jsonl`, from `lit_bench_full.py`); the cheaper of the two
  is the reference in the `mrcm` field.

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

Reading on C, fixed in advance (D-41), all of C at the largest budget: the same three
questions, (2)'s tie broken by the mean ratio to the planner. (1) and (3) → round 3 is worth
running where the paper's claim is; (1) without (3) → the value head does not carry over to
full-size stages 3-5; neither → search does not beat the planner there, rethink before round 3.

Race, fixed in advance (D-43): each record keeps the cost of the cheapest whole plan over time
(`trace`), so one run gives every time budget up to its cap. Cost at t = the cheaper of the best
planner and the search's best plan by t (what one would use if stopped at t). On all of C at
the largest budget: **the network makes search faster** if at 10 s or at 60 s `net`'s mean cost
/ planner is at least 0.02 below `uniform`'s and `net` is cheaper than `uniform` on more
problems than the reverse; otherwise no speed gain at these budgets. B is reported the same way.

Race on D, fixed in advance (D-56): the same rule as D-43, on D at the largest budget; with
fewer than 10 problems in D it is reported only.

- **T (D-59):** the scale check's 10 probes with 40 targets (`scale_check.problems("t40", 10,
  0)`, E-53: one planner finish takes a median 26 s there). No planner records exist; the
  reference `planner` is the sharing pass from the empty chip (`finish_from`), computed before
  the search's clock starts. The env's `T_pad` is raised to the target count (a memory setting).
Race on T, fixed in advance (D-59): D-43's rule for `net` and for `zero`, each at every budget
run, against `uniform`; **the network makes search faster at 40 targets** if any of them passes.

    uv run python scripts/search_check.py sets=T agent=outputs/expert/round2/copy_v/imitation.pt \
        sims=1,16,256 uniform=256 workers=30 out=outputs/scale/search_t.jsonl

Arguments are key=value: `sets` (A,B; C, D, F, T and W on request), `limit=N` problems per
set, `sims` (budgets, default 64,256), `uniform` (budget for the flat prior, default the
largest of `sims`; 0 = off), `seconds` (cap per problem and search, default 900), `hints` (the env's
`hint_source`; must be the one the agent was trained with, default planner), `reward_scale`
(the agent's PPO reward scale, for `zero`; default 0.01 as E-22), `workers`,
`finishers` (leaf finishers: share, mrcm, paper, strong, share_mrcm; default share).

Stronger finisher (D-64, after E-59): `strong` finishes each leaf with E-58's planner (lookahead
3 with MRCM's tree, about 15x slower than the plain pass). On C with the flat prior, same 900 s
cap as E-39; read with `planner_check.py search=<E-39 race>,<this> look=<E-58 records>`
(reading in its docstring).

    uv run python scripts/search_check.py sets=C finishers=strong uniform=256 sims=256 \
        workers=30 out=outputs/finish/strong_c.jsonl

Does the search need the planner's lookahead (D-65, after E-60)? `share_mrcm` finishes each leaf
with MRCM's tree but greedy recipe choice (no lookahead, about 2.5x faster per finish); same
run as D-64 otherwise, read the same way with its records added to `search=`.

    uv run python scripts/search_check.py sets=C finishers=share_mrcm uniform=256 sims=256 \
        workers=30 out=outputs/finish/greedy_c.jsonl

- **W (D-66, after E-61):** published wins. Every lit-bench problem (`notes/lit_bench/runs/e22`,
  s1-s2 and the litbench sets; `runs/full`, s3-s5, where present) on which a published method is
  cheaper than the old best planner recorded there; the cheapest method is the reference (the
  `mrcm` field, its name in `method`). Each problem first runs the new planner
  (`share_mrcm_look3`, off the clock); where that already ties or beats the method, no search
  (`closed_by_planner`). Otherwise the search runs as E-61 (flat prior, greedy MRCM finishes).
Reading on W, fixed in advance (D-66): on the problems the new planner leaves, the search at or
below the best published method on all → **no published method beats our method on these
sets**; on >= 2/3 → most wins closed, the rest listed; fewer → published methods keep real wins.

    uv run python scripts/search_check.py sets=W finishers=share_mrcm uniform=256 sims=256 \
        workers=30 out=outputs/finish/wins_w.jsonl

Finisher test (D-55): the leaf finisher is a parameter. `finishers=share,mrcm` runs each search
once per finisher: `share` is the sharing pass (`finish_from`), `mrcm` is MRCM finishing from the
current chip (`MrcmFinish`, chip droplets offered as e-MRCM offers shared ones). The root's first
leaf is the finisher's own whole plan from the empty chip, and the answer is the cheapest plan
seen, so the search is never dearer than its finisher (checked: a dearer plan is a bug). Reading
on B, fixed in advance, `uniform` at the largest budget with `mrcm`: **the search improves MRCM**
if it is below MRCM on >= 1/3 of B; below on some but fewer: a weak gain; on none: MRCM's misses
are not reachable by lookahead with its own finishes at this budget. Reported, no bar: the same
against the `share` finisher (which finisher the search does better with).

    uv run python scripts/search_check.py sets=B finishers=share,mrcm sims=256 workers=26 \
        out=notes/search/finish_b.jsonl

- **F (D-60):** the finisher test for the single-target graph methods DFM, FacDA, Generic and
  DFMM. Per method, sub-set `F-<method>`: the 13 `val_s1_v2` / `val_s2_v2` problems the method
  takes on which it does best against our best planner (lowest method cost / planner; ties by
  id), from `notes/lit_bench/runs/e22/records.jsonl`; the method's cost is the reference (the
  `mrcm` field). Finisher `paper` is the problem's own method finishing from the current chip
  (`GraphFinish`: its graph, with nodes already on chip taken from there).
Reading on F, fixed in advance (D-60), `uniform` at the largest budget with `paper`: per
method, the search improves it if it is below the method on >= 1/3 of its 13 problems (as
D-55). With MRCM (E-55), **the search improves most published methods** if it improves at
least 3 of these 4 (then 4 or more of the 7 we compare with); 1-2: some; 0: none of these.

    uv run python scripts/search_check.py sets=F finishers=share,paper sims=256 workers=30 \
        out=outputs/finish/finish_f.jsonl
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from meda.analysis.evaluate import build_env
from meda.core.actions import Action
from meda.core.composition import Rational
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError
from meda.core.state import PrepState
from meda.envs.sample_prep_env import EnvSnapshot, SamplePrepEnv
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import share
from meda.solvers.baselines.dfmm import dfmm
from meda.solvers.baselines.facda import dfm, facda
from meda.solvers.baselines.generic import generic
from meda.solvers.baselines.paper_finish import GraphFinish, MrcmFinish
from meda.solvers.baselines.share import finish_from, share_plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom_share as hs  # noqa: E402

_load = lambda rel: OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / rel))  # noqa: E731
WEIGHTS = hs.WEIGHTS
PLANNERS = ("share", "share_look3", "share_orders", "share_mrcm_look3")
C_PUCT = 1.5
# D-72: an MRCM finish slower than this switches the leaves to the plain pass (E-67: N = 3
# ~0.5 s a leaf, where the search works; N = 4 ~9 s, where it starves); the root's finish
# also fills MRCM's cache, so it gets AUTO_ROOT_LIMIT (Mac, val_emrcm_v1: N3-T1 37 s, N4-T1
# 42 s, N5-T1 73 s, N4-T3 301 s, N5-T3 378 s; the plain pass 0.01-1.2 s)
AUTO_LIMIT, AUTO_ROOT_LIMIT = 2.0, 60.0
# D-74 (pilot A2), fixed before the coe9 run: greedy MRCM checks leaves besides the plain
# pass while it has used at most this share of the search's time
TIER_SHARE = 0.5
# D-75 (pilot A3), from E-73's offline check on val: switch a problem from greedy MRCM at every
# leaf to A2's tiered finisher once MRCM's leaf finishes (the root's left out) average more
# than this many seconds, counted over at least SWITCH_MIN_LEAVES leaves
SWITCH_MEAN, SWITCH_MIN_LEAVES = 20.0, 3
# Fixed in advance (module docstring).
FOUND_SHARE, MISS = 1 / 3, 0.02
D_MIN = 10
LIT_FULL = ROOT / "notes/lit_bench/runs/full/records.jsonl"
T_SIZE, T_N = "t40", 10  # D-59: the scale check's probes
F_METHODS = {"dfm": dfm, "facda": facda, "generic": generic, "dfmm": dfmm}  # D-60
F_N, F_MOST = 13, 3  # D-60: problems per method; methods improved for "most"
LIT_S12 = ROOT / "notes/lit_bench/runs/e22/records.jsonl"
W_METHODS = ("dfm", "facda", "generic", "dfmm", "mrcm", "emrcm", "nfsp")  # D-66
W_MOST = 2 / 3  # D-66: share of the problems still lost on which the search must catch up
COSTS = "costs/geometric.yaml"  # the env's costs; must match WEIGHTS
RACE_AT, RACE_GAP = (1, 10, 60, 300, 900), 0.02


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
    if "C" in sets:
        recs = [json.loads(x) for x in (ROOT / "notes/headroom/records.jsonl").open()]
        for stage in (3, 4, 5):
            src = f"val_s{stage}_v2"
            want = {
                r["id"]: r
                for r in recs
                if r.get("source") == src and best_planner(r) > (1 + MISS) * r["bound"]
            }
            insts = [i for i in BenchmarkSet.read(src).instances if i.instance_id in want]
            out += [
                _task(f"C{stage}", i, want[i.instance_id]["cell"], want[i.instance_id], None)
                for i in insts[:limit]
            ]
    if "D" in sets:
        recs = [json.loads(x) for x in (ROOT / "notes/headroom/records.jsonl").open()]
        lit = {}
        for r in map(json.loads, LIT_FULL.open()):
            costs = [r[k]["cost"] for k in ("emrcm", "nfsp") if r[k].get("cost") is not None]
            if costs:
                lit[(r["source"], r["id"])] = min(costs)
        for stage in (3, 4, 5):
            src = f"val_s{stage}_v2"
            want = {
                r["id"]: r
                for r in recs
                if r.get("source") == src
                and (src, r["id"]) in lit
                and lit[(src, r["id"])] < best_planner(r) - 1e-9
            }
            insts = [i for i in BenchmarkSet.read(src).instances if i.instance_id in want]
            out += [
                _task("D", i, want[i.instance_id]["cell"], want[i.instance_id], None)
                | {"mrcm": lit[(src, i.instance_id)]}
                for i in insts[:limit]
            ]
    if "F" in sets:
        recs = [json.loads(x) for x in (ROOT / "notes/lit_bench/runs/e22/records.jsonl").open()]
        insts = {
            (src, i.instance_id): i
            for src in ("val_s1_v2", "val_s2_v2")
            for i in BenchmarkSet.read(src).instances
        }
        for m in F_METHODS:
            takes = [
                r
                for r in recs
                if (r["source"], r["id"]) in insts and (r.get(m) or {}).get("cost") is not None
            ]
            takes.sort(key=lambda r: (r[m]["cost"] / best_planner(r), r["id"]))
            out += [
                _task(f"F-{m}", insts[(r["source"], r["id"])], r["cell"], r, None)
                | {"mrcm": r[m]["cost"], "method": m}
                for r in takes[:F_N][:limit]
            ]
    if "W" in sets:
        out += w_tasks(limit)
    if "T" in sets:
        import scale_check

        for inst in scale_check.problems(T_SIZE, T_N, 0)[:limit]:
            out.append(_task("T", inst, inst.metadata["cell"], {}, None) | {"planner": None})
    return out


def w_tasks(limit: int | None) -> list[dict[str, Any]]:
    """Set W (D-66): every lit-bench problem on which a published method is cheaper than the
    best planner recorded there (old planners; the new one is computed in `run_task`)."""
    out: list[dict[str, Any]] = []
    for path in (LIT_S12, LIT_FULL):
        if not path.exists():
            print(f"set W: {path} not found, its problems are left out", flush=True)
            continue
        by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in map(json.loads, path.open()):
            if r["source"] == "e2":  # not a benchmark set; its one MRCM win is closed (E-57)
                continue
            old = r["planner"] if "planner" in r else best_planner(r)
            pub = {m: r[m]["cost"] for m in W_METHODS if (r.get(m) or {}).get("cost") is not None}
            if pub and min(pub.values()) < old - 1e-9:
                by[r["source"]][r["id"]] = r | {"old": old, "published": pub}
        for src, want in by.items():
            insts = [i for i in BenchmarkSet.read(src).instances if i.instance_id in want]
            for i in insts[:limit]:
                r = want[i.instance_id]
                best = min(r["published"], key=lambda m: (r["published"][m], m))
                out.append(
                    _task("W", i, r.get("cell", src), {}, None)
                    | {
                        "source": src,
                        "planner": r["old"],
                        "published": r["published"],
                        "method": best,
                        "mrcm": r["published"][best],
                    }
                )
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
        "planner": best_planner(rec) if rec else None,
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
    # D-73: planner jumps, the moves after the env's candidates (index len(legal) + j);
    # None until the search first comes back to the node (most leaves are never revisited)
    jumps: tuple[tuple[Action, ...], ...] | None = None

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
        finish: Any = None,
        jumps: Any = None,
        jump_share: float = 0.5,
    ) -> None:
        self.env, self.inst, self.agent, self.deadline = env, inst, agent, deadline
        # the leaf finisher (D-55): state -> (actions, cost) or None; the sharing pass by default
        self.finish = finish or (lambda state: finish_from(state, inst, WEIGHTS))
        # D-73: state -> the planner's next moves as action runs (None: single actions only),
        # and the share of a node's prior they get together
        self.jumps, self.jump_share = jumps, jump_share
        self.planner_leaf, self.reward_scale = planner_leaf, reward_scale
        self.rng = rng  # breaks ties at random; without it ties go in candidate order (E-33)
        self.best: tuple[float, tuple[Action, ...], int] | None = None  # cost, plan, tail
        # (seconds, cost) each time the cheapest whole plan improves: the race of D-43
        self.t0 = time.perf_counter()
        self.trace: list[tuple[float, float]] = []
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
            self.trace.append((round(time.perf_counter() - self.t0, 3), cost))

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
        found = self.finish(env.state)
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

    def add_jumps(self, node: Node) -> None:
        """Give `node` its planner jumps (D-73) the first time the search comes back to it, so
        a leaf never revisited costs no more than in v1, and the finisher (at the root, the
        MRCM switch) has run before the jumps choose their recipes."""
        if self.jumps is None or node.jumps is not None or node.terminal:
            node.jumps = node.jumps or ()
            return
        self.env.restore(node.snap)
        node.jumps = tuple(self.jumps(self.env.state))
        m = len(node.jumps)
        if m:
            node.prior = self._with_jumps(node.prior, m)
            node.visits = np.concatenate([node.visits, np.zeros(m, dtype=np.int64)])
            node.total = np.concatenate([node.total, np.zeros(m)])

    def _with_jumps(self, prior: np.ndarray, m: int) -> np.ndarray:
        """The env's candidates keep 1 - `jump_share` of the prior, the `m` jumps share the
        rest evenly (all of it when there are no candidates)."""
        if m == 0:
            return prior
        if len(prior) == 0:
            return np.full(m, 1.0 / m)
        return np.concatenate([prior * (1 - self.jump_share), np.full(m, self.jump_share / m)])

    def score(self, cost: float) -> float:
        return max(-1.0, (self.c0 - cost) / self.c0)

    def child(self, node: Node, i: int) -> Node:
        self.env.restore(node.snap)
        n = len(self.env.legal)
        if i < n:
            action = self.env.legal[i]
            obs, _, term, trunc, info = self.env.step(i)
            return self.leaf(obs, (*node.actions, action), info if (term or trunc) else None)
        done: list[Action] = []  # a planner jump (D-73): its actions, until the episode ends
        try:
            assert node.jumps is not None
            for action in node.jumps[i - n]:
                obs, _, term, trunc, info = self.env.step_action(action)
                done.append(action)
                if term or trunc:
                    break
        except IllegalActionError:  # the planner's relaxed step limit, or a bug: a dead end
            return Node(self.env.snapshot(), (*node.actions, *done), np.zeros(0), -1.0, True)
        return self.leaf(obs, (*node.actions, *done), info if (term or trunc) else None)

    def simulate(self, root: Node) -> None:
        path: list[tuple[Node, int]] = []
        node = root
        while True:
            if node.jumps is None:
                self.add_jumps(node)
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
            if self.jumps is not None:  # D-73: a jump can end the plan; its cost is kept as
                # an answer, but the search moves on along the busiest unfinished move
                open_ = [
                    j
                    for j in np.argsort(-root.visits, kind="stable")
                    if root.visits[j] > 0 and not root.children[int(j)].terminal
                ]
                if not open_:
                    break
                i = int(open_[0])
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
        self.path = root.actions  # the moves played (`scripts/dead_ends.py` replays them)
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


class StrongFinish:
    """The leaf finisher of D-64: the sharing pass from the current chip with lookahead 3 and
    MRCM's tree (E-58's planner), one MRCM cache per problem. With `lookahead=0`, D-65's
    `share_mrcm` finisher: MRCM's tree, recipes chosen greedily."""

    def __init__(self, inst: Instance, lookahead: int = 3) -> None:
        self.inst, self.lookahead = inst, lookahead
        self.cache: dict[Any, list[Any]] = {}
        self.fails: dict[str, int] = {}

    def __call__(self, state: PrepState) -> tuple[tuple[Action, ...], Rational] | None:
        found = finish_from(state, self.inst, WEIGHTS, lookahead=self.lookahead, mrcm=self.cache)
        if found is None:
            self.fails["no order finishes"] = self.fails.get("no order finishes", 0) + 1
        return found


class AutoFinish:
    """The leaf finisher of D-72 (pilot A): greedy MRCM (`share_mrcm`, D-65) while it is fast,
    else the plain sharing pass. The root's finish fills MRCM's model cache, so it is slow even
    where later finishes are fast (N = 3: about 37 s, then about 0.5 s a leaf): the switch to
    the plain pass happens when the root's MRCM finish takes more than `root_limit` seconds or
    any later one more than `limit` (E-67: at N = 4-5 a later finish takes 9 s to 400 s and the
    search starves). Whenever the switch happens, that leaf's answer is the cheaper of the two
    finishes, so the search keeps v1's plans up to that point."""

    def __init__(self, inst: Instance, limit: float, root_limit: float) -> None:
        self.inst, self.limit, self.root_limit = inst, limit, root_limit
        self.mrcm = StrongFinish(inst, lookahead=0)
        self.fails = self.mrcm.fails
        self.mode = "share_mrcm"  # becomes "share" at the switch, never back
        self.root_seconds: float | None = None
        self.calls = 0  # finishes run; `switched_at` is the call that switched
        self.switched_at: int | None = None

    def __call__(self, state: PrepState) -> tuple[tuple[Action, ...], Rational] | None:
        self.calls += 1
        if self.mode == "share":
            return self._plain(state)
        t0 = time.perf_counter()
        found = self.mrcm(state)
        took = time.perf_counter() - t0
        if self.calls == 1:
            self.root_seconds = round(took, 3)
        if took <= (self.root_limit if self.calls == 1 else self.limit):
            return found
        self.mode, self.switched_at = "share", self.calls
        plain = self._plain(state)
        if plain is not None and (found is None or plain[1] < found[1]):
            return plain
        return found

    def _plain(self, state: PrepState) -> tuple[tuple[Action, ...], Rational] | None:
        found = finish_from(state, self.inst, WEIGHTS)
        if found is None:
            self.fails["no order finishes"] = self.fails.get("no order finishes", 0) + 1
        return found


class TieredFinish:
    """The leaf finisher of D-74 (pilot A2): both finishers, no switch. Every leaf gets the
    plain sharing pass (fast) and, while greedy MRCM (`share_mrcm`, D-65) has used at most
    `share` of the time since the search began, MRCM as well; the cheaper answer counts. Where
    MRCM is fast it checks most leaves (v1); where it is slow it checks a few and the plain pass
    keeps the search fed (pilot A). E-69: no time limit separates the two kinds of problem.
    The root is always checked with both (its MRCM finish fills MRCM's cache)."""

    def __init__(self, inst: Instance, share: float, mrcm: StrongFinish | None = None) -> None:
        self.inst, self.share = inst, share
        self.mrcm = mrcm if mrcm is not None else StrongFinish(inst, lookahead=0)
        self.fails = self.mrcm.fails
        self.mode = "tiered"
        self.start = time.perf_counter()
        self.mrcm_seconds = 0.0
        self.mrcm_calls = self.mrcm_wins = self.calls = 0
        self.root_seconds: float | None = None

    def __call__(self, state: PrepState) -> tuple[tuple[Action, ...], Rational] | None:
        self.calls += 1
        plain = finish_from(state, self.inst, WEIGHTS)
        if plain is None:
            self.fails["no order finishes"] = self.fails.get("no order finishes", 0) + 1
        elapsed = time.perf_counter() - self.start
        if self.calls > 1 and self.mrcm_seconds > self.share * elapsed:
            return plain
        t0 = time.perf_counter()
        found = self.mrcm(state)
        took = time.perf_counter() - t0
        self.mrcm_seconds += took
        self.mrcm_calls += 1
        if self.calls == 1:
            self.root_seconds = round(took, 3)
        if found is not None and (plain is None or found[1] < plain[1]):
            self.mrcm_wins += 1
            return found
        return plain


class SwitchFinish:
    """The leaf finisher of D-75 (pilot A3): one finisher per problem. Greedy MRCM at every
    leaf (v1) until MRCM's leaf finishes, the root's left out (it fills MRCM's cache), average
    more than `mean` seconds, counted over at least `min_leaves` leaves (so one finish slower
    than `mean * min_leaves` switches at once); from then on A2's tiered finisher (D-74),
    keeping MRCM's cache. E-72: mixing finishers in one tree loses v1's quality where MRCM is
    the better finisher, so the switch is for problems where MRCM is slow on average (E-73),
    not after one slow finish (pilot A)."""

    def __init__(self, inst: Instance, mean: float, min_leaves: int, share: float) -> None:
        self.inst, self.mean, self.min_leaves, self.share = inst, mean, min_leaves, share
        self.mrcm = StrongFinish(inst, lookahead=0)
        self.fails = self.mrcm.fails
        self.tiered: TieredFinish | None = None
        self.calls = 0
        self.leaf_seconds = 0.0  # MRCM's leaf finishes before the switch, root left out
        self.root_seconds: float | None = None
        self.switched_at: int | None = None

    @property
    def mode(self) -> str:
        return "share_mrcm" if self.tiered is None else "tiered"

    def __call__(self, state: PrepState) -> tuple[tuple[Action, ...], Rational] | None:
        self.calls += 1
        if self.tiered is not None:
            return self.tiered(state)
        t0 = time.perf_counter()
        found = self.mrcm(state)
        took = time.perf_counter() - t0
        if self.calls == 1:
            self.root_seconds = round(took, 3)
            return found
        self.leaf_seconds += took
        if self.leaf_seconds > self.mean * max(self.calls - 1, self.min_leaves):
            self.switched_at = self.calls
            self.tiered = TieredFinish(self.inst, self.share, self.mrcm)
            self.tiered.calls = 1  # MRCM's cache is filled: no forced root check
        return found


class PlannerJumps:
    """The search's planner jumps (D-73): from a state, for each open target, the planner's
    `k` cheapest candidates for that target's next delivery group (`share._options`), each as
    the run of actions that makes it. With them the search can take the planner's decisions
    (which target next, which recipe) as single moves, besides single actions; the 4 planner
    runs are paths of such moves. MRCM's tree is a recipe only while `finish` uses it (pilot
    A's switch, D-72), so a slow MRCM never slows the jumps."""

    def __init__(self, inst: Instance, k: int, finish: Any) -> None:
        self.inst, self.k, self.finish = inst, k, finish
        self.relaxed = replace(inst, step_limit=share.RELAXED_STEP_LIMIT)

    def _cache(self) -> dict[Any, list[Any]] | None:
        if isinstance(self.finish, AutoFinish):
            return self.finish.mrcm.cache if self.finish.mode == "share_mrcm" else None
        if isinstance(self.finish, StrongFinish):
            return self.finish.cache
        return None

    def __call__(self, state: PrepState) -> list[tuple[Action, ...]]:
        pool = {d.uid for d in state.droplets}
        b = share._Builder(self.relaxed, state, [], WEIGHTS, pool, mrcm=self._cache())
        out: list[tuple[Action, ...]] = []
        for tid in range(len(self.inst.targets)):
            if not state.remaining_demand[tid]:
                continue
            try:
                options = share._options(b, tid)
            except IllegalActionError:
                continue
            for _, _, trial in options[: self.k]:
                run = tuple(trial.actions)
                if len(run) > 1 and run not in out:  # one action is already a candidate
                    out.append(run)
        return out


def run_task(
    task: dict[str, Any], kind: str, sims: int, opts: dict[str, Any], finisher: str = "share"
) -> dict[str, Any]:
    inst = Instance.from_json(task["instance"])
    agent = None
    if kind in ("net", "zero"):
        import torch

        from meda.agents.ppo.policy import PPOAgent

        if opts["agent"] not in _AGENT:
            torch.set_num_threads(1)
            _AGENT[opts["agent"]] = PPOAgent.load(opts["agent"])
        agent = _AGENT[opts["agent"]]
    cfg = {"costs": _load(COSTS), "env": _load("env/sample_prep.yaml")}
    cfg["env"]["hint_source"] = opts["hints"]
    cfg["env"]["T_pad"] = max(cfg["env"]["T_pad"], len(inst.targets))
    env = build_env(cfg, [inst])
    task = dict(task)
    if task["planner"] is None:  # set T: the sharing pass from the empty chip, off the clock
        t0 = time.perf_counter()
        start = finish_from(PrepState.initial(inst), inst, WEIGHTS)
        if start is None:
            raise RuntimeError(f"{inst.instance_id}: the sharing pass does not finish")
        task["planner"], task["planner_seconds"] = float(start[1]), time.perf_counter() - t0
    if task["set"] == "W":  # D-66: the new planner, off the clock; skip if it closes the gap
        t0 = time.perf_counter()
        new = hs.plan_stats(
            inst, "share_mrcm_look3", share_plan(inst, WEIGHTS, lookahead=3, mrcm=True), 0.0
        )["cost"]
        task["new_planner"], task["planner_seconds"] = new, time.perf_counter() - t0
        task["old_planner"] = task["planner"]
        if new is not None and new < task["planner"]:
            task["planner"] = new
        if task["mrcm"] >= task["planner"] - 1e-9:
            rec = {k: v for k, v in task.items() if k != "instance"}
            return rec | {
                "kind": kind,
                "sims": sims,
                "finisher": finisher,
                "search_cost": None,
                "closed_by_planner": True,
            }
    t0 = time.perf_counter()
    finish: (
        MrcmFinish | GraphFinish | StrongFinish | AutoFinish | TieredFinish | SwitchFinish | None
    ) = None
    if finisher == "mrcm":
        finish = MrcmFinish(inst, WEIGHTS)
    elif finisher == "paper":  # D-60: the problem's own published method
        finisher = task["method"]
        finish = GraphFinish(finisher, F_METHODS[finisher], inst, WEIGHTS)
    elif finisher == "strong":  # D-64: lookahead 3 with MRCM's tree (E-58's planner)
        finish = StrongFinish(inst)
    elif finisher == "share_mrcm":  # D-65: MRCM's tree, greedy recipe choice
        finish = StrongFinish(inst, lookahead=0)
    elif finisher == "auto":  # D-72: greedy MRCM while fast, else the plain sharing pass
        finish = AutoFinish(
            inst,
            float(opts.get("switch_seconds", AUTO_LIMIT)),
            float(opts.get("root_switch_seconds", AUTO_ROOT_LIMIT)),
        )
    elif finisher == "tiered":  # D-74: plain pass at every leaf, MRCM within its time share
        finish = TieredFinish(inst, float(opts.get("tier_share", TIER_SHARE)))
    elif finisher == "switch":  # D-75: v1 per problem until MRCM is slow on average, then A2
        finish = SwitchFinish(
            inst,
            float(opts.get("switch_mean", SWITCH_MEAN)),
            int(opts.get("switch_min_leaves", SWITCH_MIN_LEAVES)),
            float(opts.get("tier_share", TIER_SHARE)),
        )
    k = int(opts.get("jumps", 0))  # D-73: planner jumps per open target (0: none, v1)
    search = Search(
        env,
        inst,
        agent,
        t0 + opts["seconds"],
        kind != "zero",
        opts["reward_scale"],
        finish=finish,
        jumps=PlannerJumps(inst, k, finish) if k > 0 else None,
        jump_share=float(opts.get("jump_share", 0.5)),
    )
    out = search.run(sims)
    stats = hs.plan_stats(inst, f"search_{kind}", out["plan"], time.perf_counter() - t0)
    opt, cost = task["optimum"], stats["cost"]
    if opt is not None and cost is not None and cost < opt - 1e-9:  # a bug (rule 9)
        raise RuntimeError(f"{inst.instance_id}: search plan {stats['cost']} < optimum")
    ref = task["mrcm"] if task["set"] != "W" else None  # W: the reference is not the finisher
    if finish is not None and ref is not None and (cost is None or cost > ref + 1e-9):
        raise RuntimeError(f"{inst.instance_id}: search with {finisher} finishes dearer than it")
    rec = {k: v for k, v in task.items() if k != "instance"}
    rec.update(
        kind=kind,
        sims=sims,
        leaves=search.leaves,
        capped=out["capped"],
        played_to_end=out["played_to_end"],
        planner_tail=out["tail"],
        trace=search.trace,
        finisher=finisher,
        finish_fails=finish.fails if finish is not None else {},
        **finish_record(finish, finisher),
        jumps=k,
        jump_share=float(opts.get("jump_share", 0.5)) if k > 0 else None,
        **{f"search_{k}": v for k, v in stats.items()},
    )
    if opts.get("keep_plan"):  # bench.py: the plan itself, in-process only (not JSON)
        rec["plan"] = out["plan"]
    return rec


def finish_record(finish: Any, finisher: str) -> dict[str, Any]:
    """What the pilots' leaf finishers (D-72, D-74, D-75) record about themselves."""
    tiered = finish.tiered if isinstance(finish, SwitchFinish) else finish
    tiered = tiered if isinstance(tiered, TieredFinish) else None
    timed = isinstance(finish, AutoFinish | TieredFinish | SwitchFinish)
    return {
        "finish_mode": finish.mode if timed else finisher,
        "root_finish_seconds": finish.root_seconds if timed else None,
        "switched_at": finish.switched_at
        if isinstance(finish, AutoFinish | SwitchFinish)
        else None,
        "mrcm_calls": tiered.mrcm_calls if tiered else None,
        "mrcm_wins": tiered.mrcm_wins if tiered else None,
        "mrcm_seconds": round(tiered.mrcm_seconds, 3) if tiered else None,
    }


def _star(args: tuple[dict[str, Any], str, int, dict[str, Any], str]) -> dict[str, Any]:
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
    below_mrcm = [r for r in rs if r["mrcm"] is not None and r["search_cost"] < r["mrcm"] - 1e-9]
    ratio_mrcm = [r["search_cost"] / r["mrcm"] for r in rs if r["mrcm"] is not None]
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
        "below_mrcm": len(below_mrcm),
        "ratio_mrcm": float(np.mean(ratio_mrcm)) if ratio_mrcm else None,
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


def _tie(st: dict[str, Any]) -> float | None:
    return st["ratio_opt"] if st["ratio_opt"] is not None else st["ratio_plan"]


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
                and _tie(net) is not None
                and _tie(uniform) is not None
                and _tie(net) < _tie(uniform)
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


def cost_at(rec: dict[str, Any], t: float) -> float:
    """The cheaper of the best planner and the search's best plan found by `t` seconds."""
    return min([rec["planner"], *(c for s, c in rec.get("trace", []) if s <= t)])


def race(groups: dict[tuple[str, str, int], list[dict[str, Any]]]) -> list[str]:
    """Cost over time per set (C pooled), and the reading of D-43."""
    pools: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for (s, kind, sims), rs in groups.items():
        if rs and "trace" in rs[0]:
            pools[("C" if s.startswith("C") else s, kind, sims)] += rs
    if not pools:
        return []
    head = " | ".join(f"{t} s" for t in RACE_AT)
    lines = [
        "",
        "Race (D-43): mean cost / best planner if stopped at t (planner when no plan yet); "
        "with a plan by t in brackets",
        "",
        f"| set | search | sims | n | {head} |",
        "|---" * (4 + len(RACE_AT)) + "|",
    ]
    for key in sorted(pools):
        rs = pools[key]
        cells = []
        for t in RACE_AT:
            ratio = float(np.mean([cost_at(r, t) / r["planner"] for r in rs]))
            have = sum(1 for r in rs if any(s <= t for s, _ in r["trace"]))
            cells.append(f"{ratio:.3f} ({have})")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(rs)} | {' | '.join(cells)} |")
    for s in sorted({k[0] for k in pools}):
        if s == "T":
            lines += race_t(pools)
            continue
        top = max(k[2] for k in pools if k[0] == s)
        net, uni = pools.get((s, "net", top)), pools.get((s, "uniform", top))
        if net and uni:
            rule = "D-56" if s == "D" else "D-43"
            few = (
                f"only {len(net)} problems, reported only (D-56); "
                if s == "D" and len(net) < D_MIN
                else ""
            )
            lines.append(
                f"Race reading on {s} at {top} sims (fixed in advance, {rule}): "
                + few
                + race_verdict(net, uni)
            )
    return lines


def race_t(pools: dict[tuple[str, str, int], list[dict[str, Any]]]) -> list[str]:
    """D-59: `net` and `zero` at every budget against `uniform` at its largest; the network
    makes search faster at 40 targets if any of them passes D-43's rule."""
    unis = [k for k in pools if k[0] == "T" and k[1] == "uniform"]
    if not unis:
        return []
    uni = pools[max(unis, key=lambda k: k[2])]
    keys = sorted(k for k in pools if k[0] == "T" and k[1] in ("net", "zero"))
    if not keys:
        return []
    lines, faster = [], False
    for key in keys:
        found = race_verdict(pools[key], uni, key[1])
        faster |= found.endswith("the network makes search faster")
        lines.append(f"Race on T, {key[1]} at {key[2]} sims: {found}")
    verdict = "the network makes search faster at 40 targets" if faster else "no speed gain"
    return [*lines, f"Race reading on T (fixed in advance, D-59): {verdict}"]


def race_verdict(net: list[dict[str, Any]], uni: list[dict[str, Any]], name: str = "net") -> str:
    by_id = {r["id"]: r for r in uni}
    pairs = [(r, by_id[r["id"]]) for r in net if r["id"] in by_id]
    out, faster = [], False
    for t in (10, 60):
        a = float(np.mean([cost_at(n, t) / n["planner"] for n, _ in pairs]))
        b = float(np.mean([cost_at(u, t) / u["planner"] for _, u in pairs]))
        wins = sum(cost_at(n, t) < cost_at(u, t) - 1e-9 for n, u in pairs)
        losses = sum(cost_at(n, t) > cost_at(u, t) + 1e-9 for n, u in pairs)
        faster |= a <= b - RACE_GAP and wins > losses
        out.append(
            f"at {t} s {name} {a:.3f} vs uniform {b:.3f}, {name} cheaper on {wins}, dearer {losses}"
        )
    verdict = "the network makes search faster" if faster else "no speed gain at these budgets"
    return "; ".join(out) + f": {verdict}"


def _r(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    w = [r for r in recs if r["set"] == "W"]
    recs = [r for r in recs if r["set"] != "W"]
    for r in recs:
        fin = r.get("finisher", "share")
        kind = r["kind"] if fin == "share" else f"{r['kind']}/{fin}"
        groups[(r["set"], kind, r["sims"])].append(r)
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
    c = [k for k in groups if k[0].startswith("C")]
    if c:
        top = max(k[2] for k in c)
        pooled = {
            kind: group_stats([r for k in c if k[1] == kind and k[2] == top for r in groups[k]])
            for kind in ("net", "uniform", "zero")
            if any(k[1] == kind and k[2] == top for k in c)
        }
        for kind, st in pooled.items():
            lines.append(
                f"C all, {kind} {top}: below {st['below']}/{st['n']}, ratio to planner "
                f"{_r(st['ratio_plan'])}, no plan {st['failed']}, capped {st['capped']}"
            )
        net, uni, zero = (pooled.get(kind) for kind in ("net", "uniform", "zero"))
        lines += [f"Reading on C at {top} sims (fixed in advance, D-41): {verdict(net, uni, zero)}"]
    b = {k: group_stats(rs) for k, rs in groups.items() if k[0] == "B"}
    if b:
        lines += ["", "B against MRCM (D-55): below MRCM, ratio to MRCM"]
        for k in sorted(b):
            st = b[k]
            lines.append(f"  {k[1]} {k[2]}: {st['below_mrcm']}/{st['n']}, {_r(st['ratio_mrcm'])}")
        top = max(k[2] for k in b)
        st = b.get(("B", "uniform/mrcm", top))
        if st is not None:
            lines.append(
                f"Reading on B at {top} sims (fixed in advance, D-55): {finisher_verdict(st)}"
            )
    lines += f_reading(groups)
    lines += race(groups)
    lines += w_reading(w)
    return "\n".join(lines)


def w_reading(rs: list[dict[str, Any]]) -> list[str]:
    """Set W (D-66): do published methods still beat the new planner, and the search over it?"""
    if not rs:
        return []
    left = [r for r in rs if not r.get("closed_by_planner")]
    lines = [
        "",
        "W, published wins (D-66): problems a method beats the old planner on -> the new planner "
        "-> the search (uniform, greedy MRCM finishes)",
        "",
        "| source / method | old planner | new planner | search |",
        "|---|---|---|---|",
    ]

    def beats(r: dict[str, Any], m: str, ref: float | None) -> bool:
        c = r["published"].get(m)
        return c is not None and ref is not None and c < ref - 1e-9

    keys = [(s, None) for s in sorted({r["source"] for r in rs})]
    keys += [(None, m) for m in W_METHODS]
    for src, m in keys:
        sub = [r for r in rs if src is None or r["source"] == src]
        ms = [m] if m else list(W_METHODS)
        old = sum(1 for r in sub if any(beats(r, x, r["old_planner"]) for x in ms))
        if not old:
            continue
        new = sum(1 for r in sub if any(beats(r, x, r["planner"]) for x in ms))
        srch = sum(
            1
            for r in sub
            if not r.get("closed_by_planner")
            and any(beats(r, x, min(r["planner"], r["search_cost"] or math.inf)) for x in ms)
        )
        lines.append(f"| {src or m} | {old} | {new} | {srch} |")
    caught = [
        r for r in left if r["search_cost"] is not None and r["search_cost"] <= r["mrcm"] + 1e-9
    ]
    still = sorted(
        (r for r in left if r not in caught),
        key=lambda r: -(min(r["planner"], r["search_cost"] or math.inf) / r["mrcm"]),
    )
    for r in still[:15]:
        mine = min(r["planner"], r["search_cost"] or math.inf)
        lines.append(
            f"  left: {r['source']} {r['id']} {r['method']} {r['mrcm']:g} vs {mine:g} "
            f"({mine / r['mrcm']:.3f})"
        )
    n, k = len(left), len(caught)
    head = (
        f"of {len(rs)} problems, the new planner closes {len(rs) - n}; on the {n} left the search "
        f"is at or below the best published method on {k}"
    )
    if k == n:
        verdict_ = "no published method beats our method on these sets"
    elif k >= W_MOST * n:
        verdict_ = "most published wins closed; the rest are listed above"
    else:
        verdict_ = "published methods keep real wins; look at the ones left"
    return [*lines, "", f"Reading on W (fixed in advance, D-66): {head}: {verdict_}"]


def f_reading(groups: dict[tuple[str, str, int], list[dict[str, Any]]]) -> list[str]:
    """Set F (D-60): per method, the search with that method's finishes against the method."""
    f = {k: group_stats(rs) for k, rs in groups.items() if k[0].startswith("F-")}
    if not f:
        return []
    lines = ["", "F against each method (D-60): below the method, ratio to it"]
    for k in sorted(f):
        st = f[k]
        lines.append(
            f"  {k[0]} {k[1]} {k[2]}: {st['below_mrcm']}/{st['n']}, {_r(st['ratio_mrcm'])}"
        )
    improved = 0
    for m in F_METHODS:
        tops = [k for k in f if k[0] == f"F-{m}" and k[1] == f"uniform/{m}"]
        if not tops:
            continue
        st = f[max(tops, key=lambda k: k[2])]
        improved += st["below_mrcm"] >= FOUND_SHARE * st["n"]
        lines.append(f"Reading on F-{m} (D-60): {finisher_verdict(st, m.upper())}")
    if improved >= F_MOST:
        overall = "the search improves most published methods (with MRCM, E-55)"
    elif improved:
        overall = "the search improves some of these methods"
    else:
        overall = "the search improves none of these methods"
    return [*lines, f"Reading on F (fixed in advance, D-60): {improved} of 4 improved: {overall}"]


def finisher_verdict(st: dict[str, Any], name: str = "MRCM") -> str:
    n, k = st["n"], st["below_mrcm"]
    if k >= FOUND_SHARE * n:
        return f"below {name} on {k} of {n}: the search improves {name}"
    if k > 0:
        return f"below {name} on {k} of {n}: a weak gain"
    return f"below {name} on none: its misses are not reachable by lookahead with its own finishes"


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
    finishers = args.get("finishers", "share").split(",")
    runs = [(kind, s) for kind in ("net", "zero") for s in sims] if opts["agent"] else []
    runs += [("uniform", uniform)] if uniform else []
    todo = [
        (t, kind, s, opts, fin)
        for t in found
        for kind, s in runs
        for fin in (finishers if kind != "zero" else ["share"])
    ]
    out = Path(args.get("out", "notes/search/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(
        f"{len(found)} problems x {runs} x {finishers} = {len(todo)} searches, {workers} workers",
        flush=True,
    )
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
