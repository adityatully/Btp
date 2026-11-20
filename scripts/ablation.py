"""Justification sweeps of method-v3's search (D-98, `notes/justification_audit.md` Part 2).

Method-v3 (SPEC 0.44) on a fixed dev subset, one setting changed at a time; the test sets stay
untouched. This script only imports `search_check` and `bench`: every variant is a subclass of
`search_check.Search` or a module setting, so the method's own code is unchanged.

**Subset** (`subset=build`, written once to `notes/ablation/subset_v1.json`): 110 problems from
the dev sets, drawn with seed `SUBSET_SEED` and spread evenly over each set's cells: val s1 10,
s2 10, s3 20, s4 20, s5 20 (`_v2`), `val_emrcm_v1` 30 (over its 8 cells: N = 3 8, N = 4 12,
N = 5 10). Fluid goal (`costs/geometric`), as the finals.

**Runs** (`runs=`, comma-separated groups or configs; each is (config, repeat)):
- `J1`, the noise floor: method-v3 three times (`v3` repeats 0-2) and with random tie-breaking
  under two seeds (`seed1`, `seed2`).
- `J2`, two curves (D-101): sims per move {8, 16, 32, 64, 128, 256, 512, 1024, 2048} at C_PUCT
  1.5, and C_PUCT {0.25, 0.5, 1, 1.5, 2, 3, 4, 8} at 256 sims; `v3` repeat 0 is (256, 1.5). Any
  other pair runs by name (`sims64_c0.5`). The summary prints each curve with the moves the
  tree committed and the plateau (values within 0.5% of the best).
- `J3`, search mechanics one at a time: best-leaf backup, first-play value {0, -1}, value
  scale from the first plan, no floor, dead-end value {-2, -0.5}, play the move with the
  cheapest plan, never pick a Discard. Join with J2's chosen point by '+'
  (`backup_max+sims64_c0.5`).
- `J5`, the finisher switch, only on problems where it fires (val s5, `val_emrcm_v1` N 4-5):
  switch mean {5, 10, 40} s, leaves before switching {1, 10}, MRCM's time share {0.25, 0.75}.

Each record: config, repeat, problem, final cost, trace (CPU s, cost), leaves, moves the tree
committed (`depth`, E-81) and plan length, capped, finisher mode and switch, seconds; and,
measured without changing anything, leaf finishes longer than the step limit (S14), leaves
that repeat an earlier leaf's state (S21), and how often the split tree's and MRCM's search
budgets ran out (P15, P16). Resumes: a (config, repeat, problem) already in `out` is skipped.

**Reading, fixed in advance (D-98):** per config, the paired cost ratio to `v3` repeat 0 per
problem, its mean and 95% bootstrap CI, losses and wins. The **noise band** is the largest
|mean ratio - 1| among J1's other runs (v3 repeats 1-2, seed1, seed2) against v3 repeat 0, per
stage group (s1-s2, s3, s4-s5, emrcm). A config is **level** when its CI contains 1 or its mean
lies inside the band; **better / worse** when the CI excludes 1 and the mean is outside it.
Method-v3's value is justified when its neighbours on each side are level or worse.

    OMP_NUM_THREADS=1 uv run python scripts/ablation.py subset=build
    OMP_NUM_THREADS=1 uv run python scripts/ablation.py runs=J1 workers=16 \\
        out=outputs/ablation/j1.jsonl
    uv run python scripts/ablation.py summary=outputs/ablation/j1.jsonl
    uv run python scripts/ablation.py summary=outputs/ablation/j2.jsonl \\
        noise=outputs/ablation/j1.jsonl

Each later run (J2, J3, J4, J5, J7) holds its own method-v3 run and is read against it, with
J1's noise band (`noise=`). J1 itself ran on `e3927e8` / `05f8bde`, before the memo caches were
cleared per run (E-104), so its method-v3 runs are not the reference for later ones (E-105).
"""

from __future__ import annotations

import itertools
import json
import math
import signal
import subprocess
import sys
import zlib
from collections import defaultdict
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import justify_planner as jp  # noqa: E402  (counts the tree's and MRCM's budget runs)
import search_check as sc  # noqa: E402
from meda.core.actions import Discard  # noqa: E402
from meda.core.costs import CostWeights  # noqa: E402
from meda.core.rules import replay  # noqa: E402
from meda.generators.benchmark_io import BenchmarkSet  # noqa: E402
from meda.solvers.baselines import share  # noqa: E402

SUBSET = ROOT / "notes/ablation/subset_v1.json"
SUBSET_SEED = 20261118
# (set, count): the dev sets the subset is drawn from, spread evenly over each set's cells
SUBSET_SETS = (
    ("val_s1_v2", 10),
    ("val_s2_v2", 10),
    ("val_s3_v2", 20),
    ("val_s4_v2", 20),
    ("val_s5_v2", 20),
    ("val_emrcm_v1", 30),
)
SECONDS, SIMS = 900.0, 256  # method-v3 (bench.METHOD)
# J2's curves (D-101): each number swept well past its value, the other held at method-v3's
SIMS_CURVE = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
CPUCT_CURVE = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)


J3 = {  # search mechanics, one at a time (S6-S10, S12, S20)
    "backup_max": {"backup": "max"},  # a move's value = its best leaf, not the mean
    "fpu0": {"fpu": 0.0},  # an unvisited move's value (method-v3: the parent's)
    "fpu_m1": {"fpu": -1.0},
    "scale_first": {"scale": "first"},  # c0 = the first plan's cost, not the root's finish
    "nofloor": {"floor": None},  # no floor at -1 on a leaf's value
    "dead_m2": {"dead": -2.0},  # a dead end's value (method-v3: -1)
    "dead_m05": {"dead": -0.5},
    "move_cheapest": {"move": "cheapest"},  # play the move with the cheapest plan through it
    "no_discard": {"discard": False},  # the search never picks a Discard
}
J5 = {  # the finisher switch (F3-F5), only where it fires
    "switch5": {"switch_mean": 5.0, "only": "switch"},
    "switch10": {"switch_mean": 10.0, "only": "switch"},
    "switch40": {"switch_mean": 40.0, "only": "switch"},
    "leaves1": {"switch_min_leaves": 1, "only": "switch"},
    "leaves10": {"switch_min_leaves": 10, "only": "switch"},
    "tier25": {"tier_share": 0.25, "only": "switch"},
    "tier75": {"tier_share": 0.75, "only": "switch"},
}
J4 = {  # other ways to spend the same time over the same finisher (S1)
    "beam1": {"beam": 1},  # the rollout algorithm: play the move whose finish is cheapest
    "beam4": {"beam": 4},
    "beam16": {"beam": 16},
}
J7 = {  # planner settings inside the search's finisher (P1, P5, P6); J6's patches
    "leaf_chain2": {"planner": "chain2"},
    "leaf_chain4": {"planner": "chain4"},
    "leaf_chain1": {"planner": "chain1"},  # depth 1 = one mix only: no chain recipe
    "leaf_chain5": {"planner": "chain5"},  # E-110: depth 4 cheaper in the search; where it stops
    # #9: a recipe taken out of the whole method (first plan and every leaf finish)
    "leaf_drop_tree": {"planner": "drop_tree"},
    "leaf_drop_mrcm": {"planner": "drop_mrcm"},
    "leaf_blend2": {"planner": "blend2"},
    "leaf_blend8": {"planner": "blend8"},
    "orders_given": {"orders": ("given",)},
    "orders_dear": {"orders": ("dear",)},
    "orders_dear_cheap": {"orders": ("dear", "cheap")},
    # E-109: more orders than the 4 rules (Q-163): the rules' orders, then n more distinct
    # ones, seeded per problem (every permutation shuffled up to 4 targets, else random)
    "more_orders4": {"more_orders": 4},
    "more_orders12": {"more_orders": 12},
}
# J8: the problem's own numbers (E2, E3). Weight profiles in configs/costs/sensitivity; each
# J8 run also runs the 4 planner runs under the same weights and limit (`baselines`)
J8_PROFILES = ("fluid16", "fluid256", "mix0", "mix4", "waste0", "waste4")
J8 = {
    "j8_base": {"baselines": True},
    **{f"j8_{p}": {"costs": f"costs/sensitivity/{p}.yaml", "baselines": True} for p in J8_PROFILES},
    "j8_step_x075": {"step_scale": 0.75, "baselines": True},
    "j8_step_x2": {"step_scale": 2.0, "baselines": True},
}
PLANNER_CAP = 1800.0  # CPU s per planner run in J8 (share_mrcm_look3 ran up to 7.8 h, E-102)


def configs() -> dict[str, dict[str, Any]]:
    """Every named config: changes against method-v3 (none = method-v3 itself). A run may join
    configs with '+' (e.g. `backup_max+sims64_c0.5`, J3 at J2's chosen point)."""
    out: dict[str, dict[str, Any]] = {"v3": {}, "seed1": {"seed": 1}, "seed2": {"seed": 2}}
    for s in SIMS_CURVE:
        for c in CPUCT_CURVE:
            if (s, c) != (SIMS, sc.C_PUCT):
                out[f"sims{s}_c{c:g}"] = {"sims": s, "cpuct": c}
    return out | J3 | J5 | J4 | J7 | J8


def config(name: str) -> dict[str, Any]:
    known = configs()
    merged: dict[str, Any] = {}
    for part in name.split("+"):
        if part not in known:
            raise ValueError(f"unknown config '{part}' (groups {sorted(GROUPS)}, {sorted(known)})")
        merged |= known[part]
    return merged


GROUPS: dict[str, list[tuple[str, int]]] = {
    "J1": [("v3", 0), ("v3", 1), ("v3", 2), ("seed1", 0), ("seed2", 0)],
    "J2": [("v3", 0)]
    + [(f"sims{s}_c1.5", 0) for s in SIMS_CURVE if s != SIMS]
    + [(f"sims256_c{c:g}", 0) for c in CPUCT_CURVE if c != 1.5],
    "J3": [("v3", 0)] + [(name, 0) for name in J3],
    "J5": [("v3", 0)] + [(name, 0) for name in J5],
    "J4": [("v3", 0)] + [(name, 0) for name in J4],
    "J7": [("v3", 0)] + [(name, 0) for name in J7 if name.startswith("leaf_")],
    # D-96's search pilot: only if the order check (scripts/order_check.py) passes its bar
    "J7o": [("v3", 0)] + [(name, 0) for name in J7 if name.startswith("orders_")],
    "J7r": [("v3", 0), ("leaf_chain1", 0), ("leaf_drop_tree", 0), ("leaf_drop_mrcm", 0)],
    "J7m": [("v3", 0)] + [(name, 0) for name in J7 if name.startswith("more_orders")],
    "J8": [(name, 0) for name in J8],
}


def switch_problem(entry: dict[str, Any]) -> bool:
    """Where the finisher switch fires (E-69, E-73): 4-5 fluid mixtures and stage 5."""
    return entry["set"] == "val_s5_v2" or (
        entry["set"] == "val_emrcm_v1" and entry["cell"].startswith(("N4", "N5"))
    )


# --------------------------------------------------------------------------- subset


def build_subset() -> list[dict[str, Any]]:
    """The fixed subset: per set, `count` problems spread evenly over its cells (a remainder
    goes to the first cells in sorted order), drawn with `SUBSET_SEED`."""
    rng = np.random.default_rng(SUBSET_SEED)
    picked: list[dict[str, Any]] = []
    for name, count in SUBSET_SETS:
        bench = BenchmarkSet.read(name)
        cells: dict[str, list[str]] = defaultdict(list)
        for inst in bench.instances:
            cells[str(inst.metadata.get("cell", "-"))].append(inst.instance_id)
        names = sorted(cells)
        share = [count // len(names) + (i < count % len(names)) for i in range(len(names))]
        for cell, n in zip(names, share, strict=True):
            ids = sorted(cells[cell])
            if n > len(ids):
                raise ValueError(f"{name}: cell {cell} has {len(ids)} problems, {n} wanted")
            for i in sorted(rng.choice(len(ids), size=n, replace=False)):
                picked.append({"set": name, "cell": cell, "id": ids[int(i)]})
    return picked


def stage(entry: dict[str, Any]) -> str:
    """The stage group a problem is read in."""
    s = entry["set"]
    if s.startswith("val_emrcm"):
        return "emrcm"
    if s.startswith(("val_s1", "val_s2")):
        return "s1-s2"
    return "s3" if s.startswith("val_s3") else "s4-s5"


# --------------------------------------------------------------------------- search variants

_SETTINGS: dict[str, Any] = {}  # this worker's config (set before each task)
_LAST: list[Any] = []  # the last search built in this worker


class AblationSearch(sc.Search):
    """`search_check.Search` with the config's settings. With no setting every method is
    `search_check`'s own, so method-v3 runs unchanged; each J3 variant changes one line of a
    copied method. Keeps a handle on itself so the record can read the moves it committed, and
    counts (always, without changing anything): leaf finishes longer than the step limit
    (S14) and leaves that repeat an earlier leaf's state (S21)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._c0 = 1.0
        super().__init__(*args, **kwargs)
        if _SETTINGS.get("seed") is not None:
            self.rng = np.random.default_rng(int(_SETTINGS["seed"]))
        self.over_limit = 0
        self.finishes = 0
        self.seen: set[Any] = set()
        self.repeats = 0
        inner = self.finish

        def counted(state: Any) -> Any:
            found = inner(state)
            self.finishes += 1
            if found is not None and state.step + len(found[0]) > self.inst.step_limit:
                self.over_limit += 1
            return found

        self.finish = counted
        _LAST[:] = [self]

    # S6: the value's scale
    @property
    def c0(self) -> float:
        if _SETTINGS.get("scale") == "first" and self.trace:
            return max(self.trace[0][1], 1.0)  # the plain pass's plan, offered before the root
        return self._c0

    @c0.setter
    def c0(self, value: float) -> None:
        self._c0 = value

    def score(self, cost: float) -> float:
        if "floor" in _SETTINGS and _SETTINGS["floor"] is None:  # S7
            return (self.c0 - cost) / self.c0
        return super().score(cost)

    def leaf(self, obs: dict[str, Any], actions: tuple[Any, ...], info: dict | None) -> Any:
        node = super().leaf(obs, actions, info)
        state = self.env.state
        node.key = state
        if state in self.seen:
            self.repeats += 1
        else:
            self.seen.add(state)
        if "dead" in _SETTINGS and node.value == -1.0 and not np.isfinite(node.best):  # S8
            node.value = float(_SETTINGS["dead"])
        if _SETTINGS.get("discard") is False and not node.terminal:  # S20
            node.discard = np.array([isinstance(a, Discard) for a in self.env.legal], dtype=bool)
        return node

    def simulate(self, root: Any) -> None:
        """`search_check.Search.simulate` (no planner jumps, as in method-v3), with S9's
        backup, S10's first-play value and S20's Discard mask as options."""
        if not ({"backup", "fpu", "discard"} & set(_SETTINGS)):
            return super().simulate(root)
        best_backup = _SETTINGS.get("backup") == "max"
        path: list[tuple[Any, int]] = []
        node = root
        while True:
            if node.jumps is None:
                self.add_jumps(node)
            n = node.visits.sum()
            if best_backup:
                if not hasattr(node, "top"):
                    node.top = np.full(len(node.prior), -np.inf)
                seen = node.top
            else:
                seen = node.total / np.maximum(node.visits, 1)
            unvisited = float(_SETTINGS.get("fpu", node.value))
            q = np.where(node.visits > 0, seen, unvisited)
            u = sc.C_PUCT * node.prior * math.sqrt(n + 1) / (1 + node.visits)
            score = q + u
            mask = getattr(node, "discard", None)
            if mask is not None and len(mask) == len(score) and not mask.all():
                score = np.where(mask, -np.inf, score)
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
            if best_backup:
                if not hasattr(nd, "top"):
                    nd.top = np.full(len(nd.prior), -np.inf)
                nd.top[i] = max(nd.top[i], end.value)
            nd.best = min(nd.best, end.best)

    def _pass(self, sims: int, root_only: bool = False) -> dict[str, Any]:
        """J4 (S1): with `beam` = W, a beam search over the same moves, finisher and time
        instead of PUCT. Every child of every beam state gets a finish (its value is the
        cheapest whole plan through it); the W cheapest distinct states form the next beam.
        W = 1 is the rollout algorithm (one-step lookahead with the planner as base policy)."""
        if "beam" not in _SETTINGS:
            return super()._pass(sims, root_only)
        width = int(_SETTINGS["beam"])
        obs, _ = self.env.reset(options={"instance": self.inst})
        first = sc.finish_from(sc.PrepState.initial(self.inst), self.inst, sc.WEIGHTS)
        if first is not None and len(first[0]) <= self.inst.step_limit:
            self._offer(float(first[1]), first[0], len(first[0]))
        root = self.leaf(obs, (), None)
        beam, capped, depth = [root], False, root
        while beam and not capped:
            children = []
            for node in beam:
                self.env.restore(node.snap)
                for i in range(len(self.env.legal)):
                    if sc.CLOCK() > self.deadline:
                        capped = True
                        break
                    children.append(self.child(node, i))
                if capped:
                    break
            keep: dict[Any, Any] = {}
            for ch in sorted(children, key=lambda c: (c.best, len(c.actions))):
                if ch.terminal or not np.isfinite(ch.best) or ch.key in keep:
                    continue
                keep[ch.key] = ch
                if len(keep) == width:
                    break
            beam = list(keep.values())
            if beam and not capped:
                depth = beam[0]
        self.path = depth.actions
        out = {"capped": capped, "played_to_end": not beam and not capped}
        if self.best is None:
            return {"plan": None, "tail": 0, **out}
        return {"plan": self.best[1], "tail": self.best[2], **out}

    def _descend(self, root: Any, sims: int) -> dict[str, Any]:
        """`search_check.Search._descend` (no jumps, no sample collection, as in method-v3),
        with S12's move choice: the move with the cheapest whole plan seen through it."""
        if _SETTINGS.get("move") != "cheapest":
            return super()._descend(root, sims)
        capped = False
        while not root.terminal:
            for _ in range(sims):
                if sc.CLOCK() > self.deadline:
                    capped = True
                    break
                self.simulate(root)
            if capped or root.visits.sum() == 0:
                break
            visited = [j for j in root.children if root.visits[j] > 0]
            i = min(visited, key=lambda j: (root.children[j].best, -root.visits[j], j))
            root = root.children[i]
        self.path = root.actions
        out = {"capped": capped, "played_to_end": root.terminal}
        if self.best is None:
            return {"plan": None, "tail": 0, **out}
        return {"plan": self.best[1], "tail": self.best[2], **out}


sc.Search = AblationSearch  # type: ignore[misc]  # run_task builds the search by this name


_BASE = {"weights": sc.WEIGHTS, "costs": sc.COSTS, "orders": share.named_orders}


def _set_problem(cfg: dict[str, Any]) -> CostWeights:
    """Put back method-v3's weights, planner and orders, then set the config's (J7, J8)."""
    w = _BASE["weights"]
    sc.COSTS = _BASE["costs"]
    if "costs" in cfg:  # J8: every module's weights and the env's costs, as goal_check does
        w = CostWeights.from_dict(yaml.safe_load((ROOT / "configs" / cfg["costs"]).read_text()))
        sc.COSTS = cfg["costs"]
    sc.WEIGHTS = sc.hs.WEIGHTS = w
    jp.apply(jp.configs()[cfg["planner"]] if "planner" in cfg else {})
    share.named_orders = _BASE["orders"]
    if "orders" in cfg:  # J7: only the target orders these rules give (D-96)
        allowed = set(cfg["orders"])

        def limited(instance: Any, weights: Any) -> Any:
            every = _BASE["orders"](instance, weights)
            return [o for o in every if allowed & set(o[0])] or every[:1]

        share.named_orders = limited
    if "more_orders" in cfg:  # E-109, Q-163: the rules' orders and n more
        share.named_orders = more_orders(int(cfg["more_orders"]))
    return w


def more_orders(n: int) -> Any:
    """`named_orders` with n extra distinct target orders after the rules' (Q-163): every
    other permutation in a seeded shuffle when there are at most 4 targets, else seeded random
    ones, as J6's order curve (`justify_planner.order_list`) draws them."""
    memo: dict[str, Any] = {}

    def named(instance: Any, weights: Any) -> Any:
        if instance.instance_id in memo:
            return memo[instance.instance_id]
        out = list(_BASE["orders"](instance, weights))
        seen = {tuple(o) for _, o in out}
        k = len(instance.targets)
        seed = zlib.crc32(instance.instance_id.encode())
        if math.factorial(k) <= jp.ALL_ORDERS:
            perms = [list(p) for p in itertools.permutations(range(k))]
            rng = np.random.default_rng([seed, 1])
            extra = [perms[int(i)] for i in rng.permutation(len(perms))]
        else:
            rng = np.random.default_rng([seed, 0])
            extra = [[int(i) for i in rng.permutation(k)] for _ in range(4 * n)]
        most = len(out) + n
        for order in extra:
            if len(out) >= most:
                break
            if tuple(order) not in seen:
                seen.add(tuple(order))
                out.append((("extra",), order))
        memo[instance.instance_id] = out
        return out

    return named


def baselines(inst: Any, w: CostWeights) -> dict[str, dict[str, Any]]:
    """J8: the 4 planner runs (bench's baselines) under the run's weights and limit, each
    capped at `PLANNER_CAP` CPU s."""
    runs = {
        "share": lambda: share.share_plan(inst, w),
        "share_look3": lambda: share.share_plan(inst, w, lookahead=3),
        "share_orders": lambda: sc.hs.share_orders(inst, 32),
        "share_mrcm_look3": lambda: share.share_plan(inst, w, lookahead=3, mrcm=True),
    }
    out = {}
    for name, make in runs.items():
        signal.signal(signal.SIGPROF, jp._alarm)
        signal.setitimer(signal.ITIMER_PROF, PLANNER_CAP)
        t0, plan, capped = sc.CLOCK(), None, False
        try:
            plan = make()
        except jp._Cap:
            capped = True
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
        cost = None
        if plan is not None:
            _, total, _ = replay(inst, plan)
            cost = float(w.scalar(total, inst.fluids))
        out[name] = {"cost": cost, "seconds": round(sc.CLOCK() - t0, 3), "capped": capped}
    return out


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    """One (config, repeat, problem): method-v3 with the config's changes."""
    cfg = config(task["config"])
    _SETTINGS.clear()
    _SETTINGS.update(cfg)
    sc.C_PUCT = float(cfg.get("cpuct", 1.5))
    w = _set_problem(cfg)
    jp.cold()  # every run starts with empty memo caches (D-98)
    inst = BenchmarkSet.read(task["set"]).by_id(task["id"])
    if "step_scale" in cfg:  # J8: the step limit scaled, never below the witness plan
        witness = int(inst.metadata.get("witness_actions", 1))
        limit = max(witness, math.ceil(float(cfg["step_scale"]) * inst.step_limit))
        inst = replace(inst, step_limit=limit)
    search_task = {
        "set": "J",
        "id": task["id"],
        "cell": task["cell"],
        "instance": inst.to_json(),
        "optimum": None,
        "planner": None,  # the plain pass from the empty chip, off the clock (search_check)
        "mrcm": None,
        "greedy": None,
    }
    opts = {
        "agent": None,
        "seconds": float(task.get("seconds", SECONDS)),
        "hints": "planner",
        "reward_scale": 0.01,
        "tier_share": float(cfg.get("tier_share", sc.TIER_SHARE)),
        "switch_mean": float(cfg.get("switch_mean", sc.SWITCH_MEAN)),
        "switch_min_leaves": int(cfg.get("switch_min_leaves", sc.SWITCH_MIN_LEAVES)),
        "first_plan": True,
    }
    sims = int(cfg.get("sims", SIMS))
    jp.COUNTS.update(tree_budget=0, mrcm_budget=0)
    try:
        try:
            rec = sc.run_task(search_task, "uniform", sims, opts, "switch")
        except RuntimeError as e:  # #9: without a recipe the pass may not finish at all
            if "does not finish" not in str(e):
                raise
            return {
                "config": task["config"],
                "repeat": task["repeat"],
                "set": task["set"],
                "cell": task["cell"],
                "id": task["id"],
                "commit": task["commit"],
                "settings": cfg,
                "cost": None,
                "no_plan": str(e),
                "trace": [],
                "leaves": 0,
                "depth": 0,
                "capped": False,
            }
        search = _LAST[0]
        jp.apply({})
        share.named_orders = _BASE["orders"]
        base = baselines(inst, w) if cfg.get("baselines") else None
    finally:
        _set_problem({})
    return {
        "config": task["config"],
        "repeat": task["repeat"],
        "set": task["set"],
        "cell": task["cell"],
        "id": task["id"],
        "commit": task["commit"],
        "settings": cfg,
        "sims": sims,
        "cpuct": sc.C_PUCT,
        "cost": rec["search_cost"],
        "trace": rec["trace"],
        "leaves": rec["leaves"],
        "depth": len(search.path),
        "plan_length": rec.get("search_steps"),
        "capped": rec["capped"],
        "played_to_end": rec["played_to_end"],
        "finish_mode": rec["finish_mode"],
        "switched_at": rec["switched_at"],
        "seconds": rec.get("search_seconds"),
        "plain_pass": rec["planner"],
        "finishes": search.finishes,
        "over_limit": search.over_limit,  # S14: leaf finishes longer than the step limit
        "repeats": search.repeats,  # S21: leaves whose state an earlier leaf had
        "tree_budget": jp.COUNTS["tree_budget"],  # P15
        "mrcm_budget": jp.COUNTS["mrcm_budget"],  # P16
        "step_limit": inst.step_limit,
        "baselines": base,
    }


# --------------------------------------------------------------------------- summary


def boot_ci(xs: list[float], seed: int = 0, n: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(xs)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cost_at(rec: dict[str, Any], t: float) -> float | None:
    """The cheapest whole plan in the trace by CPU second t."""
    found = [c for s, c in rec["trace"] if s <= t]
    return min(found) if found else None


def compare(base: dict[str, dict], other: dict[str, dict]) -> dict[str, Any] | None:
    """Paired ratios other / base over the problems both finished."""
    ids = [i for i in base if i in other and base[i]["cost"] and other[i]["cost"] is not None]
    if not ids:
        return None
    r = [other[i]["cost"] / base[i]["cost"] for i in ids]
    lo, hi = boot_ci(r)
    return {
        "n": len(ids),
        "mean": float(np.mean(r)),
        "lo": lo,
        "hi": hi,
        "worse": sum(x > 1 + 1e-9 for x in r),
        "better": sum(x < 1 - 1e-9 for x in r),
    }


def noise_band(by: dict[tuple[str, int], dict[str, dict]], group: str) -> float | None:
    """The largest |mean ratio - 1| of J1's other runs against v3 repeat 0, in `group`."""
    base = by.get(("v3", 0))
    if base is None:
        return None
    gaps = []
    for key in GROUPS["J1"][1:]:
        if key in by:
            c = compare(
                {i: r for i, r in base.items() if r["group"] == group},
                {i: r for i, r in by[key].items() if r["group"] == group},
            )
            if c:
                gaps.append(abs(c["mean"] - 1))
    return max(gaps) if gaps else None


def verdict(c: dict[str, Any], band: float | None) -> str:
    inside = band is not None and abs(c["mean"] - 1) <= band
    if (c["lo"] <= 1 <= c["hi"]) or inside:
        return "level"
    return "worse" if c["mean"] > 1 else "better"


def _load(paths: list[Path]) -> dict[tuple[str, int], dict[str, dict]]:
    entries = {e["id"]: e for e in json.loads(SUBSET.read_text())}
    by: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            r["group"] = stage(entries[r["id"]])
            key = (r["config"], r["repeat"])
            if r["id"] in by[key]:
                raise ValueError(f"{path}: {key} {r['id']} twice; read one run's file at a time")
            by[key][r["id"]] = r
    return by


def summarise(paths: list[Path], noise: Path | None = None) -> str:
    """One run's records against its own method-v3 run (v3 repeat 0, in the same files),
    read with the noise band of J1's records (`noise`; default: the records themselves).
    Each run file holds its own method-v3 run, so a run whose code or caches differ from J1's
    is still compared like with like (D-98, E-105)."""
    by = _load(paths)
    groups = ("s1-s2", "s3", "s4-s5", "emrcm", "all")
    ref = _load([noise]) if noise else by
    bands = {g: noise_band(ref, g) for g in groups[:-1]}
    lines = [
        "| config | rep | group | n | mean ratio to v3 [95% CI] | worse / better | verdict |"
        " depth (median) | leaves (median) | capped |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    base = by.get(("v3", 0), {})
    for key in sorted(by):
        for g in groups:
            rs = {i: r for i, r in by[key].items() if g == "all" or r["group"] == g}
            if not rs:
                continue
            mine = {i: r for i, r in base.items() if i in rs}
            c = compare(mine, rs) if key != ("v3", 0) else None
            known = [b for b in bands.values() if b is not None]
            band = bands.get(g) if g != "all" else max(known, default=None)
            ratio = f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]" if c else "reference"
            wb = f"{c['worse']} / {c['better']}" if c else "-"
            v = verdict(c, band) if c else "-"
            depth = int(np.median([r["depth"] for r in rs.values()]))
            leaves = int(np.median([r["leaves"] for r in rs.values()]))
            capped = sum(bool(r["capped"]) for r in rs.values())
            none = sum(r["cost"] is None for r in rs.values())
            lines.append(
                f"| {key[0]} | {key[1]} | {g} | {len(rs)} | {ratio} | {wb} | {v} | {depth} |"
                f" {leaves} | {capped}{f', {none} no plan' if none else ''} |"
            )
    lines += ["", *j2_curves(by)]
    lines += [
        "",
        "Noise band (D-98): "
        + ", ".join(f"{g} {b:.4f}" if b is not None else f"{g} -" for g, b in bands.items()),
    ]
    return "\n".join(lines)


def j2_curves(by: dict[tuple[str, int], dict[str, dict]]) -> list[str]:
    """J2's curves (D-101): per value, the mean of cost / the per-problem best over the curve
    (problems every value finished), the median moves committed and leaves. Under a fixed time
    budget more simulations are not always better, so the reading is the plateau: every value
    within 0.5% of the curve's best, and whether method-v3's value is on it."""
    lines = ["## J2 curves (D-101): mean cost / best of the curve; depth; leaves", ""]
    for name, pts in (
        (
            "sims per move (C_PUCT 1.5)",
            [(v, "v3" if v == SIMS else f"sims{v}_c1.5") for v in SIMS_CURVE],
        ),
        ("C_PUCT (256 sims)", [(v, "v3" if v == 1.5 else f"sims256_c{v:g}") for v in CPUCT_CURVE]),
    ):
        runs = [by.get((cfg, 0), {}) for _, cfg in pts]
        if sum(bool(r) for r in runs) < 2:
            continue
        have = [(v, r) for (v, _), r in zip(pts, runs, strict=True) if r]
        for g in ("s1-s2", "s3", "s4-s5", "emrcm", "all"):
            ids = [
                i
                for i, r in have[0][1].items()
                if (g == "all" or r["group"] == g)
                and all(i in rs and rs[i]["cost"] is not None for _, rs in have)
            ]
            if not ids:
                continue
            best = {i: min(rs[i]["cost"] for _, rs in have) for i in ids}
            cells, ratios = [], {}
            for v, rs in have:
                r = float(np.mean([rs[i]["cost"] / best[i] for i in ids]))
                ratios[v] = r
                depth = int(np.median([rs[i]["depth"] for i in ids]))
                leaves = int(np.median([rs[i]["leaves"] for i in ids]))
                mark = "*" if v in (SIMS, 1.5) and (v == SIMS) == name.startswith("sims") else ""
                cells.append(f"{v:g}{mark}: {r:.4f} (d {depth}, {leaves} leaves)")
            top = min(ratios.values())
            plateau = [f"{v:g}" for v, r in ratios.items() if r <= top + 0.005]
            lines.append(
                f"- **{name}**, {g} (n {len(ids)}): "
                + "; ".join(cells)
                + f" -> plateau (within 0.5% of the best): {', '.join(plateau)}"
            )
    return lines


# --------------------------------------------------------------------------- main


def expand(runs: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    names = configs()
    for item in runs.split(","):
        if item in GROUPS:
            out += GROUPS[item]
        elif all(part in names for part in item.split("+")):
            out.append((item, 0))
        else:
            raise ValueError(
                f"unknown run '{item}' (groups {sorted(GROUPS)}, configs {sorted(names)})"
            )
    return list(dict.fromkeys(out))


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if args.get("subset") == "build":
        if SUBSET.exists():
            raise FileExistsError(f"{SUBSET} is frozen; delete it by hand to rebuild")
        SUBSET.parent.mkdir(parents=True, exist_ok=True)
        picked = build_subset()
        SUBSET.write_text(json.dumps(picked, indent=1) + "\n")
        print(f"{len(picked)} problems -> {SUBSET}")
        return
    if "summary" in args:
        noise = Path(args["noise"]) if "noise" in args else None
        print(summarise([Path(p) for p in args["summary"].split(",")], noise))
        return
    entries = json.loads(SUBSET.read_text())
    if "sets" in args:  # light tests: only these sets
        entries = [e for e in entries if e["set"] in args["sets"].split(",")]
    if "limit" in args:  # light tests: the first n problems of each set
        per: dict[str, int] = defaultdict(int)
        kept = []
        for e in entries:
            if per[e["set"]] < int(args["limit"]):
                kept.append(e)
                per[e["set"]] += 1
        entries = kept
    out = Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["repeat"], r["id"]))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    todo = [
        {
            **e,
            "config": c,
            "repeat": rep,
            "commit": commit,
            **({"seconds": float(args["seconds"])} if "seconds" in args else {}),
        }
        for c, rep in expand(args["runs"])
        for e in entries
        if (c, rep, e["id"]) not in done
        and (config(c).get("only") != "switch" or switch_problem(e))
    ]
    # the slow problems first, so the last workers are not left with an s5 at the end
    order = {"s4-s5": 0, "emrcm": 1, "s3": 2, "s1-s2": 3}
    todo.sort(key=lambda t: order[stage(t)])
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} runs, {workers} workers, commit {commit}", flush=True)
    with out.open("a") as fh:
        pool = Pool(workers) if workers > 1 else None
        results = pool.imap_unordered(run_one, todo) if pool else map(run_one, todo)
        for i, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] {rec['config']}#{rec['repeat']} {rec['id']} cost {rec['cost']}"
                f" depth {rec['depth']} leaves {rec['leaves']} {rec['seconds']}s",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
