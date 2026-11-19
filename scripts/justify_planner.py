"""J6 of the justification sweeps (D-98, `notes/justification_audit.md` Part 1 section 3): every
setting of the sharing pass, one at a time, on the planner alone from the empty chip.

The pass's code is unchanged: each setting is a module constant set for the run, or a function
of `share` / `tree` swapped for a variant written here, then put back. Two bases: `plain`
(`share_plan`, the `share` baseline and the search's plain finisher) and `mrcm` (with MRCM's
tree, greedy: the search's default finisher, D-65). Dev sets only: val s1-s5 (`_v2`) and
`val_emrcm_v1`, 322 problems; the test sets stay untouched.

**Curves (D-101)**, each number swept well past its value (base value in brackets): chain
depth 1-6 [3]; blend partners 0, 1, 2, 3, 4, 6, 8, 12, all [4]; lookahead k 1-6, 8, 10 [1 =
greedy, method-v3's finisher; 3 in the `look3` baselines]; split-tree extra steps 1-6, 8 [4];
the split tree's and MRCM's search budgets x 0.25, 0.5, 1, 4, 16; the recipes added one at a
time (tree, + direct reuse, + one mix, + chain, + MRCM tree). Ceilings, counted in every run:
candidates per delivery group (lookahead k at or above it is exhaustive) and idle droplets
(blend partners at or above it take every droplet). **Target orders** (`orders=1`): each
problem planned once per order: the 4 rules, then every permutation when there are at most 24
(the curve ends at the best order there is), else 32 random orders; the curve is the best of
the first n orders, and every subset of the rules is scored. **Other settings**, one at a
time: each recipe dropped, the chain's last-input rule, eviction {smallest, oldest, random},
greedy rule total cost, every deliveries-per-droplet count, tolerant aim finest, one greedy
over every open target.

Every run is capped at `cap` CPU seconds (default 1800) and records whether it hit the cap,
the cost, CPU seconds, and how often the split tree's and MRCM's search budgets ran out
(P15, P16).

**Reading, fixed in advance (D-101):** per curve, base and group, the mean of cost / the
per-problem best over the curve's values (problems every value finished) and the median time
over the base's. The **saturation point** is the smallest value from which no larger value is
more than 0.5% cheaper. The base value is justified if it is at or past the saturation point;
if it is below it, the curve gives the rest of the gain and its time. Other settings (D-98):
paired ratio to the base, 95% bootstrap CI; level = CI contains 1.

    OMP_NUM_THREADS=1 uv run python scripts/justify_planner.py workers=30 \\
        out=outputs/ablation/j6.jsonl
    OMP_NUM_THREADS=1 uv run python scripts/justify_planner.py orders=1 workers=30 \\
        out=outputs/ablation/j6_orders.jsonl
    uv run python scripts/justify_planner.py summary=outputs/ablation/j6.jsonl
"""

from __future__ import annotations

import itertools
import json
import math
import random
import signal
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.composition import ZERO, Composition, linf_distance
from meda.core.costs import CostWeights
from meda.core.problem import Instance, Target
from meda.core.rules import IllegalActionError, replay
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import mrcm, share, tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom_share as hs  # noqa: E402

WEIGHTS: CostWeights = hs.WEIGHTS
SETS = ("val_s1_v2", "val_s2_v2", "val_s3_v2", "val_s4_v2", "val_s5_v2", "val_emrcm_v1")
BASES = ("plain", "mrcm")
CAP = 1800.0

# --------------------------------------------------------------------------- variants

_ORIGINAL = {
    (share, "CHAIN_DEPTH"): share.CHAIN_DEPTH,
    (share, "BLEND_PARTNERS"): share.BLEND_PARTNERS,
    (share, "_chain"): share._chain,
    (share, "_candidates"): share._candidates,
    (share, "_options"): share._options,
    (share, "_run"): share._run,
    (share, "named_orders"): share.named_orders,
    (share, "aim"): share.aim,
    (share._Builder, "_idleness"): share._Builder._idleness,
    (tree, "aim"): tree.aim,
    (tree, "MAX_EXTRA_STEPS"): tree.MAX_EXTRA_STEPS,
    (tree, "SEARCH_BUDGET"): tree.SEARCH_BUDGET,
    (mrcm, "SEARCH_BUDGET"): mrcm.SEARCH_BUDGET,
}
COUNTS: dict[str, Any] = {"tree_budget": 0, "mrcm_budget": 0, "options": [], "pool": []}


def _recipe_kind(emit: Callable[..., Any]) -> str:
    """Which of the pass's recipes an emitter is, from the function that made it."""
    return {
        "_fixed": "direct",
        "_mixer": "one_mix",
        "_chained": "chain",
        "_tree": "tree",
        "_mrcm_tree": "mrcm",
    }[emit.__qualname__.split(".")[0]]


def chain_any_last(b: Any, aimed: Composition, need: int, depth: int) -> Any:
    """`share._chain` without its last-level rule (the chain's last input may be any droplet
    the search reaches, not only a pure fluid or a pool droplet). Otherwise the same code."""
    inst, chip, w = b.instance, b.instance.chip, b.weights
    fluid_w = [w.fluid_weight(f.kind) for f in inst.fluids]
    idle: dict[Composition, tuple[int, int]] = {}
    for uid, comp, vol in b.idle():
        if comp not in idle or vol > idle[comp][1]:
            idle[comp] = (uid, vol)
    partners = sorted(idle.items(), key=lambda kv: (linf_distance(kv[0], aimed), kv[1][0]))
    partners = partners[: share.BLEND_PARTNERS]
    memo: dict[tuple[Composition, int, int], Any] = {}

    def search(x: Composition, take: int, left: int) -> Any:
        if x.den == 1:
            f = x.num.index(1)
            return fluid_w[f] * take, ("res", f), ()
        hit = idle.get(x)
        if (
            hit is not None
            and hit[1] >= take
            and share._take_ok(hit[1], take, chip.min_droplet_volume)
        ):
            return ZERO, ("pool", hit[0]), ()
        if left == 0:
            return None
        key = (x, take, left)
        if key in memo:
            return memo[key]
        best = None
        for size in range(max(take, chip.min_mix_volume), chip.max_mix_volume + 1):
            for f, l, xp in share._dilute_steps(x, size, chip.max_denominator):
                sub = search(xp, size - l, left - 1)
                if sub is not None:
                    cost = sub[0] + fluid_w[f] * l + w.w_mix
                    if best is None or cost < best[0]:
                        best = (cost, sub[1], (*sub[2], (f, size - l, l)))
            for pc, (uid, pv) in partners:
                for l in range(1, min(size - 1, pv) + 1):
                    if not share._take_ok(pv, l, chip.min_droplet_volume):
                        continue
                    xp = share._unmix(x, size, pc, l)
                    if xp is None or xp == pc or xp.den > chip.max_denominator:
                        continue
                    sub = search(xp, size - l, left - 1)
                    if sub is None or sub[1] == ("pool", uid):
                        continue
                    cost = sub[0] + w.w_mix
                    if best is None or cost < best[0]:
                        best = (cost, sub[1], (*sub[2], (-1 - uid, size - l, l)))
        memo[key] = best
        return best

    return search(aimed, need, depth)


def candidates_all_counts(b: Any, target: Target, left: int) -> list[Any]:
    """`share._candidates` trying every count of deliveries per droplet (most down to 1), not
    only the most and 1, for the one-mix, chain and MRCM recipes."""
    chip = b.instance.chip
    v = target.droplet_volume
    most = max(1, min(left, chip.max_mix_volume // v))
    out: list[Any] = []
    for uid, comp, vol in b.idle():
        if vol >= v and share._close(comp, target):
            out.append((min(left, vol // v), share._fixed(share.Source.droplet(uid))))
    best: dict[int, Any] = {}
    for count in range(most, 0, -1):
        for draw, s, sa, k, sb, l in share._one_mix(b, target, count * v):
            served = min(left, s // v)
            if served not in best or draw < best[served][0]:
                best[served] = (draw, sa, k, sb, l)
    for served, (_, sa, k, sb, l) in best.items():
        out.append((served, share._mixer(sa, k, sb, l)))
    aimed = share.aim(target, chip)
    if aimed is not None and aimed.den > 1:
        for count in range(most, 0, -1):
            found = share._chain(b, aimed, count * v, share.CHAIN_DEPTH)
            if found is not None:
                out.append((count, share._chained(found[1], found[2])))
    if aimed is not None:
        count = share._deliveries_per_droplet(b, target, most)
        out.append((count, share._tree(aimed, count * v)))
    if b.mrcm is not None and aimed is not None and aimed.den > 1:
        for count in range(most, 0, -1):
            out.append((count, share._mrcm_tree(aimed, count * v)))
    return out


def options_total(b: Any, tid: int) -> list[Any]:
    """`share._options` ranked by total marginal cost, not cost per delivery."""
    out = _ORIGINAL[(share, "_options")](b, tid)
    out.sort(key=lambda o: (o[2].cost - b.cost, -o[1], len(o[2].actions)))
    return out


def run_global(b: Any, order: list[int], lookahead: int) -> Any:
    """One greedy over every open target: each delivery group is the cheapest per delivery
    among all open targets' options (ties in `order`), not target by target."""
    if lookahead:
        raise ValueError("run_global is greedy only")
    rank = {t: i for i, t in enumerate(order)}
    while True:
        open_ = [t for t in order if b.state.remaining_demand[t]]
        if not open_:
            break
        best = None
        for tid in open_:
            opts = share._options(b, tid)
            if opts and (best is None or (opts[0][0], rank[tid]) < (best[0], best[1])):
                best = (opts[0][0], rank[tid], opts[0][2])
        if best is None:
            raise IllegalActionError(f"instance '{b.instance.instance_id}': no way to go on")
        b = best[2]
    if not b.state.is_success:
        raise IllegalActionError(f"instance '{b.instance.instance_id}': demand left")
    return b


def aim_finest(target: Target, chip: Any) -> Composition | None:
    """`tree.aim` taking the finest reachable denominator within tolerance, not the coarsest."""
    comp, tol = target.composition, target.tolerance
    if tol.num == 0:
        return comp
    for den in reversed(
        tree._reachable_denominators(chip.min_mix_volume, chip.max_mix_volume, chip.max_denominator)
    ):
        found = tree.within_tolerance(comp, tol, den)
        if found is not None:
            return found
    return None


def order_list(inst: Instance, weights: CostWeights) -> list[tuple[tuple[str, ...], list[int]]]:
    """Every target order the order curve plans (D-101): the pass's rules first (distinct
    orders, with the rules that give each), then every other permutation of the targets when
    there are at most `ALL_ORDERS` (the curve then ends at the best of all orders), else up to
    `RANDOM_ORDERS` random ones; both seeded by the problem's id, so reruns match."""
    out = [(tuple(r), list(o)) for r, o in share.named_orders(inst, weights)]
    seen = {tuple(o) for _, o in out}
    n = len(inst.targets)
    seed = zlib.crc32(inst.instance_id.encode())
    if math.factorial(n) <= ALL_ORDERS:
        perms = [list(p) for p in itertools.permutations(range(n))]
        rng = np.random.default_rng([seed, 1])
        extra, kind, most = [perms[int(i)] for i in rng.permutation(len(perms))], "all", len(perms)
    else:
        rng = np.random.default_rng([seed, 0])
        extra = [[int(i) for i in rng.permutation(n)] for _ in range(4 * RANDOM_ORDERS)]
        kind, most = "random", len(out) + RANDOM_ORDERS
    for order in extra:
        if len(out) >= most:
            break
        if tuple(order) not in seen:
            seen.add(tuple(order))
            out.append(((kind,), order))
    return out


ALL_ORDERS = 24  # every permutation up to 4 targets (24 orders)
RANDOM_ORDERS = 32  # past that, 32 random orders (E-28 used 32)


def run_orders(task: dict[str, Any]) -> dict[str, Any]:
    """One problem planned once per target order (`order_list`), each order on its own, with
    one MRCM cache per problem as in `share_plan`: per order its cost and CPU seconds."""
    apply({})
    cold()
    inst: Instance = BenchmarkSet.read(task["set"]).by_id(task["id"])
    cache: dict[Any, list[Any]] | None = {} if task["base"] == "mrcm" else None
    signal.signal(signal.SIGPROF, _alarm)
    signal.setitimer(signal.ITIMER_PROF, float(task["cap"]))
    runs: list[dict[str, Any]] = []
    capped = False
    try:
        for rules, order in order_list(inst, WEIGHTS):
            t0 = time.process_time()
            try:
                start = share._Builder(inst, PrepState.initial(inst), [], WEIGHTS, mrcm=cache)
                cost: float | None = float(share._run(start, order, 0).cost)
            except IllegalActionError:
                cost = None
            runs.append(
                {
                    "rules": list(rules),
                    "order": order,
                    "cost": cost,
                    "seconds": round(time.process_time() - t0, 3),
                }
            )
    except _Cap:
        capped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    return {
        "mode": "orders",
        "base": task["base"],
        "set": task["set"],
        "id": task["id"],
        "commit": task["commit"],
        "targets": len(inst.targets),
        "capped": capped,
        "runs": runs,
    }


def _idle_smallest(self: Any, uid: int) -> tuple[Any, ...]:
    return (-self.state.droplet(uid).volume, uid)


def _idle_oldest(self: Any, uid: int) -> tuple[Any, ...]:
    return (-uid,)


def _idle_random(self: Any, uid: int) -> tuple[Any, ...]:
    return (random.Random(uid * 1_000_003 + 7).random(), uid)


def configs() -> dict[str, dict[str, Any]]:
    """Each setting as changes against the base: `set` (module constants), `swap`
    (functions), `drop` (a recipe), `lookahead`."""
    out: dict[str, dict[str, Any]] = {"base": {}}
    for d in CURVES["chain"]:
        if d != share.CHAIN_DEPTH:
            out[f"chain{d}"] = {"set": {(share, "CHAIN_DEPTH"): d}}
    for p in CURVES["blend"]:
        if p != share.BLEND_PARTNERS:
            out[f"blend{p}"] = {"set": {(share, "BLEND_PARTNERS"): p}}
    out["last_any"] = {"swap": {(share, "_chain"): chain_any_last}}
    for k in CURVES["look"]:
        if k > 1:
            out[f"look{k}"] = {"lookahead": k}
    for r in RECIPES:
        out[f"drop_{r}"] = {"drop": r}
    for n in range(1, len(RECIPES)):  # the recipes added one at a time, in RECIPES' order
        out[f"recipes{n}"] = {"keep": RECIPES[:n]}
    out["evict_smallest"] = {"swap": {(share._Builder, "_idleness"): _idle_smallest}}
    out["evict_oldest"] = {"swap": {(share._Builder, "_idleness"): _idle_oldest}}
    out["evict_random"] = {"swap": {(share._Builder, "_idleness"): _idle_random}}
    out["greedy_total"] = {"swap": {(share, "_options"): options_total}}
    out["counts_all"] = {"swap": {(share, "_candidates"): candidates_all_counts}}
    out["aim_finest"] = {"swap": {(share, "aim"): aim_finest, (tree, "aim"): aim_finest}}
    for e in CURVES["extra"]:
        if e != tree.MAX_EXTRA_STEPS:
            out[f"extra{e}"] = {"set": {(tree, "MAX_EXTRA_STEPS"): e}}
    out["global_greedy"] = {"swap": {(share, "_run"): run_global}}
    for f in CURVES["budget"]:
        if f != 1:
            out[f"tree_budget{f:g}x"] = {"set": {(tree, "SEARCH_BUDGET"): int(f * TREE_BUDGET)}}
            out[f"mrcm_budget{f:g}x"] = {"set": {(mrcm, "SEARCH_BUDGET"): int(f * MRCM_BUDGET)}}
    return out


# The curves (D-101): each number swept well past its value, the base value included
CURVES: dict[str, tuple[Any, ...]] = {
    "chain": (1, 2, 3, 4, 5, 6),  # CHAIN_DEPTH = 3
    "blend": (0, 1, 2, 3, 4, 6, 8, 12, 10_000),  # BLEND_PARTNERS = 4; 10,000 = every droplet
    "look": (1, 2, 3, 4, 5, 6, 8, 10),  # lookahead k; 1 = the greedy pass (method-v3's finisher)
    "extra": (1, 2, 3, 4, 5, 6, 8),  # tree.MAX_EXTRA_STEPS = 4
    "budget": (0.25, 0.5, 1, 4, 16),  # tree.SEARCH_BUDGET, mrcm.SEARCH_BUDGET x this
}
TREE_BUDGET, MRCM_BUDGET = tree.SEARCH_BUDGET, mrcm.SEARCH_BUDGET
# the recipes in the order they joined the pass (E-5, E-17, D-61): the split tree is the
# construction every target has, then reuse, one mix, the chain, and MRCM's tree
RECIPES = ("tree", "direct", "one_mix", "chain", "mrcm")


def family(name: str) -> list[tuple[Any, str]]:
    """(value, config) along one curve, the base value as `base`."""
    base_value = {
        "chain": share.CHAIN_DEPTH,
        "blend": share.BLEND_PARTNERS,
        "look": 1,
        "extra": tree.MAX_EXTRA_STEPS,
    }
    if name in base_value:
        prefix = {"chain": "chain", "blend": "blend", "look": "look", "extra": "extra"}[name]
        return [(v, "base" if v == base_value[name] else f"{prefix}{v}") for v in CURVES[name]]
    if name in ("tree_budget", "mrcm_budget"):
        return [(f, "base" if f == 1 else f"{name}{f:g}x") for f in CURVES["budget"]]
    if name == "recipes":
        return [(n, f"recipes{n}") for n in range(1, len(RECIPES))] + [(len(RECIPES), "base")]
    raise ValueError(name)


FAMILIES = ("chain", "blend", "look", "extra", "recipes", "tree_budget", "mrcm_budget")


def _count_tree_budget(self: Any, comp: Any, need: int, cap: int) -> Any:
    found = _TREE_FIND(self, comp, need, cap)
    if found is None and self.left == 0:
        COUNTS["tree_budget"] += 1
    return found


def _count_mrcm_budget(self: Any, n: int = 1) -> None:
    if self.budget - n < 0:
        COUNTS["mrcm_budget"] += 1
    _MRCM_TICK(self, n)


def _count_options(b: Any, tid: int) -> list[Any]:
    """`share._options`, counting the candidates of each delivery group and the idle droplets
    then on chip: lookahead k at or above the first is exhaustive, blend partners at or above
    the second take every droplet (the curves' ceilings, D-101)."""
    out = _OPTIONS(b, tid)
    COUNTS["options"].append(len(out))
    COUNTS["pool"].append(len(b.idle()))
    return out


_OPTIONS = share._options
share._options = _count_options  # type: ignore[assignment]
_ORIGINAL[(share, "_options")] = _count_options
_TREE_FIND = tree._Search.find
_MRCM_TICK = mrcm._Search.tick
tree._Search.find = _count_tree_budget  # type: ignore[method-assign]
mrcm._Search.tick = _count_mrcm_budget  # type: ignore[method-assign]


def cold() -> None:
    """Empty the pass's memo caches (pure functions, so only time changes): every run starts
    cold, whatever ran before it in the same worker; otherwise later runs look faster."""
    for fn in (share._dilute_steps, share._unmix, tree._reachable_denominators):
        fn.cache_clear()


def apply(cfg: dict[str, Any]) -> None:
    """Put every patched name back, then set this config's changes."""
    for (owner, name), value in _ORIGINAL.items():
        setattr(owner, name, value)
    for (owner, name), value in cfg.get("set", {}).items():
        setattr(owner, name, value)
    for (owner, name), value in cfg.get("swap", {}).items():
        setattr(owner, name, value)
    if "drop" in cfg or "keep" in cfg:
        inner = share._candidates  # the swapped one if any (none of the drops swap it)
        keep = set(cfg.get("keep", RECIPES)) - {cfg.get("drop")}

        def dropped(b: Any, target: Target, left: int) -> list[Any]:
            return [c for c in inner(b, target, left) if _recipe_kind(c[1]) in keep]

        share._candidates = dropped


# --------------------------------------------------------------------------- runs


class _Cap(Exception):
    pass


def _alarm(signum: int, frame: Any) -> None:
    raise _Cap


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    cfg = configs()[task["config"]]
    apply(cfg)
    cold()
    COUNTS.update(tree_budget=0, mrcm_budget=0, options=[], pool=[])
    inst: Instance = BenchmarkSet.read(task["set"]).by_id(task["id"])
    signal.signal(signal.SIGPROF, _alarm)
    signal.setitimer(signal.ITIMER_PROF, float(task["cap"]))
    t0 = time.process_time()
    capped, plan = False, None
    try:
        plan = share.share_plan(
            inst, WEIGHTS, lookahead=int(cfg.get("lookahead", 0)), mrcm=task["base"] == "mrcm"
        )
    except _Cap:
        capped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        apply({})
    secs = time.process_time() - t0
    cost: float | None = None
    if plan is not None:
        _, total, _ = replay(inst, plan)  # rule 7: the plan is legal and this is its cost
        cost = float(WEIGHTS.scalar(total, inst.fluids))
    return {
        "config": task["config"],
        "base": task["base"],
        "set": task["set"],
        "id": task["id"],
        "commit": task["commit"],
        "cost": cost,
        "seconds": round(secs, 3),
        "capped": capped,
        "tree_budget": COUNTS["tree_budget"],
        "mrcm_budget": COUNTS["mrcm_budget"],
        "options": _spread(COUNTS["options"]),  # candidates per delivery group
        "pool": _spread(COUNTS["pool"]),  # idle droplets per delivery group
    }


def _spread(xs: list[int]) -> dict[str, int] | None:
    if not xs:
        return None
    return {"median": int(np.median(xs)), "p90": int(np.percentile(xs, 90)), "max": max(xs)}


def group(set_name: str) -> str:
    if set_name.startswith("val_emrcm"):
        return "emrcm"
    if set_name.startswith(("val_s1", "val_s2")):
        return "s1-s2"
    return "s3" if set_name.startswith("val_s3") else "s4-s5"


def boot_ci(xs: list[float], seed: int = 0, n: int = 10_000) -> tuple[float, float]:
    a = np.asarray(xs)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


EPS = 0.005  # D-101: a curve has saturated where no larger value is > 0.5% cheaper
GROUPS = ("s1-s2", "s3", "s4-s5", "emrcm", "all")


def saturation(points: list[tuple[Any, float]]) -> Any:
    """The smallest value v from which no larger value is more than EPS cheaper than v:
    `points` = (value, mean ratio to the per-problem best), in curve order. A larger value
    that is dearer (the greedy pass is not monotone in its settings) does not stop it."""
    for i, (v, r) in enumerate(points):
        if all(r2 >= r - EPS for _, r2 in points[i + 1 :]):
            return v
    return None


def curves(by: dict[tuple[str, str], dict[str, dict]]) -> list[str]:
    """Each number's curve (D-101): per value, the mean of cost / the per-problem best over
    the curve's values (on the problems every value finished), the median time over the
    base's, and the runs capped; then the saturation point."""
    lines = [
        "## Curves (D-101): mean cost / best of the curve; (median time / base's); capped",
        "",
        f"Saturation point = the smallest value from which no larger value is more than "
        f"{EPS:.1%} cheaper. Ceilings: candidates and idle droplets per delivery group in the "
        "base runs (lookahead k and blend partners at or above them are exhaustive).",
        "",
    ]
    for fam in FAMILIES:
        for base in BASES:
            pts = family(fam)
            runs = [by.get((cfg, base), {}) for _, cfg in pts]
            if not all(runs):
                continue
            for g in GROUPS:
                ids = [
                    i
                    for i, r in runs[0].items()
                    if (g == "all" or r["group"] == g)
                    and all(i in rs and rs[i]["cost"] is not None for rs in runs)
                ]
                if not ids:
                    continue
                best = {i: min(rs[i]["cost"] for rs in runs) for i in ids}
                ref = by[("base", base)]
                cells, points = [], []
                for (v, _), rs in zip(pts, runs, strict=True):
                    r = float(np.mean([rs[i]["cost"] / best[i] for i in ids]))
                    t = [
                        rs[i]["seconds"] / ref[i]["seconds"]
                        for i in ids
                        if ref[i]["seconds"] > 0.01
                    ]
                    capped = sum(
                        1 for x in rs.values() if (g == "all" or x["group"] == g) and x["capped"]
                    )
                    mark = "*" if dict(pts)[v] == "base" else ""
                    cells.append(
                        f"{v if v != 10_000 else 'all'}{mark}: {r:.4f} ({np.median(t):.2f}x)"
                        + (f" {capped}c" if capped else "")
                    )
                    points.append((v, r))
                sat = saturation(points)
                lines.append(
                    f"- **{fam}**, {base}, {g} (n {len(ids)}): "
                    + "; ".join(cells)
                    + f" -> saturates at **{sat if sat != 10_000 else 'all'}**"
                )
    ceil = defaultdict(list)
    for (cfg, base), rs in by.items():
        if cfg == "base":
            for r in rs.values():
                if r.get("options"):
                    ceil[(base, r["group"])].append((r["options"]["max"], r["pool"]["max"]))
    lines += ["", "Ceilings in the base runs (max per problem; median over problems / max):"]
    for (base, g), xs in sorted(ceil.items()):
        o, q = [x[0] for x in xs], [x[1] for x in xs]
        lines.append(
            f"- {base}, {g}: candidates per group {int(np.median(o))} / {max(o)};"
            f" idle droplets {int(np.median(q))} / {max(q)}"
        )
    return lines


def order_curve(recs: list[dict[str, Any]]) -> list[str]:
    """The order curve (D-101): best of the first n orders (the 4 rules first, then the
    extra orders in their seeded order) over the best of every order tried, per problem; for
    problems with at most 4 targets every order is tried, so that is the true best order."""
    lines = [
        "## Target orders (D-101): mean cost of the best of n orders / best of all orders tried",
        "",
    ]
    for base in BASES:
        for g in GROUPS:
            for scope in ("all orders tried (<= 4 targets)", "rules + 32 random (> 4 targets)"):
                full = scope.startswith("all")
                rs = [
                    r
                    for r in recs
                    if r["base"] == base
                    and (g == "all" or group(r["set"]) == g)
                    and not r["capped"]
                    and (r["targets"] <= 4) == full
                    and r["targets"] > 1
                    and any(x["cost"] is not None for x in r["runs"])
                ]
                if not rs:
                    continue
                nmax = max(len(r["runs"]) for r in rs)
                steps = sorted({1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 28, 36, nmax})
                cells = []
                for n in steps:
                    ratios, times = [], []
                    for r in rs:
                        costs = [x["cost"] for x in r["runs"] if x["cost"] is not None]
                        first = [x["cost"] for x in r["runs"][:n] if x["cost"] is not None]
                        if first:
                            ratios.append(min(first) / min(costs))
                            times.append(sum(x["seconds"] for x in r["runs"][:n]))
                    if ratios and n <= nmax:
                        cells.append(f"{n}: {np.mean(ratios):.4f}")
                rules = [
                    len([x for x in r["runs"] if x["rules"][0] not in ("all", "random")])
                    for r in rs
                ]
                lines.append(
                    f"- {base}, {g}, {scope} (n {len(rs)}; distinct rule orders median "
                    f"{int(np.median(rules))}): " + "; ".join(cells)
                )
                # the pass itself: the best of all 4 rules' orders; time per order; caps
                pass_ratio = [
                    min(
                        x["cost"]
                        for x in r["runs"]
                        if x["cost"] is not None and x["rules"][0] not in ("all", "random")
                    )
                    / min(x["cost"] for x in r["runs"] if x["cost"] is not None)
                    for r in rs
                    if any(
                        x["cost"] is not None and x["rules"][0] not in ("all", "random")
                        for x in r["runs"]
                    )
                ]
                per_order = [x["seconds"] for r in rs for x in r["runs"]]
                capped = sum(
                    r["capped"]
                    for r in recs
                    if r["base"] == base
                    and (g == "all" or group(r["set"]) == g)
                    and (r["targets"] <= 4) == full
                    and r["targets"] > 1
                )
                lines.append(
                    f"  - the pass (all 4 rules): {np.mean(pass_ratio):.4f}, worst "
                    f"{max(pass_ratio):.4f}, >0.5% dearer on "
                    f"{sum(x > 1.005 for x in pass_ratio)}/{len(pass_ratio)}; seconds per "
                    f"order median {np.median(per_order):.2f}, p90 "
                    f"{np.percentile(per_order, 90):.2f}; capped (left out) {capped}"
                )
                # every subset of the rules, by size: best subset and mean over subsets
                rule_names = ("given", "dear", "cheap", "near")
                for k in (1, 2, 3):
                    scores = {}
                    for sub in itertools.combinations(rule_names, k):
                        ratios = []
                        for r in rs:
                            costs = [x["cost"] for x in r["runs"] if x["cost"] is not None]
                            picked = [
                                x["cost"]
                                for x in r["runs"]
                                if x["cost"] is not None and set(sub) & set(x["rules"])
                            ]
                            if picked:
                                ratios.append(min(picked) / min(costs))
                        if ratios:
                            scores[sub] = float(np.mean(ratios))
                    if scores:
                        top = min(scores, key=scores.get)
                        lines.append(
                            f"  - {k} rule(s): best {'+'.join(top)} {scores[top]:.4f}, mean over"
                            f" subsets {np.mean(list(scores.values())):.4f}"
                        )
    return lines


def summarise(paths: list[Path]) -> str:
    by: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    orders: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("mode") == "orders":
                orders.append(r)
                continue
            by[(r["config"], r["base"])][r["id"]] = r | {"group": group(r["set"])}
    if orders and not by:
        return "\n".join(order_curve(orders))
    lines = [
        "| setting | base | group | n | mean cost ratio [95% CI] | worse / better | verdict |"
        " median time ratio | capped | no plan |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    budget = defaultdict(lambda: [0, 0, 0])
    for (cfg, base), rs in sorted(by.items()):
        ref = by.get(("base", base), {})
        for r in rs.values():
            if cfg == "base":
                b = budget[(base, r["group"])]
                b[0] += 1
                b[1] += r["tree_budget"] > 0
                b[2] += r["mrcm_budget"] > 0
        if cfg == "base":
            continue
        for g in ("s1-s2", "s3", "s4-s5", "emrcm", "all"):
            mine = {i: r for i, r in rs.items() if g == "all" or r["group"] == g}
            ids = [i for i in mine if i in ref and ref[i]["cost"] and mine[i]["cost"] is not None]
            if not ids:
                continue
            ratio = [mine[i]["cost"] / ref[i]["cost"] for i in ids]
            lo, hi = boot_ci(ratio)
            mean = float(np.mean(ratio))
            verdict = "level" if lo <= 1 <= hi else ("worse" if mean > 1 else "better")
            times = [
                mine[i]["seconds"] / ref[i]["seconds"] for i in ids if ref[i]["seconds"] > 0.01
            ]
            t = f"{np.median(times):.2f}" if times else "-"
            lines.append(
                f"| {cfg} | {base} | {g} | {len(ids)} | {mean:.4f} [{lo:.4f}, {hi:.4f}] |"
                f" {sum(x > 1 + 1e-9 for x in ratio)} / {sum(x < 1 - 1e-9 for x in ratio)} |"
                f" {verdict} | {t} | {sum(r['capped'] for r in mine.values())} |"
                f" {sum(r['cost'] is None and not r['capped'] for r in mine.values())} |"
            )
    lines += [
        "",
        "Search budgets running out in the base runs (P15, P16): problems with at"
        " least one tree / MRCM budget hit, of n",
        "",
    ]
    for (base, g), (n, t, m) in sorted(budget.items()):
        lines.append(f"- {base}, {g}: tree {t} / {n}, MRCM {m} / {n}")
    lines += ["", *curves(by)]
    if orders:
        lines += ["", *order_curve(orders)]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise([Path(p) for p in args["summary"].split(",")]))
        return
    if args.get("orders") == "1":
        run_order_curve(args)
        return
    names = list(configs()) if "configs" not in args else ["base", *args["configs"].split(",")]
    for n in names:
        if n not in configs():
            raise ValueError(f"unknown setting '{n}' (known: {sorted(configs())})")
    bases = args.get("bases", ",".join(BASES)).split(",")
    sets = args.get("sets", ",".join(SETS)).split(",")
    limit = int(args["limit"]) if "limit" in args else None
    out = Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["base"], r["id"]))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    problems = []
    for s in sets:
        ids = [i.instance_id for i in BenchmarkSet.read(s).instances]
        problems += [(s, i) for i in (ids[:limit] if limit else ids)]
    todo = [
        {
            "config": c,
            "base": b,
            "set": s,
            "id": i,
            "commit": commit,
            "cap": float(args.get("cap", CAP)),
        }
        for c in dict.fromkeys(names)
        for b in bases
        for s, i in problems
        if (c, b, i) not in done and not (c == "drop_mrcm" and b == "plain")
    ]
    order = {"s4-s5": 0, "emrcm": 1, "s3": 2, "s1-s2": 3}
    todo.sort(key=lambda t: order[group(t["set"])])
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} runs, {workers} workers, commit {commit}", flush=True)
    with out.open("a") as fh:
        pool = Pool(workers, maxtasksperchild=50) if workers > 1 else None
        results = pool.imap_unordered(run_one, todo) if pool else map(run_one, todo)
        for i, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] {rec['config']}/{rec['base']} {rec['id']} cost {rec['cost']}"
                f" {rec['seconds']}s{' capped' if rec['capped'] else ''}",
                flush=True,
            )


def run_order_curve(args: dict[str, str]) -> None:
    """`orders=1`: every problem with at least 2 targets, planned once per order."""
    bases = args.get("bases", ",".join(BASES)).split(",")
    sets = args.get("sets", ",".join(SETS)).split(",")
    limit = int(args["limit"]) if "limit" in args else None
    out = Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(json.loads(x)["base"], json.loads(x)["id"]) for x in out.read_text().splitlines()}
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    todo = []
    for s in sets:
        insts = [i for i in BenchmarkSet.read(s).instances if len(i.targets) > 1]
        for inst in insts[:limit] if limit else insts:
            for b in bases:
                if (b, inst.instance_id) not in done:
                    todo.append(
                        {
                            "base": b,
                            "set": s,
                            "id": inst.instance_id,
                            "commit": commit,
                            "cap": float(args.get("cap", CAP)),
                        }
                    )
    todo.sort(key=lambda t: {"s4-s5": 0, "emrcm": 1, "s3": 2, "s1-s2": 3}[group(t["set"])])
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problems x bases, {workers} workers, commit {commit}", flush=True)
    with out.open("a") as fh:
        pool = Pool(workers, maxtasksperchild=20) if workers > 1 else None
        results = pool.imap_unordered(run_orders, todo) if pool else map(run_orders, todo)
        for i, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(todo)}] {rec['base']} {rec['id']}: {len(rec['runs'])} orders"
                f"{' capped' if rec['capped'] else ''}",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
