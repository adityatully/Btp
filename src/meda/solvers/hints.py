"""Exact per-target tables for the candidate hint features (SPEC 9.3; paper log E-16, E-18).

The network cannot infer from compositions given as floats which intermediate leads to a
target: 33/40 S at M = 9 is built from 1/8 S (40 = 8 x 5), a droplet far from the target by
distance (E-15, E-16). These tables give the env that fact exactly, so it can put it in the
candidate rows (rule 1: the env does the arithmetic, the network only scores).

For a target, with `aim` = `tree.aim(target)` (the composition the tree builds; the target
itself when exact), keyed by an exact composition `(numerators..., denominator)`:

- **recipe:** the recipes for one droplet of the target by the split tree (`tree.build`, for
  `aim`) and by the sharing pass (`share_plan`, which also uses dilute-up chains): every
  intermediate either makes -> the fewest mixes from it to the delivered droplet along that
  recipe (0 for the delivered composition itself).
- **chain:** every composition x from which at most `CHAIN_DEPTH` mixes, each of k units of
  the running droplet with l units of one pure fluid (k + l <= M), reach `aim` with enough
  volume -> {mixes: the smallest k of x the first mix takes}. Worked back from `aim`, so it
  also covers recipes the tree does not use (dilute-up chains). Depth 2 keeps it to a few
  thousand compositions; the recipe covers deeper structure.

Tables depend only on (aim, droplet volume, chip), so they are cached across instances.
Hints are information, not instructions: nothing here changes what is legal or its cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import gcd

from meda.core.actions import Action, Mix, Output
from meda.core.composition import Composition
from meda.core.costs import CostWeights
from meda.core.problem import Instance, Target
from meda.core.rules import IllegalActionError, apply
from meda.core.state import PrepState
from meda.solvers.base import PlanBuilder
from meda.solvers.baselines.share import share_plan
from meda.solvers.baselines.tree import aim, build
from meda.solvers.fallback import RELAXED_STEP_LIMIT

Key = tuple[int, ...]
CHAIN_DEPTH = 2
CACHE_LIMIT = 20_000  # tables; the cache is cleared when full (values never depend on it)


def key_of(comp: Composition) -> Key:
    return (*comp.num, comp.den)


def _reduced(num: list[int], den: int) -> Key:
    g = den
    for v in num:
        g = gcd(g, v)
    return (*(v // g for v in num), den // g)


@dataclass(frozen=True, slots=True)
class HintTables:
    recipe: dict[Key, int]
    chain: dict[Key, dict[int, int]]

    def recipe_left(self, key: Key) -> int | None:
        """Mixes from this composition to the target along the tree's recipe, or None."""
        return self.recipe.get(key)

    def chain_left(self, key: Key, volume: int) -> int | None:
        """Fewest pure-fluid mixes from a droplet of this composition and volume to the
        target, or None (beyond `CHAIN_DEPTH`, or not on a chain)."""
        found = self.chain.get(key)
        if not found:
            return None
        fits = [d for d, take in found.items() if take <= volume]
        return min(fits) if fits else None


class CandidateHints:
    """`tables(instance, target_id)` for the env's encoder, cached. Picklable."""

    def __init__(self, weights: CostWeights) -> None:
        self.weights = weights
        self._cache: dict[tuple[object, ...], HintTables] = {}

    def tables(self, instance: Instance, target_id: int) -> HintTables:
        target = instance.targets[target_id]
        kinds = tuple(f.kind for f in instance.fluids)
        cache_key: tuple[object, ...] = (
            replace(target, id=0, experiment_id=0),
            instance.chip,
            kinds,
        )
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        aimed = aim(target, instance.chip) or target.composition
        tables = HintTables(
            _recipe_table(instance, target, aimed, self.weights),
            _chain_table(aimed, target.droplet_volume, instance),
        )
        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()
        self._cache[cache_key] = tables
        return tables


def _recipe_table(
    instance: Instance, target: Target, aimed: Composition, weights: CostWeights
) -> dict[Key, int]:
    table = {key_of(aimed): 0}
    single = Instance(
        f"{instance.instance_id}#hint-t{target.id}",
        instance.fluids,
        (replace(target, id=0, experiment_id=0, demand=1),),
        instance.chip,
        RELAXED_STEP_LIMIT,
    )
    if aimed.den != 1:
        builder = PlanBuilder.start(single)
        try:
            root = build(builder, aimed, target.droplet_volume)
        except IllegalActionError:
            root = None
        if root is not None and not root.is_reservoir:
            _add_left(table, single, builder.actions, root.index)
    plan = share_plan(single, weights)
    outputs = [a for a in plan or () if isinstance(a, Output) and not a.src.is_reservoir]
    if plan is not None and outputs:
        _add_left(table, single, plan, outputs[0].src.index)
    return table


def _add_left(table: dict[Key, int], single: Instance, plan: Sequence[Action], root: int) -> None:
    """Record, for every droplet the plan makes, the fewest mixes from it to droplet `root`."""
    state = PrepState.initial(single)
    made: dict[int, Composition] = {}
    consumers: dict[int, list[int]] = {}
    for action in plan:
        if isinstance(action, Mix):
            uid = state.next_uid
            for src in (action.a, action.b):
                if not src.is_reservoir:
                    consumers.setdefault(src.index, []).append(uid)
            state, _, _ = apply(state, action, single)
            made[uid] = state.droplet(uid).composition
        else:
            state, _, _ = apply(state, action, single)
    if root not in made:
        return
    left: dict[int, int] = {root: 0}
    for uid in sorted(made, reverse=True):  # a droplet is consumed by later (larger) uids
        if uid in left:
            continue
        downstream = [left[c] for c in consumers.get(uid, ()) if c in left]
        if downstream:
            left[uid] = 1 + min(downstream)
    for uid, mixes in left.items():
        key = key_of(made[uid])
        table[key] = min(mixes, table.get(key, mixes))


def _chain_table(aimed: Composition, need: int, instance: Instance) -> dict[Key, dict[int, int]]:
    chip = instance.chip
    m, nf = chip.max_mix_volume, instance.num_fluids
    table: dict[Key, dict[int, int]] = {}
    frontier: list[tuple[Key, int]] = [(key_of(aimed), need)]  # (composition, units needed)
    for depth in range(1, CHAIN_DEPTH + 1):
        nxt: list[tuple[Key, int]] = []
        for y, units in frontier:
            num, den = y[:-1], y[-1]
            for total in range(max(units, chip.min_mix_volume), m + 1):
                for k in range(1, total):
                    pure = total - k
                    for f in range(nf):
                        x_num = [total * v for v in num]
                        x_num[f] -= pure * den
                        if x_num[f] < 0:
                            continue
                        x = _reduced(x_num, den * k)
                        if x[-1] > chip.max_denominator:
                            continue
                        known = table.setdefault(x, {})
                        if any(d <= depth and take <= k for d, take in known.items()):
                            continue
                        known[depth] = min(k, known.get(depth, k))
                        nxt.append((x, k))
        frontier = nxt
    return table
