"""Recipes: forward-built plans with a known feasible cost (SPEC 8.2).

Generating a solved problem forward is easy; solving it backward is hard (the AlphaTensor idea).
A **recipe** is a mix tree built by random *legal* mixes from the reservoirs, so the composition
it produces is reachable under the chip limits by construction, and replaying it is a
feasibility witness with a known cost (an upper bound for that instance).

- `Draw(fluid_id)` — units taken straight from a reservoir (unlimited).
- `Blend(a, k, b, l)` — build `a`, take k units of it; build `b`, take l units; mix.
  A leftover of a sub-recipe is discarded. A recipe used twice in one tree is simply built
  twice: recipes never share a droplet, which is what makes them *naive* witnesses.

Used for (a) the `reachable_walk` composition sampler (SPEC 8.1), (b) synthetic instances whose
feasible cost is known, and (c) optional imitation warm-starts later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from meda.core.actions import Action, Output, Source
from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Composition, linf_distance, mix
from meda.core.fluids import Fluid
from meda.core.problem import Instance, Target
from meda.core.rules import IllegalActionError, replay
from meda.solvers.base import PlanBuilder


@dataclass(frozen=True, slots=True)
class Draw:
    """Units drawn from reservoir `fluid_id`."""

    fluid_id: int


@dataclass(frozen=True, slots=True)
class Blend:
    """Mix k units of recipe `a` with l units of recipe `b`."""

    a: Recipe
    k: int
    b: Recipe
    l: int


Recipe = Draw | Blend


def composition_of(recipe: Recipe, num_fluids: int) -> Composition:
    if isinstance(recipe, Draw):
        return Composition.pure(recipe.fluid_id, num_fluids)
    return mix(
        composition_of(recipe.a, num_fluids),
        recipe.k,
        composition_of(recipe.b, num_fluids),
        recipe.l,
    )


def volume_of(recipe: Recipe) -> int | None:
    """Volume the recipe produces; None for a `Draw` (a reservoir is unlimited)."""
    return None if isinstance(recipe, Draw) else recipe.k + recipe.l


def mix_count(recipe: Recipe) -> int:
    if isinstance(recipe, Draw):
        return 0
    return 1 + mix_count(recipe.a) + mix_count(recipe.b)


def recipe_to_dict(recipe: Recipe) -> dict[str, Any]:
    if isinstance(recipe, Draw):
        return {"draw": recipe.fluid_id}
    return {
        "a": recipe_to_dict(recipe.a),
        "k": recipe.k,
        "b": recipe_to_dict(recipe.b),
        "l": recipe.l,
    }


def recipe_from_dict(data: dict[str, Any]) -> Recipe:
    if "draw" in data:
        return Draw(int(data["draw"]))
    return Blend(
        recipe_from_dict(data["a"]), int(data["k"]), recipe_from_dict(data["b"]), int(data["l"])
    )


def build_recipe(builder: PlanBuilder, recipe: Recipe) -> Source:
    """Emit the actions that produce `recipe` and return the source holding its droplet.

    A `Draw` returns its reservoir (nothing is emitted). Leftovers of sub-recipes are
    discarded, so storage stays bounded.
    """
    if isinstance(recipe, Draw):
        return Source.reservoir(recipe.fluid_id)
    src_a = build_recipe(builder, recipe.a)
    src_b = build_recipe(builder, recipe.b)
    uid = builder.mix(src_a, recipe.k, src_b, recipe.l)
    for src in (src_a, src_b):
        if not src.is_reservoir:
            builder.discard_leftover(src.index)
    return Source.droplet(uid)


def witness_plan(instance: Instance, recipes: Sequence[Recipe]) -> tuple[Action, ...]:
    """Plan that delivers every target from its recipe, one droplet at a time.

    `recipes[t]` builds target t. Raises `IllegalActionError` if the plan does not fit the
    instance (for example its step limit), and `ValueError` if a recipe is too small for the
    target's droplet volume or does not hit the target.
    """
    builder = PlanBuilder.start(instance)
    for target, recipe in zip(instance.targets, recipes, strict=True):
        produced = composition_of(recipe, instance.num_fluids)
        if not linf_distance(produced, target.composition) <= target.tolerance:
            raise ValueError(
                f"instance '{instance.instance_id}': recipe for target {target.id} makes "
                f"{produced}, outside tolerance {target.tolerance} of {target.composition}"
            )
        for _ in range(target.demand):
            src = build_recipe(builder, recipe)
            if src.is_reservoir:
                builder.do(Output(src, target.id))
                continue
            if builder.volume(src.index) < target.droplet_volume:
                raise ValueError(
                    f"instance '{instance.instance_id}': recipe for target {target.id} makes "
                    f"{builder.volume(src.index)} units, below droplet_volume "
                    f"{target.droplet_volume}"
                )
            builder.do(Output(src, target.id))
            builder.discard_leftover(src.index)
    if not builder.state.is_success:
        raise ValueError(f"instance '{instance.instance_id}': witness plan left demand unmet")
    return builder.plan()


def support_size_of(comp: Composition) -> int:
    """How many fluids the composition actually contains."""
    return sum(1 for n in comp.num if n > 0)


def random_recipe(
    rng: np.random.Generator,
    num_fluids: int,
    chip: ChipLimits,
    num_mixes: int,
    support_size: int | None = None,
    max_tries: int = 64,
) -> Recipe:
    """Grow a pool of recipes with `num_mixes` random legal mixes and return the last one.

    Every mix respects the chip's volume and denominator limits, so the result is reachable.
    `support_size` is how many fluids the result may draw on; fluid 0 (a SAMPLE) is always
    one of them, as in `dirichlet_snapped`. The result may use fewer than `support_size` (the
    walk need not reach every allowed fluid); the caller decides whether that is acceptable.
    Raises if no legal mix is found within `max_tries`, rather than returning something
    smaller than asked for.
    """
    if num_mixes < 1:
        raise ValueError(f"random_recipe needs num_mixes >= 1, got {num_mixes}")
    allowed = list(range(num_fluids))
    if support_size is not None and support_size < num_fluids:
        others = rng.choice(range(1, num_fluids), size=support_size - 1, replace=False)
        allowed = [0, *(int(f) for f in others)]
    pool: list[tuple[Recipe, Composition, int | None]] = [
        (Draw(f), Composition.pure(f, num_fluids), None) for f in allowed
    ]
    last: Recipe | None = None
    for _ in range(num_mixes):
        for _try in range(max_tries):
            i, j = (int(x) for x in rng.choice(len(pool), size=2, replace=False))
            (ra, ca, va), (rb, cb, vb) = pool[i], pool[j]
            if ca == cb and not chip.allow_same_composition_merge:
                continue
            k = int(rng.integers(1, min(chip.max_mix_volume - 1, va or chip.max_mix_volume) + 1))
            top = min(chip.max_mix_volume - k, vb or chip.max_mix_volume)
            if top < 1:
                continue
            l = int(rng.integers(1, top + 1))
            if not chip.min_mix_volume <= k + l <= chip.max_mix_volume:
                continue
            if mix(ca, k, cb, l).den > chip.max_denominator:
                continue
            recipe = Blend(ra, k, rb, l)
            pool.append((recipe, mix(ca, k, cb, l), k + l))
            last = recipe
            break
        else:
            raise RuntimeError(
                f"random_recipe: no legal mix found in {max_tries} tries on chip "
                f"'{chip.name}' with {len(allowed)} fluids"
            )
    assert last is not None
    return last


def synthetic_instance(
    instance_id: str,
    fluids: Sequence[Fluid],
    chip: ChipLimits,
    rng: np.random.Generator,
    num_targets: int,
    num_mixes: int,
    demand: int = 1,
    droplet_volume: int = 1,
    support_size: int | None = None,
    step_limit_multiple: float = 2.0,
) -> tuple[Instance, tuple[Recipe, ...]]:
    """An instance whose targets are compositions a random legal plan actually produced.

    Returns the instance and the recipe per target. Replaying the recipes (`witness_plan`) is a
    feasible plan, so its cost is a known upper bound, stored in `metadata`.
    """
    num_fluids = len(fluids)
    recipes = tuple(
        random_recipe(rng, num_fluids, chip, num_mixes, support_size) for _ in range(num_targets)
    )
    targets = tuple(
        Target(i, i, composition_of(r, num_fluids), ZERO, droplet_volume, demand)
        for i, r in enumerate(recipes)
    )
    # Build once with a generous step limit to measure the witness, then fix the real limit.
    probe = Instance(instance_id, tuple(fluids), targets, chip, 10_000)
    actions = witness_plan(probe, recipes)
    step_limit = max(len(actions), int(step_limit_multiple * len(actions)))
    instance = Instance(
        instance_id,
        tuple(fluids),
        targets,
        chip,
        step_limit,
        {
            "generator": "synthetic",
            "witness_actions": len(actions),
            "witness_mixes": sum(
                mix_count(r) * t.demand for r, t in zip(recipes, targets, strict=True)
            ),
            "recipes": [recipe_to_dict(r) for r in recipes],
        },
    )
    _, totals, _ = replay(instance, actions)
    assert totals.outputs == sum(t.demand for t in targets)
    return instance, recipes


def check_witness(instance: Instance, recipes: Sequence[Recipe]) -> tuple[Action, ...]:
    """Replay the recipes against `instance` and return the plan, or raise."""
    try:
        return witness_plan(instance, recipes)
    except IllegalActionError as exc:
        raise ValueError(
            f"instance '{instance.instance_id}': witness plan is not legal: {exc}"
        ) from exc
