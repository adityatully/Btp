"""The start-state distribution p(s0): `sample_instance(cfg, rng)` (SPEC 8.1).

Every sampled instance is **solvable by construction**: the sampler builds a feasibility witness
(a legal plan that meets all demand) before returning, and records its cost as a known upper
bound in `metadata`. A config that cannot be witnessed raises an error naming the instance, so
training never draws an instance nobody can solve.

Two composition samplers (SPEC 8.1):
- `dirichlet_snapped` — sample the simplex, then snap to a composition whose denominator divides
  `accuracy_denominator`. Witnessed by the naive construction, so the accuracy denominator must
  factorise into mix volumes the chip allows.
- `reachable_walk` — compositions a random legal plan actually produced (`generators.synthetic`).
  Reachable by construction, and the walk itself is the witness.

`dilution_grid` is a deterministic helper (not a p(s0)): every single-target dilution n/D of one
sample with one buffer, the "sweep all targets at an accuracy level" set of the reporting
conventions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from meda.core.actions import Action
from meda.core.chip import ChipLimits
from meda.core.composition import ZERO, Composition, Rational
from meda.core.costs import CostWeights
from meda.core.fluids import Fluid, FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.generators.synthetic import (
    Recipe,
    composition_of,
    random_recipe,
    recipe_to_dict,
    support_size_of,
    witness_plan,
)
from meda.solvers.baselines.naive import naive_plan

COMPOSITION_SAMPLERS = ("dirichlet_snapped", "reachable_walk", "fixed_list")
GENERATOR_VERSION = 1


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """One generator stage (`configs/generator/*.yaml`), validated on construction."""

    name: str
    samples: tuple[int, int]
    reagents: tuple[int, int]
    buffers: tuple[int, int]
    num_experiments: tuple[int, int]
    targets_per_experiment: tuple[int, int]
    composition_sampler: str
    accuracy_denominator: int
    support_size: int
    demand: tuple[int, int]
    droplet_volume: tuple[int, int]  # drawn per target
    tolerance: Rational
    step_limit: int | None  # None: size it from the witness
    step_limit_multiple: float
    walk_mixes: tuple[int, int]
    fixed_compositions: tuple[tuple[int, ...], ...]
    # Chip inputs drawn per instance (SPEC 1.3); None keeps the chip config's value.
    max_mix_volume: tuple[int, int] | None = None
    storage_capacity: tuple[int, int] | None = None

    @property
    def min_fluids(self) -> int:
        return self.samples[0] + self.reagents[0] + self.buffers[0]

    @property
    def max_fluids(self) -> int:
        return self.samples[1] + self.reagents[1] + self.buffers[1]

    def __post_init__(self) -> None:
        def fail(msg: str) -> None:
            raise ValueError(f"generator '{self.name}': {msg}")

        if self.composition_sampler not in COMPOSITION_SAMPLERS:
            fail(f"composition_sampler must be one of {COMPOSITION_SAMPLERS}")
        for field, value in (
            ("samples", self.samples),
            ("reagents", self.reagents),
            ("buffers", self.buffers),
        ):
            if value[0] < 0 or value[1] < value[0]:
                fail(f"{field} must be an increasing range of counts, got {value}")
        if self.min_fluids < 2:
            fail(f"needs at least 2 fluids, got {self.min_fluids}")
        if self.samples[0] < 1:
            fail("needs at least one SAMPLE fluid")
        if not 2 <= self.support_size <= self.min_fluids:
            fail(f"support_size {self.support_size} outside [2, {self.min_fluids}]")
        if self.accuracy_denominator < 2:
            fail(f"accuracy_denominator must be >= 2, got {self.accuracy_denominator}")
        for field, value in (
            ("num_experiments", self.num_experiments),
            ("targets_per_experiment", self.targets_per_experiment),
            ("demand", self.demand),
            ("walk_mixes", self.walk_mixes),
        ):
            if value[0] < 1 or value[1] < value[0]:
                fail(f"{field} must be an increasing range of positive ints, got {value}")
        for field, span in (
            ("droplet_volume", self.droplet_volume),
            ("max_mix_volume", self.max_mix_volume),
            ("storage_capacity", self.storage_capacity),
        ):
            if span is not None and (span[0] < 1 or span[1] < span[0]):
                fail(f"{field} must be an increasing range of positive ints, got {span}")
        if self.max_mix_volume is not None and self.max_mix_volume[0] < 2:
            fail(f"max_mix_volume must be >= 2 (k, l >= 1), got {self.max_mix_volume}")
        if self.tolerance < 0:
            fail(f"tolerance must be >= 0, got {self.tolerance}")
        if self.composition_sampler == "fixed_list" and not self.fixed_compositions:
            fail("fixed_list needs a non-empty fixed_compositions list")

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> GeneratorConfig:
        known = {
            "name",
            "num_fluids",
            "num_experiments",
            "targets_per_experiment",
            "composition_sampler",
            "accuracy_denominator",
            "support_size",
            "demand",
            "droplet_volume",
            "tolerance",
            "step_limit",
            "step_limit_multiple",
            "walk_mixes",
            "fixed_compositions",
            "max_mix_volume",
            "storage_capacity",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"generator config has unknown keys {sorted(unknown)}")
        counts = data["num_fluids"]
        step_limit = data.get("step_limit")
        return GeneratorConfig(
            name=str(data["name"]),
            samples=_as_range(counts.get("samples", 1)),
            reagents=_as_range(counts.get("reagents", 0)),
            buffers=_as_range(counts.get("buffers", 1)),
            num_experiments=_as_range(data["num_experiments"]),
            targets_per_experiment=_as_range(data["targets_per_experiment"]),
            composition_sampler=str(data["composition_sampler"]),
            accuracy_denominator=int(data["accuracy_denominator"]),
            support_size=int(data["support_size"]),
            demand=_as_range(data["demand"]),
            droplet_volume=_as_range(data["droplet_volume"]),
            tolerance=Rational.parse(data["tolerance"]),
            step_limit=None if step_limit is None else int(step_limit),
            step_limit_multiple=float(data.get("step_limit_multiple", 2.0)),
            walk_mixes=_as_range(data.get("walk_mixes", [1, 3])),
            fixed_compositions=tuple(
                tuple(int(x) for x in parts) for parts in data.get("fixed_compositions", ())
            ),
            max_mix_volume=_optional_range(data.get("max_mix_volume")),
            storage_capacity=_optional_range(data.get("storage_capacity")),
        )

    def config_hash(self) -> str:
        """Stable hash of the stage definition, recorded in every instance's metadata."""
        payload = {
            "version": GENERATOR_VERSION,
            "name": self.name,
            "fluids": [list(self.samples), list(self.reagents), list(self.buffers)],
            "experiments": list(self.num_experiments),
            "targets": list(self.targets_per_experiment),
            "sampler": self.composition_sampler,
            "accuracy": self.accuracy_denominator,
            "support": self.support_size,
            "demand": list(self.demand),
            "droplet_volume": _hashable_span(self.droplet_volume),
            "tolerance": str(self.tolerance),
            "step_limit": self.step_limit,
            "step_limit_multiple": self.step_limit_multiple,
            "walk_mixes": list(self.walk_mixes),
            "fixed": [list(c) for c in self.fixed_compositions],
        }
        # Keys added after generator version 1, written only when set, so the hash of every
        # config that does not use them is unchanged (it names instances in frozen sets).
        for key, span in (
            ("max_mix_volume", self.max_mix_volume),
            ("storage_capacity", self.storage_capacity),
        ):
            if span is not None:
                payload[key] = list(span)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _as_range(value: Any) -> tuple[int, int]:
    """Accepts `3` (a fixed count) or `[low, high]`."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError(f"expected a [low, high] range, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _optional_range(value: Any) -> tuple[int, int] | None:
    return None if value is None else _as_range(value)


def _hashable_span(span: tuple[int, int]) -> int | list[int]:
    """A fixed value as itself (the version-1 form), a real range as [low, high]."""
    return span[0] if span[0] == span[1] else list(span)


def _draw(rng: np.random.Generator, span: tuple[int, int]) -> int:
    return int(rng.integers(span[0], span[1] + 1))


def _draw_if_range(rng: np.random.Generator, span: tuple[int, int]) -> int:
    """Like `_draw`, but a fixed value consumes no randomness, so adding a range option leaves
    every config without one drawing exactly as before."""
    return span[0] if span[0] == span[1] else _draw(rng, span)


def sample_chip(cfg: GeneratorConfig, chip: ChipLimits, rng: np.random.Generator) -> ChipLimits:
    """The instance's chip: the config's chip with M and storage capacity drawn from the
    generator's ranges where it sets them (SPEC 1.3: chip limits are problem inputs). The
    name records what was drawn; it is a label, not part of the problem (`problem_hash`)."""
    if cfg.max_mix_volume is None and cfg.storage_capacity is None:
        return chip
    m = chip.max_mix_volume if cfg.max_mix_volume is None else _draw(rng, cfg.max_mix_volume)
    cap = (
        chip.storage_capacity if cfg.storage_capacity is None else _draw(rng, cfg.storage_capacity)
    )
    if m < chip.min_mix_volume:
        raise ValueError(
            f"generator '{cfg.name}': drew max_mix_volume {m} < chip '{chip.name}' "
            f"min_mix_volume {chip.min_mix_volume}"
        )
    return replace(chip, name=f"{chip.name}-M{m}-S{cap}", max_mix_volume=m, storage_capacity=cap)


def make_fluids(cfg: GeneratorConfig, rng: np.random.Generator) -> tuple[Fluid, ...]:
    """Draw the fluid counts and name them in a fixed order: samples, reagents, buffers.

    Fluid ids are 0..F-1, so fluid 0 is always a SAMPLE (the scarcest kind).
    """
    counts = {
        FluidKind.SAMPLE: _draw(rng, cfg.samples),
        FluidKind.REAGENT: _draw(rng, cfg.reagents),
        FluidKind.BUFFER: _draw(rng, cfg.buffers),
    }
    letters = {FluidKind.SAMPLE: "S", FluidKind.REAGENT: "R", FluidKind.BUFFER: "B"}
    fluids: list[Fluid] = []
    for kind, count in counts.items():
        for i in range(count):
            name = letters[kind] + (str(i + 1) if count > 1 else "")
            fluids.append(Fluid(len(fluids), name, kind))
    return tuple(fluids)


def snap_to_denominator(
    rng: np.random.Generator, cfg: GeneratorConfig, num_fluids: int
) -> Composition:
    """Sample the simplex (Dirichlet) over `support_size` fluids, then snap to
    `accuracy_denominator`.

    Fluid 0 (a SAMPLE) is always in the support, so a dilution target really is sample+buffer.
    Every share is at least one unit, so the support size is exact.
    """
    den = cfg.accuracy_denominator
    support = cfg.support_size
    if support > den:
        raise ValueError(
            f"generator '{cfg.name}': support_size {support} exceeds accuracy_denominator {den}"
        )
    others = [f for f in range(1, num_fluids)]
    chosen = [0, *(int(f) for f in rng.choice(others, size=support - 1, replace=False))]
    weights = rng.dirichlet(np.ones(support))
    # Largest-remainder rounding to integers that sum to `den`, each at least 1.
    raw = weights * (den - support)
    floor = np.floor(raw).astype(np.int64)
    short = (den - support) - int(floor.sum())
    order = np.argsort(-(raw - floor))
    for i in range(short):
        floor[order[i % support]] += 1
    parts = [0] * num_fluids
    for fluid, units in zip(chosen, floor + 1, strict=True):
        parts[fluid] = int(units)
    return Composition.from_ints(tuple(parts), den)


def _sample_compositions(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    chip: ChipLimits,
    count: int,
    num_fluids: int,
) -> tuple[tuple[Composition, ...], tuple[Recipe, ...] | None]:
    """Target compositions, and the recipes that make them when the sampler provides any."""
    if cfg.composition_sampler == "reachable_walk":
        recipes = tuple(_walk_with_support(rng, cfg, chip, num_fluids) for _ in range(count))
        return tuple(composition_of(r, num_fluids) for r in recipes), recipes
    if cfg.composition_sampler == "fixed_list":
        picks = rng.choice(len(cfg.fixed_compositions), size=count, replace=True)
        return tuple(Composition.from_ratio(cfg.fixed_compositions[int(i)]) for i in picks), None
    return tuple(snap_to_denominator(rng, cfg, num_fluids) for _ in range(count)), None


def _walk_with_support(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    chip: ChipLimits,
    num_fluids: int,
    max_tries: int = 200,
) -> Recipe:
    """A random legal recipe whose product really contains `support_size` fluids.

    A short walk can leave an allowed fluid unused, so a recipe with too small a support is
    redrawn. A mixture stage asking for support s needs at least s - 1 mixes to reach it.
    """
    want = min(cfg.support_size, num_fluids)
    for _ in range(max_tries):
        recipe = random_recipe(rng, num_fluids, chip, _draw(rng, cfg.walk_mixes), cfg.support_size)
        if support_size_of(composition_of(recipe, num_fluids)) >= want:
            return recipe
    raise RuntimeError(
        f"generator '{cfg.name}': no reachable composition over {want} fluids in {max_tries} "
        f"walks on chip '{chip.name}'. A support of {want} needs walk_mixes of at least "
        f"{want - 1}; raise walk_mixes (now {list(cfg.walk_mixes)})."
    )


def sample_instance(
    cfg: GeneratorConfig,
    chip: ChipLimits,
    rng: np.random.Generator,
    weights: CostWeights | None = None,
    instance_id: str | None = None,
    exclude: Container[str] | None = None,
    max_tries: int = 1000,
) -> Instance:
    """One instance from this stage, with a verified witness plan recorded in `metadata`.

    `metadata` holds the generator name and version, the config hash, the witness plan length
    and, when `weights` is given, `known_upper_bound` (the witness cost as an exact `"a/b"`).

    `exclude` is a set of canonical hashes to avoid (`generators.benchmark_io.instance_hash` or
    `problem_hash`; `BenchmarkSet.hashes()` gives both). Training passes a frozen evaluation
    set's hashes so a small stage cannot leak its own held-out instances; a stage with nothing
    left to draw raises rather than looping forever.
    """
    if exclude is not None:
        for _ in range(max_tries):
            candidate = _sample_once(cfg, chip, rng, weights, instance_id)
            if not _is_excluded(candidate, exclude):
                return candidate
        raise RuntimeError(
            f"generator '{cfg.name}': every one of {max_tries} draws was in the excluded set. "
            f"This stage is too small to hold out that many instances."
        )
    return _sample_once(cfg, chip, rng, weights, instance_id)


def _is_excluded(instance: Instance, exclude: Container[str]) -> bool:
    from meda.generators.benchmark_io import instance_hash, problem_hash

    return instance_hash(instance) in exclude or problem_hash(instance) in exclude


def _hash_of(instance: Instance) -> str:
    from meda.generators.benchmark_io import instance_hash

    return instance_hash(instance)


def _sample_once(
    cfg: GeneratorConfig,
    chip: ChipLimits,
    rng: np.random.Generator,
    weights: CostWeights | None,
    instance_id: str | None,
) -> Instance:
    chip = sample_chip(cfg, chip, rng)
    fluids = make_fluids(cfg, rng)
    num_experiments = _draw(rng, cfg.num_experiments)
    per_experiment = [_draw(rng, cfg.targets_per_experiment) for _ in range(num_experiments)]
    total = sum(per_experiment)
    comps, recipes = _sample_compositions(rng, cfg, chip, total, len(fluids))

    targets: list[Target] = []
    experiment_of = [e for e, n in enumerate(per_experiment) for _ in range(n)]
    for i, comp in enumerate(comps):
        volume = _draw_if_range(rng, cfg.droplet_volume)
        if volume > chip.max_mix_volume:
            raise ValueError(
                f"generator '{cfg.name}': drew droplet_volume {volume} > max_mix_volume "
                f"{chip.max_mix_volume} of chip '{chip.name}'; no droplet could be delivered. "
                f"Narrow droplet_volume or raise max_mix_volume."
            )
        targets.append(
            Target(
                id=i,
                experiment_id=experiment_of[i],
                composition=comp,
                tolerance=cfg.tolerance,
                droplet_volume=volume,
                demand=_draw(rng, cfg.demand),
            )
        )
    name = instance_id or f"{cfg.name}-{cfg.config_hash()}-{_instance_suffix(targets, chip)}"
    probe = Instance(name, fluids, tuple(targets), chip, 10_000)
    actions = _witness(probe, recipes)
    step_limit = cfg.step_limit or max(len(actions), round(cfg.step_limit_multiple * len(actions)))
    if step_limit < len(actions):
        raise ValueError(
            f"instance '{name}': step_limit {step_limit} is below the witness plan length "
            f"{len(actions)}; raise step_limit for generator '{cfg.name}'"
        )
    metadata: dict[str, Any] = {
        "generator": cfg.name,
        "generator_version": GENERATOR_VERSION,
        "config_hash": cfg.config_hash(),
        "composition_sampler": cfg.composition_sampler,
        "witness_actions": len(actions),
        "witness_source": "reachable_walk" if recipes else "naive",
    }
    if recipes is not None:
        # Store the walk so the witness stays re-derivable from the instance alone: the naive
        # construction cannot rebuild a mixture whose denominator exceeds the mix limit.
        metadata["recipes"] = [recipe_to_dict(r) for r in recipes]
    if weights is not None:
        _, totals, _ = replay(probe, actions)
        metadata["known_upper_bound"] = str(weights.scalar(totals, probe.fluids))
    return replace(probe, step_limit=step_limit, metadata=metadata)


def _witness(instance: Instance, recipes: tuple[Recipe, ...] | None) -> tuple[Action, ...]:
    """A legal plan meeting all demand: the walk's own recipes, else the naive construction."""
    if recipes is not None:
        return witness_plan(instance, recipes)
    actions = naive_plan(instance)
    if actions is None:
        raise ValueError(
            f"instance '{instance.instance_id}': no feasibility witness. The naive construction "
            f"cannot build these targets on chip '{instance.chip.name}' (denominators must "
            f"factorise into mix volumes within [{instance.chip.min_mix_volume}, "
            f"{instance.chip.max_mix_volume}], and mixtures need denominator <= "
            f"{instance.chip.max_mix_volume}). Use composition_sampler=reachable_walk, or lower "
            f"accuracy_denominator."
        )
    return actions


def _instance_suffix(targets: Sequence[Target], chip: ChipLimits) -> str:
    payload = json.dumps(
        [[list(t.composition.num), t.composition.den, t.demand] for t in targets] + [chip.name],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def sample_instances(
    cfg: GeneratorConfig,
    chip: ChipLimits,
    rng: np.random.Generator,
    count: int,
    weights: CostWeights | None = None,
    unique: bool = True,
    exclude: Container[str] | None = None,
) -> list[Instance]:
    """`count` instances. With `unique`, duplicates (by canonical hash) are resampled."""
    out: list[Instance] = []
    seen: set[str] = set()
    tries = 0
    budget = 100 * count + 100
    while len(out) < count:
        tries += 1
        if tries > budget:
            raise RuntimeError(
                f"generator '{cfg.name}': only {len(out)} of {count} distinct instances after "
                f"{tries} draws; the stage is too small to sample {count} without repeats"
            )
        inst = sample_instance(cfg, chip, rng, weights, exclude=exclude)
        key = _hash_of(inst)
        if unique and key in seen:
            continue
        seen.add(key)
        out.append(inst)
    return out


def dilution_grid(
    chip: ChipLimits,
    denominator: int,
    step_limit: int,
    demand: int = 1,
    droplet_volume: int = 1,
    tolerance: Rational = ZERO,
) -> list[Instance]:
    """All targets n/denominator (0 < n < denominator, reduced duplicates removed) of sample S
    diluted with buffer B, one target per instance."""
    if denominator < 2:
        raise ValueError(f"dilution_grid: denominator must be >= 2, got {denominator}")
    fluids = (Fluid(0, "S", FluidKind.SAMPLE), Fluid(1, "B", FluidKind.BUFFER))
    seen: set[Composition] = set()
    out = []
    for n in range(1, denominator):
        comp = Composition.from_ints((n, denominator - n), denominator)
        if comp in seen:
            continue
        seen.add(comp)
        target = Target(0, 0, comp, tolerance, droplet_volume, demand)
        out.append(
            Instance(
                f"grid-{chip.name}-d{denominator}-{comp.num[0]}of{comp.den}",
                fluids,
                (target,),
                chip,
                step_limit,
                {"generator": "dilution_grid", "denominator": denominator},
            )
        )
    return out
