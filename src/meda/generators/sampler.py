"""The start-state distribution p(s0): `sample_instance(cfg, rng)` (SPEC 8.1).

Every sampled instance is **solvable by construction**: the sampler builds a feasibility witness
(a legal plan that meets all demand) before returning, and records its cost as a known upper
bound in `metadata`. A config that cannot be witnessed raises an error naming the instance, so
training never draws an instance nobody can solve.

Two composition samplers (SPEC 8.1):
- `dirichlet_snapped` — sample the simplex, then snap to a composition whose denominator divides
  the accuracy denominator D (fixed, or drawn per target from a range). Witnessed by the naive
  construction, else the split tree. An exact target (tolerance 0) needs every prime factor of
  D at most the instance's M, so a range keeps only those D; a tolerant target may use any D,
  and the tree builds the coarsest reachable composition within its tolerance.
- `reachable_walk` — compositions a random legal plan actually produced (`generators.synthetic`).
  Reachable by construction, and the walk itself is the witness.

`dilution_grid` is a deterministic helper (not a p(s0)): every single-target dilution n/D of one
sample with one buffer, the "sweep all targets at an accuracy level" set of the reporting
conventions.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from meda.solvers.baselines.tree import min_denominator_within, tree_plan

COMPOSITION_SAMPLERS = ("dirichlet_snapped", "reachable_walk", "fixed_list")
# How an instance's targets relate (SPEC 8.1): drawn independently by the composition sampler,
# or a linear dilution gradient (one denominator, evenly spaced sample shares), the structured
# multi-target case where sharing pays most (Track 2 E2).
TARGET_STRUCTURES = ("independent", "gradient")
GENERATOR_VERSION = 1
# How an instance's step limit is sized when `step_limit` is null (SPEC 8.1):
# - "depth": multiple x a bound from the problem alone, sum over deliveries of (the fewest mix
#   levels any plan needs for that target + 1 output), so the budget does not depend on how
#   good a construction is;
# - "witness": multiple x the witness plan's length (the rule of generator version 1).
STEP_LIMIT_RULES = ("depth", "witness")
DEFAULT_STEP_LIMIT_MULTIPLE = {"depth": 4.0, "witness": 2.0}


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
    accuracy_denominator: tuple[int, int]  # drawn per target when a real range
    support_size: tuple[int, int]  # drawn per target when a real range
    demand: tuple[int, int]
    droplet_volume: tuple[int, int]  # drawn per target
    tolerance: tuple[Rational, ...]  # one is chosen per target when there are several
    step_limit: int | None  # None: size it by `step_limit_rule`
    step_limit_multiple: tuple[float, float]  # drawn per instance when a real range
    walk_mixes: tuple[int, int]
    fixed_compositions: tuple[tuple[int, ...], ...]
    # Chip inputs drawn per instance (SPEC 1.3); None keeps the chip config's value.
    max_mix_volume: tuple[int, int] | None = None
    storage_capacity: tuple[int, int] | None = None
    step_limit_rule: str = "depth"
    # Per-problem bound on intermediate denominators (SPEC 1.7: a property of the problem, not
    # the chip); None keeps the chip config's value.
    max_denominator: int | None = None
    # `droplet_volume: [low, M]`: the high end is each instance's own M (SPEC 1.3: 1 to M).
    droplet_volume_to_m: bool = False
    # One is chosen per instance when there are several (repeat an entry to weight it).
    target_structure: tuple[str, ...] = ("independent",)

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
        s_low, s_high = self.support_size
        if not 2 <= s_low <= s_high or s_low > self.min_fluids:
            fail(f"support_size {s_low} outside [2, {self.min_fluids}] (or not a range)")
        d_low, d_high = self.accuracy_denominator
        if d_low < 2 or d_high < d_low:
            fail(f"accuracy_denominator must be >= 2 (or an increasing range), got {d_low}")
        if self.max_denominator is not None and self.max_denominator < 2:
            fail(f"max_denominator must be >= 2, got {self.max_denominator}")
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
        if not self.tolerance or any(t < 0 for t in self.tolerance):
            fail(f"tolerance must be >= 0 (or a list of such), got {self.tolerance}")
        if self.step_limit_rule not in STEP_LIMIT_RULES:
            fail(f"step_limit_rule must be one of {STEP_LIMIT_RULES}")
        low, high = self.step_limit_multiple
        if low <= 0 or high < low:
            fail(f"step_limit_multiple must be > 0 (or an increasing range), got {(low, high)}")
        if self.composition_sampler == "reachable_walk" and (
            self.support_size[0] != self.support_size[1]
        ):
            fail("reachable_walk takes a fixed support_size")
        if self.composition_sampler == "fixed_list" and not self.fixed_compositions:
            fail("fixed_list needs a non-empty fixed_compositions list")
        if not self.target_structure or any(
            t not in TARGET_STRUCTURES for t in self.target_structure
        ):
            fail(f"target_structure entries must be among {TARGET_STRUCTURES}")
        if "gradient" in self.target_structure and (
            self.composition_sampler != "dirichlet_snapped"
            or self.support_size != (2, 2)
            or self.max_fluids != 2
        ):
            fail(
                "a gradient is a dilution: it needs dirichlet_snapped, support_size 2 and "
                "exactly one sample and one buffer"
            )

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
            "step_limit_rule",
            "walk_mixes",
            "fixed_compositions",
            "max_mix_volume",
            "storage_capacity",
            "max_denominator",
            "target_structure",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"generator config has unknown keys {sorted(unknown)}")
        counts = data["num_fluids"]
        step_limit = data.get("step_limit")
        rule = str(data.get("step_limit_rule", "depth"))
        if rule not in DEFAULT_STEP_LIMIT_MULTIPLE:
            raise ValueError(
                f"generator '{data['name']}': step_limit_rule must be one of {STEP_LIMIT_RULES}"
            )
        return GeneratorConfig(
            name=str(data["name"]),
            samples=_as_range(counts.get("samples", 1)),
            reagents=_as_range(counts.get("reagents", 0)),
            buffers=_as_range(counts.get("buffers", 1)),
            num_experiments=_as_range(data["num_experiments"]),
            targets_per_experiment=_as_range(data["targets_per_experiment"]),
            composition_sampler=str(data["composition_sampler"]),
            accuracy_denominator=_as_range(data["accuracy_denominator"]),
            support_size=_as_range(data["support_size"]),
            demand=_as_range(data["demand"]),
            droplet_volume=_as_range(_volume_span(data["droplet_volume"])),
            droplet_volume_to_m=_to_m(data["droplet_volume"]),
            tolerance=_choices(data["tolerance"]),
            step_limit=None if step_limit is None else int(step_limit),
            step_limit_rule=rule,
            step_limit_multiple=_float_range(
                data.get("step_limit_multiple", DEFAULT_STEP_LIMIT_MULTIPLE[rule])
            ),
            walk_mixes=_as_range(data.get("walk_mixes", [1, 3])),
            fixed_compositions=tuple(
                tuple(int(x) for x in parts) for parts in data.get("fixed_compositions", ())
            ),
            max_mix_volume=_optional_range(data.get("max_mix_volume")),
            storage_capacity=_optional_range(data.get("storage_capacity")),
            max_denominator=None
            if data.get("max_denominator") is None
            else int(data["max_denominator"]),
            target_structure=_names(data.get("target_structure", "independent")),
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
            "accuracy": _hashable_span(self.accuracy_denominator),
            "support": _hashable_span(self.support_size),
            "demand": list(self.demand),
            "droplet_volume": (
                [self.droplet_volume[0], "M"]
                if self.droplet_volume_to_m
                else _hashable_span(self.droplet_volume)
            ),
            "tolerance": (
                str(self.tolerance[0])
                if len(self.tolerance) == 1
                else [str(t) for t in self.tolerance]
            ),
            "step_limit": self.step_limit,
            "step_limit_multiple": _hashable_float_span(self.step_limit_multiple),
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
        if self.max_denominator is not None:
            payload["max_denominator"] = self.max_denominator
        if self.step_limit_rule != "witness":  # the version-1 rule is the unwritten one
            payload["step_limit_rule"] = self.step_limit_rule
        if self.target_structure != ("independent",):
            payload["target_structure"] = list(self.target_structure)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _as_range(value: Any) -> tuple[int, int]:
    """Accepts `3` (a fixed count) or `[low, high]`."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError(f"expected a [low, high] range, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _names(value: Any) -> tuple[str, ...]:
    """Accepts one name or a list of them."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(str(v) for v in value)
    return (str(value),)


def _to_m(value: Any) -> bool:
    """Whether a `droplet_volume` range ends at the instance's own M (`[low, M]`)."""
    return isinstance(value, Sequence) and not isinstance(value, str) and value[-1] == "M"


def _volume_span(value: Any) -> Any:
    """`[low, M]` as `[low, low]` for validation; the high end is drawn per instance."""
    return [value[0], value[0]] if _to_m(value) else value


def _choices(value: Any) -> tuple[Rational, ...]:
    """Accepts one tolerance or a list of them (one is chosen per target)."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(Rational.parse(v) for v in value)
    return (Rational.parse(value),)


def is_smooth(n: int, m: int) -> bool:
    """Every prime factor of n is at most m: a mix of at most m units can reach denominator n
    (SPEC 10.3)."""
    for p in range(2, m + 1):
        while n % p == 0:
            n //= p
    return n == 1


def _float_range(value: Any) -> tuple[float, float]:
    """Accepts `2.5` (fixed) or `[low, high]`."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError(f"expected a [low, high] range, got {value}")
        return float(value[0]), float(value[1])
    return float(value), float(value)


def _hashable_float_span(span: tuple[float, float]) -> float | list[float]:
    return span[0] if span[0] == span[1] else list(span)


def mix_depth_bound(den: int, m: int) -> int:
    """The fewest mix levels any plan needs to reach denominator `den`: a mix of at most `m`
    units multiplies the denominator by at most m, so depth d reaches at most m**d."""
    depth, reach = 0, 1
    while reach < den:
        reach *= m
        depth += 1
    return depth


def step_limit_base(instance: Instance) -> int:
    """The "depth" rule's base: per delivery, its target's mix-depth bound plus one output. A
    tolerant target's bound uses the smallest denominator of any composition within its
    tolerance, since a plan may deliver that one."""
    chip = instance.chip
    m = chip.max_mix_volume
    return sum(
        t.demand
        * (
            mix_depth_bound(
                min_denominator_within(t.composition, t.tolerance, chip.max_denominator), m
            )
            + 1
        )
        for t in instance.targets
    )


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
    generator's ranges where it sets them (SPEC 1.3: chip limits are problem inputs), and the
    generator's `max_denominator` if it sets one. The name records what was drawn; it is a
    label, not part of the problem (`problem_hash`)."""
    if cfg.max_denominator is not None:
        chip = replace(chip, max_denominator=cfg.max_denominator)
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
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    num_fluids: int,
    den: int | None = None,
    support: int | None = None,
) -> Composition:
    """Sample the simplex (Dirichlet) over `support` fluids, then snap to denominator `den`
    (default: the config's fixed values).

    Fluid 0 (a SAMPLE) is always in the support, so a dilution target really is sample+buffer.
    Every share is at least one unit, so the support size is exact.
    """
    den = cfg.accuracy_denominator[0] if den is None else den
    support = cfg.support_size[0] if support is None else support
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
    tolerances: Sequence[Rational],
) -> tuple[tuple[Composition, ...], tuple[Recipe, ...] | None]:
    """Target compositions, and the recipes that make them when the sampler provides any."""
    if cfg.composition_sampler == "reachable_walk":
        recipes = tuple(_walk_with_support(rng, cfg, chip, num_fluids) for _ in range(count))
        return tuple(composition_of(r, num_fluids) for r in recipes), recipes
    if cfg.composition_sampler == "fixed_list":
        picks = rng.choice(len(cfg.fixed_compositions), size=count, replace=True)
        return tuple(Composition.from_ratio(cfg.fixed_compositions[int(i)]) for i in picks), None
    if (
        cfg.accuracy_denominator[0] == cfg.accuracy_denominator[1]
        and cfg.support_size[0] == (cfg.support_size[1])
    ):
        return tuple(snap_to_denominator(rng, cfg, num_fluids) for _ in range(count)), None
    comps = []
    for tol in tolerances:
        dens = _denominators(cfg, chip, exact=tol.num == 0)
        den = dens[int(rng.integers(len(dens)))]
        low, high = cfg.support_size
        support = _draw(rng, (low, min(high, num_fluids, den)))
        comps.append(snap_to_denominator(rng, cfg, num_fluids, den, support))
    return tuple(comps), None


def _gradient(
    rng: np.random.Generator, cfg: GeneratorConfig, chip: ChipLimits, count: int
) -> tuple[Composition, ...]:
    """A linear dilution gradient of `count` targets: one exact denominator D the instance's M
    can reach, sample shares lo, lo + step, ... evenly spaced inside (0, D)."""
    dens = [d for d in _denominators(cfg, chip, exact=True) if d - 1 >= count]
    if not dens:
        raise ValueError(
            f"generator '{cfg.name}': no accuracy denominator in {list(cfg.accuracy_denominator)} "
            f"reachable at M = {chip.max_mix_volume} has {count} distinct shares for a "
            f"{count}-target gradient. Widen accuracy_denominator."
        )
    den = dens[int(rng.integers(len(dens)))]
    step = max(1, (den - 1) // (count + 1))
    low = int(rng.integers(1, den - (count - 1) * step))
    return tuple(
        Composition.from_ints((low + i * step, den - low - i * step), den) for i in range(count)
    )


def _denominators(cfg: GeneratorConfig, chip: ChipLimits, exact: bool) -> list[int]:
    """The accuracy denominators a target may draw: every D in the range, or for an exact
    target only those the instance's M can reach (prime factors <= M)."""
    low, high = cfg.accuracy_denominator
    dens = [d for d in range(low, high + 1) if not exact or is_smooth(d, chip.max_mix_volume)]
    if not dens:
        raise ValueError(
            f"generator '{cfg.name}': no accuracy denominator in [{low}, {high}] has every "
            f"prime factor <= max_mix_volume {chip.max_mix_volume} of chip '{chip.name}', so "
            f"no exact target can be drawn. Widen accuracy_denominator."
        )
    return dens


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
    want = min(cfg.support_size[0], num_fluids)
    for _ in range(max_tries):
        recipe = random_recipe(
            rng, num_fluids, chip, _draw(rng, cfg.walk_mixes), cfg.support_size[0]
        )
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
    exclude_compositions: Container[Composition] | None = None,
) -> Instance:
    """One instance from this stage, with a verified witness plan recorded in `metadata`.

    `metadata` holds the generator name and version, the config hash, the witness plan length
    and, when `weights` is given, `known_upper_bound` (the witness cost as an exact `"a/b"`).

    `exclude` is a set of canonical hashes to avoid (`generators.benchmark_io.instance_hash` or
    `problem_hash`; `BenchmarkSet.hashes()` gives both). Training passes a frozen evaluation
    set's hashes so a small stage cannot leak its own held-out instances; a stage with nothing
    left to draw raises rather than looping forever.

    `exclude_compositions` holds out target **ratios** rather than whole problems: no target
    of a returned instance has one of them (the "unseen ratios" test sets, SPEC 8.4).
    """
    if exclude is None and exclude_compositions is None:
        return _sample_once(cfg, chip, rng, weights, instance_id)
    for _ in range(max_tries):
        candidate = _sample_once(cfg, chip, rng, weights, instance_id)
        if exclude is not None and _is_excluded(candidate, exclude):
            continue
        if exclude_compositions is not None and any(
            t.composition in exclude_compositions for t in candidate.targets
        ):
            continue
        return candidate
    raise RuntimeError(
        f"generator '{cfg.name}': every one of {max_tries} draws was in the excluded set. "
        f"This stage is too small to hold out that many instances or ratios."
    )


class UnwitnessedError(ValueError):
    """No construction found a feasibility witness for a drawn instance (never a claim that the
    instance is infeasible)."""


@dataclass(frozen=True)
class GeneratorSource:
    """An env instance source (`envs.sample_prep_env.InstanceSource`) that draws a fresh
    problem from one generator stage per episode, never one whose hash is in `exclude` (the
    frozen evaluation sets, SPEC 8.4). Picklable, so it works with `AsyncVectorEnv`."""

    config: GeneratorConfig
    chip: ChipLimits
    weights: CostWeights | None = None
    exclude: frozenset[str] = frozenset()
    exclude_compositions: frozenset[Composition] = frozenset()
    max_unwitnessed: int = 100

    def __call__(self, rng: np.random.Generator) -> Instance:
        """A fresh instance. A draw no construction can witness (e.g. a many-fluid target under
        storage 2) is redrawn, never returned, and the count is recorded in the instance's
        `metadata["unwitnessed_redraws"]` (rule 9: not silent). More than `max_unwitnessed`
        in a row raises: the stage asks for problems nobody can build."""
        for redraws in range(self.max_unwitnessed + 1):
            try:
                inst = sample_instance(
                    self.config,
                    self.chip,
                    rng,
                    self.weights,
                    exclude=self.exclude or None,
                    exclude_compositions=self.exclude_compositions or None,
                )
            except UnwitnessedError:
                continue
            return replace(inst, metadata={**inst.metadata, "unwitnessed_redraws": redraws})
        raise UnwitnessedError(
            f"generator '{self.config.name}': {self.max_unwitnessed + 1} draws in a row had no "
            f"feasibility witness; narrow the stage (storage, fluids, accuracy, tolerance)."
        )


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
    tolerances = [
        cfg.tolerance[0]
        if len(cfg.tolerance) == 1
        else cfg.tolerance[int(rng.integers(len(cfg.tolerance)))]
        for _ in range(total)
    ]
    structure = (
        cfg.target_structure[0]
        if len(cfg.target_structure) == 1
        else cfg.target_structure[int(rng.integers(len(cfg.target_structure)))]
    )
    if structure == "gradient":
        comps, recipes = _gradient(rng, cfg, chip, total), None
    else:
        comps, recipes = _sample_compositions(rng, cfg, chip, total, len(fluids), tolerances)

    targets: list[Target] = []
    experiment_of = [e for e, n in enumerate(per_experiment) for _ in range(n)]
    for i, comp in enumerate(comps):
        span = (
            (cfg.droplet_volume[0], chip.max_mix_volume)
            if cfg.droplet_volume_to_m
            else cfg.droplet_volume
        )
        volume = _draw_if_range(rng, span)
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
                tolerance=tolerances[i],
                droplet_volume=volume,
                demand=_draw(rng, cfg.demand),
            )
        )
    name = instance_id or f"{cfg.name}-{cfg.config_hash()}-{_instance_suffix(targets, chip)}"
    probe = Instance(name, fluids, tuple(targets), chip, 10_000)
    actions, witness_source = _witness(probe, recipes)
    sizing: dict[str, Any] = {}
    if cfg.step_limit is not None:
        step_limit = cfg.step_limit
    else:
        low, high = cfg.step_limit_multiple
        multiple = low if low == high else float(rng.uniform(low, high))
        if cfg.step_limit_rule == "witness":
            step_limit = max(len(actions), round(multiple * len(actions)))
        else:
            base = step_limit_base(probe)
            sized = math.ceil(multiple * base)
            step_limit = max(sized, len(actions))
            sizing = {
                "step_limit_rule": "depth",
                "step_limit_base": base,
                "step_limit_multiple": multiple,
                # Never silent (rule 9): a budget below the known plan is raised to fit it.
                "step_limit_raised_to_witness": sized < len(actions),
            }
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
        **({"target_structure": structure} if structure != "independent" else {}),
        "witness_actions": len(actions),
        "witness_source": witness_source,
        **sizing,
    }
    if recipes is not None:
        # Store the walk so the witness stays re-derivable from the instance alone: the naive
        # construction cannot rebuild a mixture whose denominator exceeds the mix limit.
        metadata["recipes"] = [recipe_to_dict(r) for r in recipes]
    if weights is not None:
        _, totals, _ = replay(probe, actions)
        metadata["known_upper_bound"] = str(weights.scalar(totals, probe.fluids))
    return replace(probe, step_limit=step_limit, metadata=metadata)


def _witness(
    instance: Instance, recipes: tuple[Recipe, ...] | None
) -> tuple[tuple[Action, ...], str]:
    """A legal plan meeting all demand: the walk's own recipes, else the naive construction,
    else the split tree (which builds any reachable target). Naive goes first so every
    instance it already witnessed is unchanged. Returns the plan and which construction made
    it (recorded as `witness_source`)."""
    if recipes is not None:
        return witness_plan(instance, recipes), "reachable_walk"
    actions = naive_plan(instance)
    if actions is not None:
        return actions, "naive"
    actions = tree_plan(instance)
    if actions is None:
        raise UnwitnessedError(
            f"instance '{instance.instance_id}': no feasibility witness on chip "
            f"'{instance.chip.name}'. A target is reachable only if every prime factor of its "
            f"denominator is <= max_mix_volume {instance.chip.max_mix_volume}, and the split "
            f"tree also needs storage for its depth (storage_capacity "
            f"{instance.chip.storage_capacity}); a tolerant target needs a reachable "
            f"composition within its tolerance at denominator <= max_denominator "
            f"{instance.chip.max_denominator}. Lower accuracy_denominator, raise "
            f"max_mix_volume or max_denominator, or use composition_sampler=reachable_walk."
        )
    return actions, "tree"


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
