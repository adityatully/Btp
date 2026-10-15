"""Frozen, versioned benchmark sets and the train/test overlap check (SPEC 8.4).

A benchmark set is one JSON file holding its name, version, the generator config and seeds it
came from, and the instances. **Never regenerate a published version; create a new one.**
Training draws from the sampler; evaluation uses only these files.

`instance_hash` is the canonical identity of a problem: the fluid kinds, the multiset of targets
and the chip limits. It deliberately ignores the instance id, the metadata and the step limit,
so a training draw cannot sneak past the overlap check by being renamed or given more steps.
`reference` holds results stored per instance (M5 fills in the optima of `tiny_exact_v1`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from meda.core.composition import Rational
from meda.core.fluids import FluidKind
from meda.core.problem import Instance

BENCHMARK_FORMAT = "meda-benchmark/1"
BENCHMARKS_DIR = Path(__file__).resolve().parents[3] / "benchmarks"


def _targets_payload(instance: Instance) -> list[Any]:
    return sorted(
        (
            list(t.composition.num),
            t.composition.den,
            [t.tolerance.num, t.tolerance.den],
            t.droplet_volume,
            t.demand,
        )
        for t in instance.targets
    )


def instance_hash(instance: Instance) -> str:
    """Canonical hash of the problem: fluid kinds, target multiset, chip limits."""
    payload = {
        "fluids": [f.kind.value for f in instance.fluids],
        "targets": _targets_payload(instance),
        "chip": instance.chip.to_dict(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def problem_hash(instance: Instance) -> str:
    """Like `instance_hash` but blind to `max_denominator` (an accuracy ceiling, not the problem).

    `tiny_exact_v1` sets each instance's `max_denominator` to the lcm of its targets, so the
    same dilution has a different `instance_hash` there than in a stage-1 draw. Holding a
    problem out of training has to compare this hash.
    """
    chip = {k: v for k, v in instance.chip.to_dict().items() if k != "max_denominator"}
    payload = {
        "fluids": [f.kind.value for f in instance.fluids],
        "targets": _targets_payload(instance),
        "chip": chip,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class InstanceReference:
    """A stored result for one instance (the "answer key" of a frozen set)."""

    cost: Rational
    status: str  # "optimal" (exact search finished) | "feasible" (an upper bound)
    producer: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost": str(self.cost),
            "status": self.status,
            "producer": self.producer,
            "detail": dict(self.detail),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> InstanceReference:
        return InstanceReference(
            cost=Rational.parse(data["cost"]),
            status=str(data["status"]),
            producer=str(data["producer"]),
            detail=dict(data.get("detail", {})),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSet:
    name: str
    version: int
    instances: tuple[Instance, ...]
    generator: Mapping[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()
    description: str = ""
    reference: Mapping[str, InstanceReference] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [i.instance_id for i in self.instances]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"benchmark '{self.name}': duplicate instance ids {duplicates}")
        unknown = set(self.reference) - set(ids)
        if unknown:
            raise ValueError(
                f"benchmark '{self.name}': reference for unknown instances {sorted(unknown)}"
            )

    @property
    def full_name(self) -> str:
        return f"{self.name}_v{self.version}"

    def __len__(self) -> int:
        return len(self.instances)

    def hashes(self) -> set[str]:
        """`instance_hash`es plus `problem_hash`es; pass as `exclude` to hold the set out."""
        return {instance_hash(i) for i in self.instances} | {
            problem_hash(i) for i in self.instances
        }

    def by_id(self, instance_id: str) -> Instance:
        for inst in self.instances:
            if inst.instance_id == instance_id:
                return inst
        raise KeyError(f"benchmark '{self.name}' has no instance '{instance_id}'")

    def with_reference(self, reference: Mapping[str, InstanceReference]) -> BenchmarkSet:
        """A copy with these stored results (M5 stores the optima this way)."""
        return replace(self, reference={**self.reference, **reference})

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BENCHMARK_FORMAT,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "generator": dict(self.generator),
            "seeds": list(self.seeds),
            "reference": {k: v.to_dict() for k, v in sorted(self.reference.items())},
            "instances": [i.to_dict() for i in self.instances],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> BenchmarkSet:
        if data.get("format") != BENCHMARK_FORMAT:
            raise ValueError(
                f"unsupported benchmark format {data.get('format')!r}, want {BENCHMARK_FORMAT}"
            )
        return BenchmarkSet(
            name=str(data["name"]),
            version=int(data["version"]),
            instances=tuple(Instance.from_dict(i) for i in data["instances"]),
            generator=dict(data.get("generator", {})),
            seeds=tuple(int(s) for s in data.get("seeds", ())),
            description=str(data.get("description", "")),
            reference={
                k: InstanceReference.from_dict(v) for k, v in data.get("reference", {}).items()
            },
        )

    def write(self, directory: Path | str = BENCHMARKS_DIR, overwrite: bool = False) -> Path:
        """Write `<name>_v<version>.json`. Refuses to overwrite a published version."""
        path = Path(directory) / f"{self.full_name}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"benchmark '{self.full_name}' already exists at {path}. A published version is "
                f"frozen: bump `version` instead of regenerating it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=False) + "\n")
        return path

    @staticmethod
    def read(name: str, directory: Path | str = BENCHMARKS_DIR) -> BenchmarkSet:
        """Read a set by file stem (`tiny_exact_v1`) or by name (newest version)."""
        directory = Path(directory)
        path = directory / f"{name}.json"
        if not path.exists():
            matches = sorted(directory.glob(f"{name}_v*.json"))
            if not matches:
                available = sorted(p.stem for p in directory.glob("*_v*.json"))
                raise FileNotFoundError(
                    f"no benchmark '{name}' in {directory} (have: {available or 'none'})"
                )
            path = matches[-1]
        return BenchmarkSet.from_dict(json.loads(path.read_text()))


def check_no_overlap(
    training: Iterable[Instance], frozen: Sequence[BenchmarkSet] | BenchmarkSet
) -> None:
    """Raise if any training instance is the same problem as a frozen evaluation instance."""
    sets = [frozen] if isinstance(frozen, BenchmarkSet) else list(frozen)
    index: dict[str, tuple[str, str]] = {}
    for bench in sets:
        for inst in bench.instances:
            index[problem_hash(inst)] = (bench.full_name, inst.instance_id)
    clashes = [
        (inst.instance_id, *index[problem_hash(inst)])
        for inst in training
        if problem_hash(inst) in index
    ]
    if clashes:
        lines = "\n".join(f"  {t} == {name}:{i}" for t, name, i in clashes[:10])
        more = f"\n  ... {len(clashes) - 10} more" if len(clashes) > 10 else ""
        raise ValueError(f"training instances overlap frozen benchmark sets:\n{lines}{more}")


def kind_counts(instance: Instance) -> dict[str, int]:
    """Fluids per kind, for benchmark summaries."""
    return {kind.value: sum(1 for f in instance.fluids if f.kind is kind) for kind in FluidKind}
