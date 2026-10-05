"""Mixing-DAG format: the single output of every solver and agent (SPEC section 7).

- `DagNode`, `DagEdge`, `DagDelta`: what one action adds (returned by `rules.apply`).
- `MixingDAG.from_deltas`: builds the full DAG of an episode and adds RESERVOIR nodes.
- `validate(dag, instance)`: every check of SPEC 7.3, each reported under its own `DagCheck`.
- JSON I/O (`to_json` / `from_json`) and `to_networkx`.

Node ids: reservoir f is node f (0..F-1). Every other node takes the next value of the shared
`PrepState.next_uid` counter. Deltas do not contain RESERVOIR nodes; the builder adds one per
fluid that appears as an edge source, with the total volume drawn from it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from meda.core.composition import Composition, linf_distance, mix
from meda.core.costs import StepCost
from meda.core.problem import Instance

DAG_FORMAT = "meda-mixing-dag/1"


class NodeType(Enum):
    RESERVOIR = "RESERVOIR"
    MIX = "MIX"
    OUTPUT = "OUTPUT"
    WASTE = "WASTE"


@dataclass(frozen=True, slots=True)
class DagNode:
    """`volume` is drawn (RESERVOIR), produced (MIX), delivered (OUTPUT) or wasted (WASTE);
    `step` is the action index that created the node (first use, for a RESERVOIR)."""

    id: int
    type: NodeType
    composition: Composition
    volume: int
    step: int
    fluid_id: int | None = None
    target_id: int | None = None


@dataclass(frozen=True, slots=True)
class DagEdge:
    """`volume` units flow from node `src` to node `dst`."""

    src: int
    dst: int
    volume: int


@dataclass(frozen=True, slots=True)
class DagDelta:
    """Nodes and edges added by one action."""

    nodes: tuple[DagNode, ...]
    edges: tuple[DagEdge, ...]


@dataclass(frozen=True, slots=True)
class MixingDAG:
    """A sample-preparation plan. `totals` are the totals reported by the producer; the
    validator recomputes them from the graph. `nodes` are sorted by id."""

    instance_id: str
    producer: str
    nodes: tuple[DagNode, ...]
    edges: tuple[DagEdge, ...]
    totals: StepCost

    @staticmethod
    def from_deltas(
        instance: Instance, deltas: Iterable[DagDelta], totals: StepCost, producer: str
    ) -> MixingDAG:
        """Join the deltas of an episode and add one RESERVOIR node per fluid used."""
        nodes: list[DagNode] = []
        edges: list[DagEdge] = []
        for delta in deltas:
            nodes.extend(delta.nodes)
            edges.extend(delta.edges)
        step_of = {n.id: n.step for n in nodes}
        drawn: dict[int, int] = defaultdict(int)
        first_step: dict[int, int] = {}
        for e in edges:
            if e.src < instance.num_fluids:
                drawn[e.src] += e.volume
                first_step.setdefault(e.src, step_of.get(e.dst, 0))
        reservoirs = [
            DagNode(
                f,
                NodeType.RESERVOIR,
                Composition.pure(f, instance.num_fluids),
                drawn[f],
                first_step[f],
                fluid_id=f,
            )
            for f in sorted(drawn)
        ]
        return MixingDAG(
            instance.instance_id,
            producer,
            tuple(sorted(reservoirs + nodes, key=lambda n: n.id)),
            tuple(edges),
            totals,
        )

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DAG_FORMAT,
            "instance_id": self.instance_id,
            "producer": self.producer,
            "totals": {
                "reagent_by_fluid": list(self.totals.reagent_by_fluid),
                "mix_ops": self.totals.mix_ops,
                "waste_volume": self.totals.waste_volume,
                "outputs": self.totals.outputs,
            },
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "fluid_id": n.fluid_id,
                    "target_id": n.target_id,
                    "composition": {"num": list(n.composition.num), "den": n.composition.den},
                    "volume": n.volume,
                    "step": n.step,
                }
                for n in self.nodes
            ],
            "edges": [{"src": e.src, "dst": e.dst, "volume": e.volume} for e in self.edges],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> MixingDAG:
        if data.get("format") != DAG_FORMAT:
            raise ValueError(f"unsupported DAG format {data.get('format')!r}, want {DAG_FORMAT}")
        t = data["totals"]
        return MixingDAG(
            instance_id=data["instance_id"],
            producer=data["producer"],
            nodes=tuple(
                DagNode(
                    id=n["id"],
                    type=NodeType(n["type"]),
                    composition=Composition(
                        tuple(n["composition"]["num"]), n["composition"]["den"]
                    ),
                    volume=n["volume"],
                    step=n["step"],
                    fluid_id=n["fluid_id"],
                    target_id=n["target_id"],
                )
                for n in data["nodes"]
            ),
            edges=tuple(DagEdge(e["src"], e["dst"], e["volume"]) for e in data["edges"]),
            totals=StepCost(
                tuple(t["reagent_by_fluid"]), t["mix_ops"], t["waste_volume"], t["outputs"]
            ),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @staticmethod
    def from_json(path: str | Path) -> MixingDAG:
        return MixingDAG.from_dict(json.loads(Path(path).read_text()))

    def to_networkx(self) -> Any:
        """`networkx.DiGraph`; node attributes are the `DagNode` fields, edges carry `volume`.

        For analysis only, never in the env hot path."""
        import networkx as nx  # type: ignore[import-untyped]

        g = nx.DiGraph(instance_id=self.instance_id, producer=self.producer)
        for n in self.nodes:
            g.add_node(
                n.id,
                type=n.type,
                composition=n.composition,
                volume=n.volume,
                step=n.step,
                fluid_id=n.fluid_id,
                target_id=n.target_id,
            )
        for e in self.edges:
            g.add_edge(e.src, e.dst, volume=e.volume)
        return g

    def with_producer(self, producer: str) -> MixingDAG:
        return replace(self, producer=producer)


# ---------------------------------------------------------------------- validation


class DagCheck(Enum):
    """One code per check of SPEC 7.3 (plus STRUCTURE and TAKE)."""

    STRUCTURE = "structure"  # ids unique, edges point at existing nodes, types/fields consistent
    ACYCLIC = "acyclic"
    VOLUME_CONSERVATION = "volume_conservation"
    COMPOSITION = "composition"
    MIX_LIMITS = "mix_limits"
    OUTPUT_TOLERANCE = "output_tolerance"
    DEMAND = "demand"
    TOTALS = "totals"
    STORAGE = "storage"
    TAKE = "take"  # step order of uses, partial-take rule, minimum droplet volume


@dataclass(frozen=True, slots=True)
class DagError:
    check: DagCheck
    message: str


_Report = Callable[[DagCheck, str], None]


class DagValidationError(ValueError):
    """Raised by `ValidationReport.raise_if_invalid`."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    instance_id: str
    errors: tuple[DagError, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def failed_checks(self) -> set[DagCheck]:
        return {e.check for e in self.errors}

    def raise_if_invalid(self) -> None:
        if self.errors:
            lines = "\n".join(f"  [{e.check.value}] {e.message}" for e in self.errors[:20])
            more = f"\n  ... {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
            raise DagValidationError(
                f"DAG for instance '{self.instance_id}' is invalid:\n{lines}{more}"
            )


def validate(dag: MixingDAG, instance: Instance, complete: bool = True) -> ValidationReport:
    """Check `dag` against `instance` (SPEC 7.3).

    `complete=False` validates an unfinished plan (truncated or dead-end episode): demand may be
    unmet and MIX droplets may still be on chip (outflow below volume). Everything else is
    checked the same way.
    """
    errors: list[DagError] = []

    def err(check: DagCheck, message: str) -> None:
        errors.append(DagError(check, message))

    def report() -> ValidationReport:
        return ValidationReport(dag.instance_id, tuple(errors))

    if dag.instance_id != instance.instance_id:
        err(DagCheck.STRUCTURE, f"DAG is for '{dag.instance_id}', not '{instance.instance_id}'")
    nodes = _check_structure(dag, instance, err)
    if errors:
        return report()  # later checks assume a well-formed graph

    parents: dict[int, list[DagEdge]] = defaultdict(list)
    children: dict[int, list[DagEdge]] = defaultdict(list)
    for e in dag.edges:
        parents[e.dst].append(e)
        children[e.src].append(e)

    order = _topological_order(nodes, children)
    if order is None:
        err(DagCheck.ACYCLIC, "graph has a cycle")
        return report()

    chip = instance.chip
    num_fluids = instance.num_fluids
    for nid in order:
        n = nodes[nid]
        inflow = sum(e.volume for e in parents[nid])
        outflow = sum(e.volume for e in children[nid])
        ins = parents[nid]

        # Volume conservation.
        if n.type is NodeType.RESERVOIR:
            if outflow != n.volume:
                err(
                    DagCheck.VOLUME_CONSERVATION,
                    f"reservoir {nid}: outflow {outflow} != drawn volume {n.volume}",
                )
        else:
            if inflow != n.volume:
                err(
                    DagCheck.VOLUME_CONSERVATION,
                    f"{n.type.value} node {nid}: inflow {inflow} != volume {n.volume}",
                )
            if n.type is NodeType.MIX and (
                outflow > n.volume or (complete and outflow != n.volume)
            ):
                err(
                    DagCheck.VOLUME_CONSERVATION,
                    f"MIX node {nid}: outflow {outflow} != volume {n.volume}",
                )

        # Composition consistency.
        if n.type is NodeType.RESERVOIR:
            assert n.fluid_id is not None
            if n.composition != Composition.pure(n.fluid_id, num_fluids):
                err(DagCheck.COMPOSITION, f"reservoir {nid}: composition is not pure fluid")
        elif ins:
            expected = nodes[ins[0].src].composition
            acc = ins[0].volume
            for e in ins[1:]:
                expected = mix(expected, acc, nodes[e.src].composition, e.volume)
                acc += e.volume
            if n.composition != expected:
                err(
                    DagCheck.COMPOSITION,
                    f"{n.type.value} node {nid}: composition {n.composition} != mix of inputs "
                    f"{expected}",
                )

        # Mix limits.
        if n.type is NodeType.MIX:
            if not chip.min_mix_volume <= n.volume <= chip.max_mix_volume:
                err(
                    DagCheck.MIX_LIMITS,
                    f"MIX node {nid}: volume {n.volume} outside "
                    f"[{chip.min_mix_volume}, {chip.max_mix_volume}]",
                )
            srcs = [e.src for e in ins]
            if len(srcs) != 2 or len(set(srcs)) != 2:
                err(DagCheck.MIX_LIMITS, f"MIX node {nid}: needs 2 distinct parents, has {srcs}")
            elif (
                nodes[srcs[0]].composition == nodes[srcs[1]].composition
                and not chip.allow_same_composition_merge
            ):
                err(DagCheck.MIX_LIMITS, f"MIX node {nid}: merges equal compositions")
            if n.composition.den > chip.max_denominator:
                err(
                    DagCheck.MIX_LIMITS,
                    f"MIX node {nid}: denominator {n.composition.den} > {chip.max_denominator}",
                )

        # Outputs.
        if n.type is NodeType.OUTPUT:
            assert n.target_id is not None
            t = instance.targets[n.target_id]
            if n.volume != t.droplet_volume:
                err(
                    DagCheck.OUTPUT_TOLERANCE,
                    f"OUTPUT node {nid}: volume {n.volume} != droplet_volume {t.droplet_volume} "
                    f"of target {t.id}",
                )
            dist = linf_distance(n.composition, t.composition)
            if not dist <= t.tolerance:
                err(
                    DagCheck.OUTPUT_TOLERANCE,
                    f"OUTPUT node {nid}: {dist} from target {t.id} (tolerance {t.tolerance})",
                )

    # Demand.
    delivered = Counter(n.target_id for n in nodes.values() if n.type is NodeType.OUTPUT)
    for t in instance.targets:
        got = delivered.get(t.id, 0)
        if got > t.demand or (complete and got != t.demand):
            err(DagCheck.DEMAND, f"target {t.id}: {got} outputs, demand {t.demand}")

    # Reported totals.
    recomputed = _recompute_totals(nodes, num_fluids)
    if dag.totals != recomputed:
        err(DagCheck.TOTALS, f"reported totals {dag.totals} != recomputed {recomputed}")

    _check_replay(nodes, parents, instance, err)
    return report()


def _check_structure(dag: MixingDAG, instance: Instance, err: _Report) -> dict[int, DagNode]:
    nodes: dict[int, DagNode] = {}
    for n in dag.nodes:
        if n.id in nodes:
            err(DagCheck.STRUCTURE, f"duplicate node id {n.id}")
        nodes[n.id] = n
        if n.volume < 1:
            err(DagCheck.STRUCTURE, f"node {n.id}: volume {n.volume} < 1")
        if n.composition.num_fluids != instance.num_fluids:
            err(DagCheck.STRUCTURE, f"node {n.id}: composition over wrong number of fluids")
        is_res = n.type is NodeType.RESERVOIR
        if is_res != (n.fluid_id is not None) or (is_res and n.fluid_id != n.id):
            err(DagCheck.STRUCTURE, f"node {n.id}: fluid_id {n.fluid_id} inconsistent with type")
        if is_res and not 0 <= n.id < instance.num_fluids:
            err(DagCheck.STRUCTURE, f"node {n.id}: no such reservoir")
        if not is_res and n.id < instance.num_fluids:
            err(DagCheck.STRUCTURE, f"node {n.id}: ids below F are reserved for reservoirs")
        is_out = n.type is NodeType.OUTPUT
        if is_out != (n.target_id is not None):
            err(DagCheck.STRUCTURE, f"node {n.id}: target_id {n.target_id} inconsistent with type")
        if is_out and not 0 <= (n.target_id or 0) < len(instance.targets):
            err(DagCheck.STRUCTURE, f"node {n.id}: no target {n.target_id}")
    in_count: Counter[int] = Counter()
    for e in dag.edges:
        if e.src not in nodes or e.dst not in nodes:
            err(DagCheck.STRUCTURE, f"edge {e.src}->{e.dst}: unknown node")
            continue
        if e.volume < 1:
            err(DagCheck.STRUCTURE, f"edge {e.src}->{e.dst}: volume {e.volume} < 1")
        if nodes[e.dst].type is NodeType.RESERVOIR:
            err(DagCheck.STRUCTURE, f"edge {e.src}->{e.dst}: edge into a reservoir")
        if nodes[e.src].type in (NodeType.OUTPUT, NodeType.WASTE):
            err(DagCheck.STRUCTURE, f"edge {e.src}->{e.dst}: edge out of a sink")
        in_count[e.dst] += 1
    for n in nodes.values():
        if n.type in (NodeType.OUTPUT, NodeType.WASTE) and in_count[n.id] != 1:
            err(DagCheck.STRUCTURE, f"{n.type.value} node {n.id}: needs exactly 1 parent")
    return nodes


def _topological_order(
    nodes: Mapping[int, DagNode], children: Mapping[int, list[DagEdge]]
) -> list[int] | None:
    """Kahn's algorithm, ties broken by id. None if there is a cycle."""
    indeg = dict.fromkeys(nodes, 0)
    for edges in children.values():
        for e in edges:
            indeg[e.dst] += 1
    ready = sorted(nid for nid, d in indeg.items() if d == 0)
    order: list[int] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for e in children.get(nid, []):
            indeg[e.dst] -= 1
            if indeg[e.dst] == 0:
                ready.append(e.dst)
        ready.sort()
    return order if len(order) == len(nodes) else None


def _recompute_totals(nodes: Mapping[int, DagNode], num_fluids: int) -> StepCost:
    reagent = [0] * num_fluids
    mixes = waste = outputs = 0
    for n in nodes.values():
        if n.type is NodeType.RESERVOIR:
            assert n.fluid_id is not None
            reagent[n.fluid_id] += n.volume
        elif n.type is NodeType.MIX:
            mixes += 1
        elif n.type is NodeType.WASTE:
            waste += n.volume
        else:
            outputs += 1
    return StepCost(tuple(reagent), mixes, waste, outputs)


def _check_replay(
    nodes: Mapping[int, DagNode],
    parents: Mapping[int, list[DagEdge]],
    instance: Instance,
    err: _Report,
) -> None:
    """Replay nodes in (step, id) order: uses come after creation, partial takes follow the
    chip rules, and the number of droplets on chip never exceeds storage capacity."""
    chip = instance.chip
    remaining: dict[int, int] = {}  # MIX node id -> units still on chip
    by_step: dict[int, list[DagNode]] = defaultdict(list)
    for n in nodes.values():
        if n.type is not NodeType.RESERVOIR:
            by_step[n.step].append(n)
    for step in sorted(by_step):
        for n in sorted(by_step[step], key=lambda x: x.id):
            for e in parents.get(n.id, []):
                src = nodes[e.src]
                if src.type is NodeType.RESERVOIR:
                    continue
                if e.src not in remaining:
                    err(
                        DagCheck.TAKE,
                        f"node {n.id} (step {n.step}) uses node {e.src}, which is not on chip",
                    )
                    continue
                left = remaining[e.src] - e.volume
                if left < 0:
                    continue  # already reported as a volume conservation error
                if left > 0 and n.type is NodeType.WASTE:
                    err(DagCheck.TAKE, f"WASTE node {n.id}: must take all of {e.src}")
                elif left > 0:
                    if not chip.allow_partial_take:
                        err(DagCheck.TAKE, f"node {n.id}: partial take from {e.src} not allowed")
                    elif left < chip.min_droplet_volume:
                        err(
                            DagCheck.TAKE,
                            f"node {n.id}: leaves {left} units of {e.src}, below "
                            f"min_droplet_volume {chip.min_droplet_volume}",
                        )
                if left == 0:
                    del remaining[e.src]
                else:
                    remaining[e.src] = left
            if n.type is NodeType.MIX:
                remaining[n.id] = n.volume
        if len(remaining) > chip.storage_capacity:
            err(
                DagCheck.STORAGE,
                f"after step {step}: {len(remaining)} droplets on chip > storage capacity "
                f"{chip.storage_capacity}",
            )
