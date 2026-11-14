"""Operation graph: what the chip runs for one mixing DAG (SPEC 14.3).

`build_opgraph(dag, instance, platform, assay)` checks the DAG with `validate`, then makes:

- one `DISPENSE` per edge out of a RESERVOIR node (one droplet of that volume at a port of that
  fluid). It feeds exactly one consumer and ends exactly when that consumer starts (just in
  time), so a dispensed droplet is never stored;
- one `MIX` per MIX node;
- one `DETECT` per OUTPUT node if `assay.detect`, else a `DELIVER` (no resource, 0 ticks);
- one `DISCARD` per WASTE node (no resource, 0 ticks). It starts once its droplet's producer
  has ended and every other use of the droplet has started.

Partial takes are not operations: a consumer takes its volume when it starts. A droplet (the
product of a MIX or DISPENSE) exists from its producer's end until its last consumer starts;
in between it is stored (14.3). Ops are numbered in the DAG's `(step, id)` order, each
consumer's dispenses just before it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from meda.core.composition import Composition
from meda.core.dag import MixingDAG, NodeType, validate
from meda.core.problem import Instance
from meda.pipeline.platform import Platform


class OpKind(Enum):
    DISPENSE = "DISPENSE"
    MIX = "MIX"
    DETECT = "DETECT"
    DELIVER = "DELIVER"
    DISCARD = "DISCARD"


@dataclass(frozen=True, slots=True)
class Op:
    """One operation. `after_end`: ops that must end before this starts; `after_start`: ops
    that must have started (a DISCARD waits for the droplet's other uses). `feeds`: for a
    DISPENSE, the op it feeds (it ends exactly when that op starts). `fluid` for a DISPENSE,
    `composition` for a DETECT (detector changeover). `node_id`/`step` tie it to the DAG."""

    id: int
    kind: OpKind
    duration: int
    node_id: int
    step: int
    after_end: tuple[int, ...] = ()
    after_start: tuple[int, ...] = ()
    feeds: int | None = None
    fluid: int | None = None
    composition: Composition | None = None


@dataclass(frozen=True, slots=True)
class Droplet:
    """A droplet made by `producer` and used (whole or in parts) by `consumers`. `volume` is
    its volume and `takes[i]` the units `consumers[i]` takes (14.3, from M11; they sum to
    `volume`). Zero and empty when unknown (hand-made graphs)."""

    producer: int
    consumers: tuple[int, ...]
    volume: int = 0
    takes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class OpGraph:
    """The operations of one plan. `ops[i].id == i`."""

    instance_id: str
    ops: tuple[Op, ...]
    droplets: tuple[Droplet, ...]
    storage_capacity: int

    def dispenses(self, op_id: int) -> tuple[Op, ...]:
        """The DISPENSE ops feeding `op_id`."""
        return tuple(o for o in self.ops if o.feeds == op_id)


@dataclass(frozen=True, slots=True)
class AssaySpec:
    """What happens to each delivered droplet. `detect`: detected once, then discarded."""

    detect: bool = True


def build_opgraph(
    dag: MixingDAG,
    instance: Instance,
    platform: Platform,
    assay: AssaySpec | None = None,
) -> OpGraph:
    """The operations for a valid complete DAG; raises `DagValidationError` otherwise.
    `assay` defaults to `AssaySpec()` (every delivered droplet detected)."""
    assay = assay or AssaySpec()
    validate(dag, instance).raise_if_invalid()
    nodes = {n.id: n for n in dag.nodes}
    children: dict[int, list[tuple[int, int]]] = defaultdict(list)  # src -> (dst, volume)
    parents: dict[int, list[int]] = defaultdict(list)
    for e in dag.edges:
        children[e.src].append((e.dst, e.volume))
        parents[e.dst].append(e.src)

    order = sorted(
        (n for n in dag.nodes if n.type is not NodeType.RESERVOIR), key=lambda n: (n.step, n.id)
    )
    kind_of = {
        NodeType.MIX: (OpKind.MIX, platform.mix_ticks),
        NodeType.OUTPUT: (
            (OpKind.DETECT, platform.detect_ticks) if assay.detect else (OpKind.DELIVER, 0)
        ),
        NodeType.WASTE: (OpKind.DISCARD, 0),
    }
    # number every op first: each node's dispenses, then the node itself
    op_of: dict[int, int] = {}
    dispense_ids: dict[tuple[int, int], int] = {}  # (reservoir, consumer node) -> op id
    next_id = 0
    for n in order:
        for p in sorted(parents[n.id]):
            if nodes[p].type is NodeType.RESERVOIR:
                dispense_ids[(p, n.id)] = next_id
                next_id += 1
        op_of[n.id] = next_id
        next_id += 1

    def consumer_ops(node_id: int) -> list[int]:
        return sorted(op_of[c] for c, _ in children[node_id])

    def takes(node_id: int) -> tuple[int, ...]:
        by_op: dict[int, int] = defaultdict(int)
        for c, v in children[node_id]:
            by_op[op_of[c]] += v
        return tuple(by_op[o] for o in sorted(by_op))

    ops: list[Op] = []
    droplets: list[Droplet] = []
    for n in order:
        for p in sorted(parents[n.id]):
            if nodes[p].type is NodeType.RESERVOIR:
                did = dispense_ids[(p, n.id)]
                ops.append(
                    Op(
                        did,
                        OpKind.DISPENSE,
                        platform.dispense_ticks,
                        p,
                        n.step,
                        feeds=op_of[n.id],
                        fluid=nodes[p].fluid_id,
                    )
                )
                vol = sum(v for c, v in children[p] if c == n.id)
                droplets.append(Droplet(did, (op_of[n.id],), vol, (vol,)))
        kind, duration = kind_of[n.type]
        oid = op_of[n.id]
        after_end = tuple(sorted(op_of[p] for p in parents[n.id] if p in op_of))
        after_start: tuple[int, ...] = ()
        if kind is OpKind.DISCARD:
            (p,) = parents[n.id]
            if nodes[p].type is NodeType.RESERVOIR:
                raise ValueError(
                    f"{dag.instance_id}: WASTE node {n.id} discards straight from a reservoir"
                )
            after_start = tuple(c for c in consumer_ops(p) if c != oid)
        ops.append(
            Op(
                oid,
                kind,
                duration,
                n.id,
                n.step,
                after_end=after_end,
                after_start=after_start,
                composition=n.composition if kind is OpKind.DETECT else None,
            )
        )
        if n.type is NodeType.MIX:
            droplets.append(
                Droplet(oid, tuple(sorted(set(consumer_ops(n.id)))), n.volume, takes(n.id))
            )
    ops.sort(key=lambda o: o.id)
    cap = platform.storage_capacity or instance.chip.storage_capacity
    return OpGraph(dag.instance_id, tuple(ops), tuple(droplets), cap)
