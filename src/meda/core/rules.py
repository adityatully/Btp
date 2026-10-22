"""`legal_actions`, `apply` (pure) and terminal status (SPEC 5.6-5.8).

Legality is defined once, by the `_*_violation` predicates. `legal_actions` enumerates with
them and `apply` checks with them, so the two can never disagree.

Equivalent actions (same canonical key, e.g. the same Mix on two droplets with equal
composition and volume) are enumerated once, using the droplet with the smallest uid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import gcd

from meda.core.actions import Action, Discard, Mix, Output, Source
from meda.core.chip import ChipLimits
from meda.core.composition import Composition, linf_distance, mix
from meda.core.costs import StepCost
from meda.core.dag import DagDelta, DagEdge, DagNode, NodeType
from meda.core.problem import Instance
from meda.core.state import Droplet, PrepState


class IllegalActionError(ValueError):
    """Raised by `apply` for an action that `legal_actions` would not return."""


class TerminalStatus(Enum):
    SUCCESS = "success"
    TRUNCATED = "truncated"  # step limit reached without success
    DEAD_END = "dead_end"  # no legal action and not success


@dataclass(frozen=True, slots=True)
class _Src:
    """A resolved source: composition, volume (None = reservoir, unlimited) and DAG node."""

    source: Source
    composition: Composition
    volume: int | None
    node: int


# --------------------------------------------------------------------------- predicates


def _take_violation(volume: int | None, take: int, chip: ChipLimits) -> str | None:
    """Can `take` units be taken from a source of `volume` units (None = reservoir)?"""
    if volume is None:
        return None
    if take > volume:
        return f"takes {take} units from a droplet of volume {volume}"
    if take < volume:
        if not chip.allow_partial_take:
            return f"partial take ({take} of {volume}) not allowed on this chip"
        if volume - take < chip.min_droplet_volume:
            return f"remainder {volume - take} below min_droplet_volume {chip.min_droplet_volume}"
    return None


def _mix_violation(
    a: _Src,
    k: int,
    b: _Src,
    l: int,
    n_droplets: int,
    chip: ChipLimits,
) -> tuple[str | None, Composition | None]:
    """Reason the mix is illegal (or None), and its result composition when computed."""
    if a.source == b.source:
        return "mixes a source with itself", None
    if k < 1 or l < 1:
        return f"mix volumes must be >= 1, got {k}:{l}", None
    if not chip.min_mix_volume <= k + l <= chip.max_mix_volume:
        return (
            f"mix volume {k + l} outside [{chip.min_mix_volume}, {chip.max_mix_volume}]",
            None,
        )
    for src, take in ((a, k), (b, l)):
        reason = _take_violation(src.volume, take, chip)
        if reason:
            return reason, None
    if a.composition == b.composition and not chip.allow_same_composition_merge:
        return "merges two droplets of equal composition (allow_same_composition_merge off)", None
    consumed = (a.volume == k) + (b.volume == l)
    if n_droplets - consumed + 1 > chip.storage_capacity:
        return f"storage after mix {n_droplets - consumed + 1} > {chip.storage_capacity}", None
    result = mix(a.composition, k, b.composition, l)
    if result.den > chip.max_denominator:
        return f"result denominator {result.den} > max_denominator {chip.max_denominator}", None
    return None, result


def _output_violation(src: _Src, target: int, state: PrepState, instance: Instance) -> str | None:
    if not 0 <= target < len(instance.targets):
        return f"no target {target}"
    if state.remaining_demand[target] == 0:
        return f"target {target} has no remaining demand"
    t = instance.targets[target]
    reason = _take_violation(src.volume, t.droplet_volume, instance.chip)
    if reason:
        return reason
    dist = linf_distance(src.composition, t.composition)
    if not dist <= t.tolerance:
        return f"composition {src.composition} is {dist} from target {target} (tol {t.tolerance})"
    return None


# --------------------------------------------------------------------------- helpers


def _resolve(source: Source, state: PrepState, instance: Instance) -> _Src:
    if source.is_reservoir:
        f = source.index
        if not 0 <= f < instance.num_fluids:
            raise IllegalActionError(f"instance '{instance.instance_id}': no reservoir {f}")
        return _Src(source, Composition.pure(f, instance.num_fluids), None, f)
    if not state.has_droplet(source.index):
        raise IllegalActionError(
            f"instance '{instance.instance_id}': no droplet with uid {source.index} on chip"
        )
    d = state.droplet(source.index)
    return _Src(source, d.composition, d.volume, d.dag_node)


def _actions_closed(state: PrepState, instance: Instance) -> bool:
    return state.is_success or state.step >= instance.step_limit


# --------------------------------------------------------------------------- public API


def _take_amounts(volume: int | None, upper: int, chip: ChipLimits) -> list[int]:
    """Amounts 1..upper that may be taken from a source of `volume` units (None = reservoir).

    Same rule as `_take_violation`."""
    if volume is None:
        return list(range(1, upper + 1))
    out = []
    for k in range(1, min(upper, volume) + 1):
        if k == volume or (chip.allow_partial_take and volume - k >= chip.min_droplet_volume):
            out.append(k)
    return out


def _representatives(state: PrepState, instance: Instance) -> tuple[list[_Src], dict[int, _Src]]:
    """One source per distinct canonical source key, in canonical key order.

    Droplet keys `("D", composition, volume)` sort before reservoir keys `("R", f)`. Droplets are
    sorted by (composition, volume, uid), so the first of each group has the smallest uid.
    `twin[i]` is a second droplet with the same key as `reps[i]`, if one exists."""
    reps: list[_Src] = []
    twin: dict[int, _Src] = {}
    last: tuple[Composition, int] | None = None
    for d in state.droplets:
        src = _Src(Source.droplet(d.uid), d.composition, d.volume, d.dag_node)
        if (d.composition, d.volume) == last:
            twin.setdefault(len(reps) - 1, src)
            continue
        last = (d.composition, d.volume)
        reps.append(src)
    for f in range(instance.num_fluids):
        reps.append(_Src(Source.reservoir(f), Composition.pure(f, instance.num_fluids), None, f))
    return reps, twin


def legal_actions(state: PrepState, instance: Instance) -> tuple[Action, ...]:
    """All legal actions in canonical order: Outputs, then Mixes, then Discards, each sorted by
    canonical key. Empty once the state is a success or the step limit is reached.

    Fast path of the `_*_violation` rules (tests check it against a plain enumeration). Actions
    are generated directly in canonical order: sources are visited in canonical key order, and a
    Mix key is `((key_a, k), (key_b, l))` with `key_a < key_b`, so the loop order (a, k, b, l)
    is the sorted order. For two identical droplets (same key) only k <= l is generated."""
    if _actions_closed(state, instance):
        return ()
    chip = instance.chip
    n = len(state.droplets)
    reps, twin = _representatives(state, instance)

    actions: list[Action] = []
    open_targets = [t for t in instance.targets if state.remaining_demand[t.id] > 0]
    for src in reps:
        for t in open_targets:
            if (
                _take_violation(src.volume, t.droplet_volume, chip) is None
                and linf_distance(src.composition, t.composition) <= t.tolerance
            ):
                actions.append(Output(src.source, t.id))

    m = chip.max_mix_volume
    lo_vol = chip.min_mix_volume
    cap = chip.storage_capacity
    max_den = chip.max_denominator
    merge = chip.allow_same_composition_merge
    takes = [_take_amounts(r.volume, m - 1, chip) for r in reps]
    for i, a in enumerate(reps):
        partners: list[tuple[_Src, list[int], bool]] = []
        if i in twin:
            partners.append((twin[i], takes[i], True))
        partners += [(reps[j], takes[j], False) for j in range(i + 1, len(reps))]
        prepared = []
        for b, b_takes, same_key in partners:
            if a.composition == b.composition and not merge:
                continue
            da, db = a.composition.den, b.composition.den
            pa = [x * db for x in a.composition.num]
            pb = [y * da for y in b.composition.num]
            prepared.append((b, b_takes, same_key, pa, pb, da * db))
        for k in takes[i]:
            a_used = a.volume == k
            for b, b_takes, same_key, pa, pb, dd in prepared:
                for l in b_takes:
                    if k + l > m:
                        break
                    if k + l < lo_vol or (same_key and l < k):
                        continue
                    if n - a_used - (b.volume == l) + 1 > cap:
                        continue
                    den = (k + l) * dd
                    # The reduced denominator divides `den`, so the gcd is only needed above it.
                    if den > max_den:
                        g = gcd(*[k * x + l * y for x, y in zip(pa, pb, strict=True)], den)
                        if den // g > max_den:
                            continue
                    actions.append(Mix(a.source, k, b.source, l))

    for src in reps:
        if not src.source.is_reservoir:
            actions.append(Discard(src.source.index))
    return tuple(actions)


def check_legal(state: PrepState, action: Action, instance: Instance) -> None:
    """Raise `IllegalActionError` naming the instance and the violated rule if illegal."""

    def fail(reason: str) -> IllegalActionError:
        return IllegalActionError(
            f"instance '{instance.instance_id}' step {state.step}: {action} is illegal: {reason}"
        )

    if state.is_success:
        raise fail("all demand is already met")
    if state.step >= instance.step_limit:
        raise fail(f"step limit {instance.step_limit} reached")
    if isinstance(action, Mix):
        a = _resolve(action.a, state, instance)
        b = _resolve(action.b, state, instance)
        reason, _ = _mix_violation(a, action.k, b, action.l, len(state.droplets), instance.chip)
    elif isinstance(action, Output):
        reason = _output_violation(
            _resolve(action.src, state, instance), action.target, state, instance
        )
    elif isinstance(action, Discard):
        reason = None if state.has_droplet(action.uid) else f"no droplet with uid {action.uid}"
    else:
        reason = f"unknown action type {type(action).__name__}"
    if reason is not None:
        raise fail(reason)


def apply(
    state: PrepState, action: Action, instance: Instance
) -> tuple[PrepState, StepCost, DagDelta]:
    """Pure transition. Returns the next state, this step's cost and the DAG nodes/edges added.

    On success (all demand met), every droplet left on chip goes to waste in this same step.
    """
    check_legal(state, action, instance)
    num_fluids = instance.num_fluids
    reagent = [0] * num_fluids
    on_chip: dict[int, Droplet] = {d.uid: d for d in state.droplets}
    demand = list(state.remaining_demand)
    nodes: list[DagNode] = []
    edges: list[DagEdge] = []
    next_id = state.next_uid
    step = state.step
    mix_ops = waste = outputs = 0

    def take(source: Source, units: int) -> tuple[Composition, int]:
        """Remove `units` from a source; return its composition and DAG node.

        This is the single place where a take is computed (phase-2 noise hook)."""
        if source.is_reservoir:
            reagent[source.index] += units
            return Composition.pure(source.index, num_fluids), source.index
        d = on_chip.pop(source.index)
        if d.volume > units:
            on_chip[d.uid] = Droplet(d.uid, d.composition, d.volume - units, d.dag_node)
        return d.composition, d.dag_node

    if isinstance(action, Mix):
        ca, na = take(action.a, action.k)
        cb, nb = take(action.b, action.l)
        comp = mix(ca, action.k, cb, action.l)
        vol = action.k + action.l
        nodes.append(DagNode(next_id, NodeType.MIX, comp, vol, step))
        edges += [DagEdge(na, next_id, action.k), DagEdge(nb, next_id, action.l)]
        on_chip[next_id] = Droplet(next_id, comp, vol, next_id)
        next_id += 1
        mix_ops = 1
    elif isinstance(action, Output):
        dv = instance.targets[action.target].droplet_volume
        comp, src_node = take(action.src, dv)
        nodes.append(DagNode(next_id, NodeType.OUTPUT, comp, dv, step, target_id=action.target))
        edges.append(DagEdge(src_node, next_id, dv))
        next_id += 1
        demand[action.target] -= 1
        outputs = 1
    else:
        d = on_chip.pop(action.uid)
        nodes.append(DagNode(next_id, NodeType.WASTE, d.composition, d.volume, step))
        edges.append(DagEdge(d.dag_node, next_id, d.volume))
        next_id += 1
        waste += d.volume

    if all(r == 0 for r in demand):
        for d in sorted(on_chip.values(), key=Droplet.sort_key):
            nodes.append(DagNode(next_id, NodeType.WASTE, d.composition, d.volume, step))
            edges.append(DagEdge(d.dag_node, next_id, d.volume))
            next_id += 1
            waste += d.volume
        on_chip.clear()

    next_state = PrepState.make(on_chip.values(), demand, step + 1, next_id)
    cost = StepCost(tuple(reagent), mix_ops, waste, outputs)
    return next_state, cost, DagDelta(tuple(nodes), tuple(edges))


def terminal_status(state: PrepState, instance: Instance) -> TerminalStatus | None:
    """SUCCESS, TRUNCATED, DEAD_END, or None if the episode continues."""
    if state.is_success:
        return TerminalStatus.SUCCESS
    if state.step >= instance.step_limit:
        return TerminalStatus.TRUNCATED
    if not legal_actions(state, instance):
        return TerminalStatus.DEAD_END
    return None


def is_terminal(state: PrepState, instance: Instance) -> bool:
    return terminal_status(state, instance) is not None


def replay(
    instance: Instance, actions: Sequence[Action]
) -> tuple[PrepState, StepCost, list[DagDelta]]:
    """Apply `actions` from the initial state; return final state, summed cost and deltas."""
    state = PrepState.initial(instance)
    total = StepCost.zero(instance.num_fluids)
    deltas: list[DagDelta] = []
    for act in actions:
        state, cost, delta = apply(state, act, instance)
        total = total + cost
        deltas.append(delta)
    return state, total, deltas
