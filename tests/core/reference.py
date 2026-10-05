"""Reference (slow, obviously correct) legal-action enumeration for cross-checking."""

from __future__ import annotations

from meda.core.actions import Action, Discard, Mix, Output, Source
from meda.core.composition import Composition
from meda.core.problem import Instance
from meda.core.rules import (
    _actions_closed,
    _mix_violation,
    _output_violation,
    _resolve,
    _Src,
)
from meda.core.state import CanonicalKey, PrepState


def _sorted_by_key(items: list[tuple[CanonicalKey, Action]]) -> list[Action]:
    items.sort(key=lambda kv: kv[0])  # type: ignore[arg-type,return-value]
    return [a for _, a in items]


def reference_legal_actions(state: PrepState, instance: Instance) -> tuple[Action, ...]:
    """Straightforward enumeration with the rule predicates, then sorting by canonical key.

    Slow but obviously correct; the fast `rules.legal_actions` must return exactly this."""
    if _actions_closed(state, instance):
        return ()
    chip = instance.chip
    n = len(state.droplets)

    # One representative per distinct (composition, volume); droplets are sorted, so the first
    # seen has the smallest uid. `twin` holds a second droplet with the same key, if any.
    reps: list[_Src] = [
        _resolve(Source.reservoir(f), state, instance) for f in range(instance.num_fluids)
    ]
    twin: dict[int, _Src] = {}
    last: tuple[Composition, int] | None = None
    for d in state.droplets:
        src = _Src(Source.droplet(d.uid), d.composition, d.volume, d.dag_node)
        if (d.composition, d.volume) == last:
            twin.setdefault(len(reps) - 1, src)
            continue
        last = (d.composition, d.volume)
        reps.append(src)

    outputs: list[tuple[CanonicalKey, Action]] = []
    for src in reps:
        for t in range(len(instance.targets)):
            if _output_violation(src, t, state, instance) is None:
                out = Output(src.source, t)
                outputs.append((out.canonical_key(state), out))

    mixes: list[tuple[CanonicalKey, Action]] = []
    m = chip.max_mix_volume
    for i, a in enumerate(reps):
        partners = [(b, False) for b in reps[i + 1 :]]
        if i in twin:
            partners.append((twin[i], True))
        for b, same_key in partners:
            for k in range(1, (m - 1 if a.volume is None else min(m - 1, a.volume)) + 1):
                for l in range(1, (m - k if b.volume is None else min(m - k, b.volume)) + 1):
                    if same_key and k > l:
                        continue  # Mix(d1, k, d2, l) == Mix(d2, l, d1, k) for identical droplets
                    reason, _ = _mix_violation(a, k, b, l, n, chip)
                    if reason is None:
                        mx = Mix(a.source, k, b.source, l)
                        mixes.append((mx.canonical_key(state), mx))

    discards: list[tuple[CanonicalKey, Action]] = []
    for src in reps[instance.num_fluids :]:
        dc = Discard(src.source.index)
        discards.append((dc.canonical_key(state), dc))

    return (*_sorted_by_key(outputs), *_sorted_by_key(mixes), *_sorted_by_key(discards))
