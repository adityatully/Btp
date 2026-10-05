"""Property tests over random legal rollouts (SPEC 17.1)."""

from __future__ import annotations

import dataclasses
from enum import Enum

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meda.core.actions import Mix
from meda.core.composition import Composition
from meda.core.costs import StepCost
from meda.core.dag import NodeType
from meda.core.problem import Instance
from meda.core.rules import apply, legal_actions, terminal_status
from meda.core.state import Droplet, PrepState
from tests.core.helpers import FINISHING_INSTANCES, ROLLOUT_INSTANCES, random_rollout
from tests.core.reference import reference_legal_actions

seeds = st.integers(0, 2**32 - 1)
instances = pytest.mark.parametrize("name", sorted(ROLLOUT_INSTANCES))
rollout_settings = settings(max_examples=25, deadline=None)


@instances
@rollout_settings
@given(seed=seeds)
def test_volume_conservation_and_exact_sums(name: str, seed: int) -> None:
    inst = ROLLOUT_INSTANCES[name]
    taken = output = wasted = 0
    for _, _, nxt, cost, delta in random_rollout(inst, seed):
        taken += cost.total_reagent
        wasted += cost.waste_volume
        output += sum(n.volume for n in delta.nodes if n.type is NodeType.OUTPUT)
        on_chip = sum(d.volume for d in nxt.droplets)
        assert taken == on_chip + output + wasted
        assert len(nxt.droplets) <= inst.chip.storage_capacity
        for d in nxt.droplets:
            assert sum(d.composition.num) == d.composition.den
            assert d.composition.den <= inst.chip.max_denominator
            assert d.volume >= inst.chip.min_droplet_volume


@instances
@rollout_settings
@given(seed=seeds)
def test_every_legal_action_applies(name: str, seed: int) -> None:
    inst = ROLLOUT_INSTANCES[name]
    for state, *_ in random_rollout(inst, seed)[:4]:
        for act in legal_actions(state, inst):
            apply(state, act, inst)


@instances
@rollout_settings
@given(seed=seeds)
def test_determinism(name: str, seed: int) -> None:
    inst = ROLLOUT_INSTANCES[name]
    assert random_rollout(inst, seed) == random_rollout(inst, seed)
    # Re-applying the same action sequence reproduces states, costs and DAG deltas.
    run = random_rollout(inst, seed)
    state = PrepState.initial(inst)
    for prev, act, nxt, cost, delta in run:
        assert state == prev
        state, c, d = apply(state, act, inst)
        assert (state, c, d) == (nxt, cost, delta)


@instances
@rollout_settings
@given(seed=seeds, data=st.data())
def test_canonical_key_ignores_uids(name: str, seed: int, data: st.DataObject) -> None:
    inst = ROLLOUT_INSTANCES[name]
    for state, *_ in random_rollout(inst, seed):
        uids = [d.uid for d in state.droplets]
        perm = data.draw(st.permutations(uids))
        relabel = dict(zip(uids, perm, strict=True))
        other = PrepState.make(
            [Droplet(relabel[d.uid], d.composition, d.volume, d.dag_node) for d in state.droplets],
            state.remaining_demand,
            state.step,
            state.next_uid,
        )
        assert other.canonical_key() == state.canonical_key()
        assert [a.canonical_key(other) for a in legal_actions(other, inst)] == [
            a.canonical_key(state) for a in legal_actions(state, inst)
        ]


@instances
@rollout_settings
@given(seed=seeds)
def test_episodes_end_terminal_with_unique_node_ids(name: str, seed: int) -> None:
    inst = ROLLOUT_INSTANCES[name]
    run = random_rollout(inst, seed)
    final = run[-1][2] if run else PrepState.initial(inst)
    assert terminal_status(final, inst) is not None
    ids = [n.id for *_, delta in run for n in delta.nodes]
    assert ids == list(range(inst.num_fluids, final.next_uid))


def _assert_no_float(obj: object, path: str) -> None:
    assert not isinstance(obj, float), f"float at {path}"
    if isinstance(obj, Enum):
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _assert_no_float(getattr(obj, f.name), f"{path}.{f.name}")
    elif isinstance(obj, tuple):
        for i, x in enumerate(obj):
            _assert_no_float(x, f"{path}[{i}]")


@instances
@settings(max_examples=20, deadline=None)
@given(seed=seeds)
def test_no_floats_in_state_or_cost(name: str, seed: int) -> None:
    inst: Instance = ROLLOUT_INSTANCES[name]
    for state, _, nxt, cost, delta in random_rollout(inst, seed):
        _assert_no_float(state, "state")
        _assert_no_float(nxt, "next_state")
        _assert_no_float(cost, "cost")
        _assert_no_float(delta, "delta")
    assert isinstance(StepCost.zero(2).mix_ops, int)
    assert isinstance(Composition.pure(0, 2).den, int)


@pytest.mark.parametrize("name", sorted(ROLLOUT_INSTANCES) + sorted(FINISHING_INSTANCES))
@settings(max_examples=25, deadline=None)
@given(seed=seeds)
def test_fast_enumeration_matches_reference(name: str, seed: int) -> None:
    inst = {**ROLLOUT_INSTANCES, **FINISHING_INSTANCES}[name]
    for state, *_ in random_rollout(inst, seed, prefer_outputs=name in FINISHING_INSTANCES):
        fast = legal_actions(state, inst)
        ref = reference_legal_actions(state, inst)
        assert [a.canonical_key(state) for a in fast] == [a.canonical_key(state) for a in ref]
        for a in fast:  # Mix sources are written in canonical order (SPEC 5.6)
            if isinstance(a, Mix):
                key_a, key_b = a.a.canonical_key(state), a.b.canonical_key(state)
                assert key_a < key_b or (key_a == key_b and a.k <= a.l)  # type: ignore[operator]
