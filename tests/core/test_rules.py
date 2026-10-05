"""Legality, transitions and the hand-checked plans of SPEC 17.3 (replayed; search is M5)."""

import re
from typing import Any

import pytest

from meda.core.actions import Action, Discard, Mix, Output, Source
from meda.core.composition import Composition, Rational
from meda.core.dag import NodeType
from meda.core.problem import Instance
from meda.core.rules import (
    IllegalActionError,
    TerminalStatus,
    apply,
    legal_actions,
    replay,
    terminal_status,
)
from meda.core.state import Droplet, PrepState
from tests.core.helpers import chip, default_weights, dilution, mixture

S, B = Source.reservoir(0), Source.reservoir(1)


def D(uid: int) -> Source:
    return Source.droplet(uid)


# --------------------------------------------------------------------------- SPEC 17.3


def test_case1_quarter_m2() -> None:
    inst = dilution([(1, 4)], chip_limits=chip("dmfb_m2"))
    plan: list[Action] = [Mix(S, 1, B, 1), Mix(D(2), 1, B, 1), Output(D(3), 0)]
    state, cost, deltas = replay(inst, plan)
    assert terminal_status(state, inst) is TerminalStatus.SUCCESS
    assert cost.reagent_by_fluid == (1, 2)
    assert cost.mix_ops == 2
    assert cost.waste_volume == 2  # 1 unit of 1/2 and 1 unit of 1/4
    assert deltas[1].nodes[0].composition == Composition.from_ints((1, 3), 4)
    assert default_weights().scalar(cost, inst.fluids) == Rational.of(15)


def test_case2_quarter_m4() -> None:
    inst = dilution([(1, 4)], chip_limits=chip(max_mix_volume=4))
    state, cost, _ = replay(inst, [Mix(S, 1, B, 3), Output(D(2), 0)])
    assert state.is_success
    assert cost.reagent_by_fluid == (1, 3)
    assert cost.mix_ops == 1
    assert cost.waste_volume == 3
    assert default_weights().scalar(cost, inst.fluids) == Rational.of(31, 2)


def test_case3_three_tenths_m8() -> None:
    inst = dilution([(3, 10)])
    state, cost, deltas = replay(inst, [Mix(S, 2, B, 3), Mix(D(2), 3, B, 1), Output(D(3), 0)])
    assert state.is_success
    assert deltas[0].nodes[0].composition == Composition.from_ints((2, 3), 5)
    assert deltas[1].nodes[0].composition == Composition.from_ints((3, 7), 10)
    assert cost.mix_ops == 2
    assert cost.waste_volume == 2 + 3


def test_case3_not_reachable_in_one_mix() -> None:
    inst = dilution([(3, 10)])
    s0 = PrepState.initial(inst)
    for act in legal_actions(s0, inst):
        s1, _, _ = apply(s0, act, inst)
        assert not any(isinstance(a, Output) for a in legal_actions(s1, inst))


def test_case4_sharing_m2() -> None:
    inst = dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2"))
    plan: list[Action] = [
        Mix(S, 1, B, 1),  # uid 2: 1/2, volume 2
        Mix(D(2), 1, S, 1),  # uid 3: 3/4
        Mix(D(2), 1, B, 1),  # uid 4: 1/4 (uid 2 fully consumed)
        Output(D(3), 0),
        Output(D(4), 1),
    ]
    state, cost, _ = replay(inst, plan)
    assert state.is_success
    assert cost.mix_ops == 3
    assert cost.reagent_by_fluid == (2, 2)
    assert cost.waste_volume == 2


# --------------------------------------------------------------------------- enumeration


def test_initial_mix_counts() -> None:
    # k, l >= 1 with k + l <= 8: 28 ratios (PROJECT_CONTEXT section 11).
    m8 = dilution([(1, 4)])
    assert len(legal_actions(PrepState.initial(m8), m8)) == 28
    m2 = dilution([(1, 4)], chip_limits=chip("dmfb_m2"))
    assert legal_actions(PrepState.initial(m2), m2) == (Mix(S, 1, B, 1),)


def test_canonical_order_and_no_duplicates() -> None:
    inst = dilution([(1, 2), (1, 4)], droplet_volume=1)
    state, _, _ = replay(inst, [Mix(S, 2, B, 2), Mix(S, 1, B, 1)])  # two 1/2 droplets, vol 4, 2
    acts = legal_actions(state, inst)
    kinds = [type(a).__name__ for a in acts]
    assert kinds == sorted(kinds, key=["Output", "Mix", "Discard"].index)
    keys = [a.canonical_key(state) for a in acts]
    assert len(set(keys)) == len(keys)


def test_identical_droplets_enumerated_once() -> None:
    inst = dilution([(1, 4)], chip_limits=chip("dmfb_m2"))
    half = Composition.from_ints((1, 1), 2)
    state = PrepState.make([Droplet(2, half, 2, 2), Droplet(3, half, 2, 3)], (1,), 2, 4)
    acts = legal_actions(state, inst)
    assert sum(isinstance(a, Discard) for a in acts) == 1
    mixes = [a for a in acts if isinstance(a, Mix)]
    # S+B, 1/2 with S, 1/2 with B. The twin (uid 3) is never used; no 1/2 + 1/2 merge.
    assert len(mixes) == 3
    assert D(3) not in {m.a for m in mixes} | {m.b for m in mixes}
    # The equivalent action on the twin is still legal and gives the same canonical state.
    s_a, _, _ = apply(state, Mix(D(2), 1, B, 1), inst)
    s_b, _, _ = apply(state, Mix(D(3), 1, B, 1), inst)
    assert s_a.canonical_key() == s_b.canonical_key()


def test_mix_symmetry_same_result() -> None:
    inst = dilution([(1, 4)])
    s0 = PrepState.initial(inst)
    a, ca, _ = apply(s0, Mix(S, 1, B, 3), inst)
    b, cb, _ = apply(s0, Mix(B, 3, S, 1), inst)
    assert a == b
    assert ca == cb
    assert Mix(S, 1, B, 3).canonical_key(s0) == Mix(B, 3, S, 1).canonical_key(s0)
    legal_keys = [a.canonical_key(s0) for a in legal_actions(s0, inst)]
    assert legal_keys.count(Mix(B, 3, S, 1).canonical_key(s0)) == 1


def test_output_from_reservoir_for_pure_target() -> None:
    inst = mixture([(1, 0, 0)])
    acts = legal_actions(PrepState.initial(inst), inst)
    assert acts[0] == Output(Source.reservoir(0), 0)
    state, cost, deltas = apply(PrepState.initial(inst), acts[0], inst)
    assert state.is_success
    assert cost.reagent_by_fluid == (1, 0, 0)
    assert deltas.nodes[0].type is NodeType.OUTPUT


def test_tolerance_allows_near_output() -> None:
    inst = dilution([(1, 3)], tolerance=Rational.of(1, 20))
    state, _, _ = apply(PrepState.initial(inst), Mix(S, 3, B, 5), inst)  # 3/8: 1/24 away
    assert Output(D(2), 0) in legal_actions(state, inst)
    tight = dilution([(1, 3)], tolerance=Rational.of(1, 25))
    state, _, _ = apply(PrepState.initial(tight), Mix(S, 3, B, 5), tight)
    assert Output(D(2), 0) not in legal_actions(state, tight)


def test_success_wastes_leftovers_and_closes() -> None:
    inst = dilution([(1, 2)])
    state, cost, delta = replay(inst, [Mix(S, 3, B, 3), Output(D(2), 0)])
    assert state.droplets == ()
    assert cost.waste_volume == 5
    assert legal_actions(state, inst) == ()
    last = delta[-1]
    assert [n.type for n in last.nodes] == [NodeType.OUTPUT, NodeType.WASTE]


def test_step_limit_truncates() -> None:
    inst = dilution([(1, 4)], step_limit=1)
    state, _, _ = apply(PrepState.initial(inst), Mix(S, 1, B, 1), inst)
    assert terminal_status(state, inst) is TerminalStatus.TRUNCATED
    assert legal_actions(state, inst) == ()


def test_dead_end() -> None:
    # max_denominator 1 forbids every mix of two different fluids; target is not pure.
    inst = dilution([(1, 2)], chip_limits=chip(max_denominator=1))
    s0 = PrepState.initial(inst)
    assert legal_actions(s0, inst) == ()
    assert terminal_status(s0, inst) is TerminalStatus.DEAD_END


# --------------------------------------------------------------------------- illegal


def _after(prefix: list[Action], **chip_overrides: Any) -> tuple[PrepState, Instance]:
    inst = dilution([(1, 4), (1, 2)], chip_limits=chip(**chip_overrides))
    state, _, _ = replay(inst, prefix)
    return state, inst


# (name, actions before, illegal action, chip overrides, expected reason in the error)
ILLEGAL: list[tuple[str, list[Action], Action, dict[str, Any], str]] = [
    ("M exceeded", [], Mix(S, 5, B, 4), {}, "mix volume 9 outside"),
    ("zero volume", [], Mix(S, 0, B, 2), {}, "must be >= 1"),
    ("self mix", [], Mix(S, 1, S, 1), {}, "with itself"),
    ("take more than volume", [Mix(S, 1, B, 1)], Mix(D(2), 3, B, 1), {}, "takes 3 units"),
    (
        "partial take off",
        [Mix(S, 2, B, 2)],
        Mix(D(2), 1, B, 1),
        {"allow_partial_take": False},
        "partial take",
    ),
    (
        "remainder too small",
        [Mix(S, 1, B, 2)],
        Mix(D(2), 2, B, 1),
        {"min_droplet_volume": 2},
        "below min_droplet_volume",
    ),
    (
        "storage exceeded",
        [Mix(S, 1, B, 1)],
        Mix(S, 1, B, 3),
        {"storage_capacity": 1},
        "storage after mix 2 > 1",
    ),
    ("denominator exceeded", [], Mix(S, 1, B, 4), {"max_denominator": 4}, "denominator 5"),
    (
        "same composition merge",
        [Mix(S, 1, B, 1), Mix(S, 2, B, 2)],
        Mix(D(2), 1, D(3), 1),
        {},
        "equal composition",
    ),
    ("output out of tolerance", [Mix(S, 1, B, 2)], Output(D(2), 0), {}, "from target 0"),
    ("output target missing", [Mix(S, 1, B, 3)], Output(D(2), 5), {}, "no target 5"),
    ("no such droplet", [], Discard(7), {}, "no droplet with uid 7"),
    ("no such reservoir", [], Mix(Source.reservoir(9), 1, B, 1), {}, "no reservoir 9"),
    (
        "output demand met",
        [Mix(S, 2, B, 2), Output(D(2), 1)],
        Output(D(2), 1),
        {},
        "no remaining demand",
    ),
]


@pytest.mark.parametrize(
    ("_name", "prefix", "action", "overrides", "reason"), ILLEGAL, ids=[c[0] for c in ILLEGAL]
)
def test_illegal_actions_raise(
    _name: str, prefix: list[Action], action: Action, overrides: dict[str, Any], reason: str
) -> None:
    state, inst = _after(prefix, **overrides)
    assert action not in legal_actions(state, inst)
    with pytest.raises(IllegalActionError, match="test-dilution") as err:
        apply(state, action, inst)
    assert re.search(reason, str(err.value)), str(err.value)


def test_same_composition_merge_allowed_by_flag() -> None:
    state, inst = _after([Mix(S, 1, B, 1), Mix(S, 2, B, 2)], allow_same_composition_merge=True)
    nxt, _, _ = apply(state, Mix(D(2), 1, D(3), 1), inst)
    assert Composition.from_ints((1, 1), 2) in {d.composition for d in nxt.droplets}


def test_actions_after_success_raise() -> None:
    inst = dilution([(1, 2)])
    state, _, _ = replay(inst, [Mix(S, 1, B, 1), Output(D(2), 0)])
    with pytest.raises(IllegalActionError, match="already met"):
        apply(state, Mix(S, 1, B, 1), inst)


def test_output_needs_full_droplet_volume() -> None:
    inst = dilution([(1, 4)], demand=2, droplet_volume=3)
    state, _, _ = replay(inst, [Mix(S, 1, B, 3), Output(D(2), 0)])  # 1/4: 4 units -> 1 left
    assert Output(D(2), 0) not in legal_actions(state, inst)
    with pytest.raises(IllegalActionError, match="takes 3 units from a droplet of volume 1"):
        apply(state, Output(D(2), 0), inst)
