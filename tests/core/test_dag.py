"""Mixing DAG: builder, validator, JSON I/O, networkx export (SPEC section 7, milestone M2)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from meda.core.actions import Action, Mix, Output, Source
from meda.core.composition import Composition
from meda.core.costs import StepCost
from meda.core.dag import (
    DagCheck,
    DagEdge,
    DagNode,
    DagValidationError,
    MixingDAG,
    NodeType,
    validate,
)
from meda.core.problem import Instance
from meda.core.rules import replay
from tests.core.helpers import (
    FINISHING_INSTANCES,
    ROLLOUT_INSTANCES,
    chip,
    dilution,
    random_rollout,
)

S, B = Source.reservoir(0), Source.reservoir(1)


def D(uid: int) -> Source:
    return Source.droplet(uid)


def plan_dag(instance: Instance, plan: list[Action]) -> MixingDAG:
    _, totals, deltas = replay(instance, plan)
    return MixingDAG.from_deltas(instance, deltas, totals, "hand")


def episode_dag(
    instance: Instance, seed: int, prefer_outputs: bool = False
) -> tuple[MixingDAG, bool]:
    """DAG of a random legal episode and whether the episode met all demand."""
    run = random_rollout(instance, seed, prefer_outputs)
    totals = StepCost.zero(instance.num_fluids)
    for *_, cost, _ in run:
        totals = totals + cost
    complete = bool(run) and run[-1][2].is_success
    return MixingDAG.from_deltas(instance, [t[4] for t in run], totals, "random"), complete


# Case 3 of SPEC 17.3 (3/10 at M = 8) and case 4 (sharing at M = 2).
CASE3 = dilution([(3, 10)])
CASE3_PLAN: list[Action] = [Mix(S, 2, B, 3), Mix(D(2), 3, B, 1), Output(D(3), 0)]
CASE4 = dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2"))
CASE4_PLAN: list[Action] = [
    Mix(S, 1, B, 1),
    Mix(D(2), 1, S, 1),
    Mix(D(2), 1, B, 1),
    Output(D(3), 0),
    Output(D(4), 1),
]


def checks(dag: MixingDAG, instance: Instance) -> set[DagCheck]:
    return validate(dag, instance).failed_checks


def replace_node(dag: MixingDAG, node_id: int, **changes: object) -> MixingDAG:
    return replace(
        dag, nodes=tuple(replace(n, **changes) if n.id == node_id else n for n in dag.nodes)
    )


# --------------------------------------------------------------------------- builder


def test_case3_dag_structure() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    assert validate(dag, CASE3).ok
    by_type = {t: [n for n in dag.nodes if n.type is t] for t in NodeType}
    assert [(n.id, n.volume) for n in by_type[NodeType.RESERVOIR]] == [(0, 2), (1, 4)]
    assert [n.composition for n in by_type[NodeType.MIX]] == [
        Composition.from_ints((2, 3), 5),
        Composition.from_ints((3, 7), 10),
    ]
    assert len(by_type[NodeType.OUTPUT]) == 1
    assert sorted(n.volume for n in by_type[NodeType.WASTE]) == [2, 3]
    assert DagEdge(2, 3, 3) in dag.edges  # 3 units of the 2/5 droplet feed the 3/10 mix


def test_case4_shares_intermediate() -> None:
    dag = plan_dag(CASE4, CASE4_PLAN)
    assert validate(dag, CASE4).ok
    half = next(n for n in dag.nodes if n.composition == Composition.from_ints((1, 1), 2))
    assert sorted(e.dst for e in dag.edges if e.src == half.id) == [3, 4]


# --------------------------------------------------------------------------- acceptance


@pytest.mark.parametrize("name", sorted(ROLLOUT_INSTANCES))
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_random_episodes_validate(name: str, seed: int) -> None:
    inst = ROLLOUT_INSTANCES[name]
    dag, complete = episode_dag(inst, seed)
    validate(dag, inst, complete=complete).raise_if_invalid()
    if not complete:
        # An unfinished plan fails the complete check only on demand / undelivered droplets.
        assert checks(dag, inst) <= {DagCheck.DEMAND, DagCheck.VOLUME_CONSERVATION}


@pytest.mark.parametrize("name", sorted(FINISHING_INSTANCES))
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_finished_random_episodes_validate_complete(name: str, seed: int) -> None:
    inst = FINISHING_INSTANCES[name]
    dag, complete = episode_dag(inst, seed, prefer_outputs=True)
    assume(complete)
    validate(dag, inst).raise_if_invalid()


@pytest.mark.parametrize("name", sorted(FINISHING_INSTANCES))
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), data=st.data())
def test_random_tampering_is_caught(name: str, seed: int, data: st.DataObject) -> None:
    inst = FINISHING_INSTANCES[name]
    dag, complete = episode_dag(inst, seed, prefer_outputs=True)
    assume(complete)
    mixes = [n for n in dag.nodes if n.type is NodeType.MIX]
    assume(mixes)
    target = data.draw(st.sampled_from(mixes))

    bigger = replace_node(dag, target.id, volume=target.volume + 1)
    assert DagCheck.VOLUME_CONSERVATION in validate(bigger, inst, complete).failed_checks

    edge_i = data.draw(st.integers(0, len(dag.edges) - 1))
    edges = list(dag.edges)
    edges[edge_i] = replace(edges[edge_i], volume=edges[edge_i].volume + 1)
    tampered = replace(dag, edges=tuple(edges))
    assert DagCheck.VOLUME_CONSERVATION in validate(tampered, inst, complete).failed_checks

    other = Composition.pure(0, inst.num_fluids)
    recoloured = replace_node(dag, target.id, composition=other)
    assert DagCheck.COMPOSITION in validate(recoloured, inst, complete).failed_checks


# --------------------------------------------------------------------------- each check


def test_mix_volume_tamper() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    found = checks(replace_node(dag, 2, volume=6), CASE3)
    # The phantom unit is never used up, so the replay also reports an incomplete WASTE take.
    assert DagCheck.VOLUME_CONSERVATION in found
    assert found <= {DagCheck.VOLUME_CONSERVATION, DagCheck.TAKE}


def test_edge_volume_tamper() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    edges = tuple(replace(e, volume=2) if (e.src, e.dst) == (2, 3) else e for e in dag.edges)
    assert DagCheck.VOLUME_CONSERVATION in checks(replace(dag, edges=edges), CASE3)


def test_mix_composition_tamper() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    bad = replace_node(dag, 2, composition=Composition.from_ints((1, 1), 2))
    assert DagCheck.COMPOSITION in checks(bad, CASE3)


def test_output_composition_tamper() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    out = next(n for n in dag.nodes if n.type is NodeType.OUTPUT)
    bad = replace_node(dag, out.id, composition=Composition.from_ints((1, 3), 4))
    assert checks(bad, CASE3) == {DagCheck.COMPOSITION, DagCheck.OUTPUT_TOLERANCE}


def test_reservoir_composition_tamper() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    bad = replace_node(dag, 0, composition=Composition.pure(1, 2))
    assert DagCheck.COMPOSITION in checks(bad, CASE3)


def test_output_out_of_tolerance_for_other_target() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    other = dilution([(1, 3)])  # same plan, different target composition
    assert checks(dag, replace(other, instance_id=CASE3.instance_id)) == {DagCheck.OUTPUT_TOLERANCE}


def test_output_volume_must_match_target() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    two_units = dilution([(3, 10)], droplet_volume=2)
    assert checks(dag, two_units) == {DagCheck.OUTPUT_TOLERANCE}


def test_demand_unmet() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    assert checks(dag, dilution([(3, 10)], demand=2)) == {DagCheck.DEMAND}


def test_totals_tamper() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    bad = replace(dag, totals=replace(dag.totals, mix_ops=1))
    assert checks(bad, CASE3) == {DagCheck.TOTALS}


def test_cycle() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    cyclic = replace(dag, edges=(*dag.edges, DagEdge(3, 2, 1)))
    assert checks(cyclic, CASE3) == {DagCheck.ACYCLIC}


def test_mix_limits_on_smaller_chip() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)  # mixes of volume 5 and 4
    small = dilution([(3, 10)], chip_limits=chip(max_mix_volume=4))
    assert checks(dag, small) == {DagCheck.MIX_LIMITS}


def test_storage_on_smaller_chip() -> None:
    dag = plan_dag(CASE4, CASE4_PLAN)  # 1/2, 3/4 on chip together, then 3/4, 1/4
    tight = dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2", storage_capacity=1))
    assert checks(dag, tight) == {DagCheck.STORAGE}


def test_partial_take_on_chip_without_it() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)  # takes 3 of 5 units of 2/5
    strict = dilution([(3, 10)], chip_limits=chip(allow_partial_take=False))
    assert checks(dag, strict) == {DagCheck.TAKE}


def test_structure_errors() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    dangling = replace(dag, edges=(*dag.edges, DagEdge(3, 99, 1)))
    assert checks(dangling, CASE3) == {DagCheck.STRUCTURE}
    assert checks(replace(dag, instance_id="other"), CASE3) == {DagCheck.STRUCTURE}
    dup = replace(dag, nodes=(*dag.nodes, dag.nodes[-1]))
    assert DagCheck.STRUCTURE in checks(dup, CASE3)
    into_res = replace(dag, edges=(*dag.edges, DagEdge(2, 0, 1)))
    assert DagCheck.STRUCTURE in checks(into_res, CASE3)


def test_raise_if_invalid_names_instance() -> None:
    dag = plan_dag(CASE3, CASE3_PLAN)
    with pytest.raises(DagValidationError, match=r"(?s)test-dilution.*totals"):
        validate(replace(dag, totals=StepCost.zero(2)), CASE3).raise_if_invalid()


# --------------------------------------------------------------------------- I/O


@pytest.mark.parametrize(("inst", "plan"), [(CASE3, CASE3_PLAN), (CASE4, CASE4_PLAN)])
def test_json_round_trip(inst: Instance, plan: list[Action], tmp_path: Path) -> None:
    dag = plan_dag(inst, plan)
    path = tmp_path / "dag.json"
    dag.to_json(path)
    back = MixingDAG.from_json(path)
    assert back == dag
    assert validate(back, inst).ok


def test_json_rejects_unknown_format(tmp_path: Path) -> None:
    data = plan_dag(CASE3, CASE3_PLAN).to_dict()
    data["format"] = "something-else"
    with pytest.raises(ValueError, match="unsupported DAG format"):
        MixingDAG.from_dict(data)


def test_networkx_export() -> None:
    import networkx as nx

    dag = plan_dag(CASE4, CASE4_PLAN)
    g = dag.to_networkx()
    assert nx.is_directed_acyclic_graph(g)
    assert g.number_of_nodes() == len(dag.nodes)
    assert g.number_of_edges() == len(dag.edges)
    assert g.nodes[2]["type"] is NodeType.MIX
    assert g.edges[2, 3]["volume"] == 1
    assert isinstance(g.nodes[0]["composition"], Composition)
    assert all(isinstance(n, DagNode) for n in dag.nodes)
