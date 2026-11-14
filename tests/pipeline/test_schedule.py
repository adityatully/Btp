"""Execution harness, part 1 (SPEC 14.2-14.4, milestone M10).

Expected numbers are worked out by hand from the graphs, not read off the code.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from meda.core.actions import Action, Mix, Output, Source
from meda.core.composition import Composition
from meda.core.costs import CostWeights
from meda.core.dag import DagValidationError, MixingDAG
from meda.core.rules import replay
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet
from meda.pipeline.opgraph import AssaySpec, Droplet, Op, OpGraph, OpKind, build_opgraph
from meda.pipeline.platform import Platform
from meda.pipeline.schedule import (
    ListScheduler,
    Schedule,
    ScheduleDeadlock,
    ScheduleError,
    ScheduleRule,
    Slot,
    check_schedule,
    schedule_metrics,
    validate_schedule,
)
from meda.pipeline.schedule_ilp import exact_makespan
from meda.solvers.baselines.share import ShareSolver
from tests.core.helpers import chip, dilution

ROOT = Path(__file__).resolve().parents[2]
PLATFORM = Platform.load()
S, B = Source.reservoir(0), Source.reservoir(1)

# 3/4 and 1/4 from one shared 1/2 (as in test_dag_metrics): node 2 = S+B, node 3 = 2+S,
# node 4 = 2+B; each output takes 1 of 2 units, the rest goes to waste.
SHARED = dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2", max_denominator=4))
SHARED_PLAN: list[Action] = [
    Mix(S, 1, B, 1),
    Mix(Source.droplet(2), 1, S, 1),
    Mix(Source.droplet(2), 1, B, 1),
    Output(Source.droplet(3), 0),
    Output(Source.droplet(4), 1),
]


def _dag(instance, plan) -> MixingDAG:  # type: ignore[no-untyped-def]
    _, totals, deltas = replay(instance, plan)
    return MixingDAG.from_deltas(instance, deltas, totals, "hand")


def _weights() -> CostWeights:
    data = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/costs/geometric.yaml"))
    assert isinstance(data, dict)
    return CostWeights.from_dict({str(k): v for k, v in data.items()})


def _four_mixes() -> OpGraph:
    """The doc's example (section 4): four 2-tick mixes, M1 and M2 feed M3, M4 independent.
    Plan order (step) M1, M4, M2, M3. Op ids: M1 0, M2 1, M3 2, M4 3."""
    ops = (
        Op(0, OpKind.MIX, 2, node_id=10, step=0),
        Op(1, OpKind.MIX, 2, node_id=11, step=2),
        Op(2, OpKind.MIX, 2, node_id=12, step=3, after_end=(0, 1)),
        Op(3, OpKind.MIX, 2, node_id=13, step=1),
    )
    return OpGraph("four-mixes", ops, (Droplet(0, (2,)), Droplet(1, (2,))), 10)


TWO_MIXERS = replace(PLATFORM, mixers=2)


# ---------------------------------------------------------------- platform


def test_platform_loads_and_rejects_bad_values() -> None:
    assert PLATFORM.mixers == 4 and PLATFORM.storage_capacity is None
    data = PLATFORM.to_dict()
    with pytest.raises(ValueError, match="missing"):
        Platform.from_dict({k: v for k, v in data.items() if k != "mixers"})
    with pytest.raises(ValueError, match="mix_ticks"):
        Platform.from_dict({**data, "mix_ticks": 0})
    with pytest.raises(ValueError, match="storage_capacity"):
        Platform.from_dict({**data, "storage_capacity": 0})


# ---------------------------------------------------------------- operation graph


def test_opgraph_of_the_shared_plan() -> None:
    g = build_opgraph(_dag(SHARED, SHARED_PLAN), SHARED, PLATFORM)
    kinds = [o.kind for o in g.ops]
    D, M, T, X = OpKind.DISPENSE, OpKind.MIX, OpKind.DETECT, OpKind.DISCARD
    assert kinds == [D, D, M, D, M, D, M, T, T, X, X]
    assert [o.feeds for o in g.ops[:2]] == [2, 2] and g.ops[3].feeds == 4
    assert g.ops[4].after_end == (2,) and g.ops[6].after_end == (2,)
    # the 1/2 droplet (op 2) is used by both later mixes; each waste waits for the detection
    assert any(d.producer == 2 and d.consumers == (4, 6) for d in g.droplets)
    # volumes (M11): every droplet's takes sum to its volume; the 1/2 droplet is 2 units
    assert all(sum(d.takes) == d.volume > 0 for d in g.droplets)
    assert next(d for d in g.droplets if d.producer == 2).takes == (1, 1)
    assert g.ops[9].after_start == (8,) and g.ops[10].after_start == (7,)
    assert g.ops[7].composition == Composition.from_ints((3, 1), 4)
    assert g.storage_capacity == SHARED.chip.storage_capacity


def test_opgraph_without_detection_delivers() -> None:
    g = build_opgraph(_dag(SHARED, SHARED_PLAN), SHARED, PLATFORM, AssaySpec(detect=False))
    delivered = [o for o in g.ops if o.kind is OpKind.DELIVER]
    assert len(delivered) == 2 and all(o.duration == 0 for o in delivered)


def test_opgraph_rejects_an_invalid_dag() -> None:
    dag = _dag(SHARED, SHARED_PLAN[:3])  # demand not met
    with pytest.raises(DagValidationError):
        build_opgraph(dag, SHARED, PLATFORM)


# ---------------------------------------------------------------- list scheduler


def test_doc_example_critical_path_gives_schedule_b() -> None:
    """Acceptance M10 (a): M1, M2 together, then M3 and M4: makespan 4, nothing stored."""
    g = _four_mixes()
    s = ListScheduler("critical_path").schedule(g, TWO_MIXERS)
    validate_schedule(s, g, TWO_MIXERS)
    assert [x.start for x in s.slots] == [0, 0, 2, 2]
    m = schedule_metrics(s, g)
    assert (m.makespan, m.storage_peak, m.storage_ticks) == (4, 0, 0)


def test_doc_example_plan_order_gives_schedule_a() -> None:
    """Acceptance M10 (a): M1, M4 / M2 / M3: makespan 6, M1's output stored 2 ticks."""
    g = _four_mixes()
    s = ListScheduler("plan_order").schedule(g, TWO_MIXERS)
    validate_schedule(s, g, TWO_MIXERS)
    assert [x.start for x in s.slots] == [0, 2, 4, 0]
    m = schedule_metrics(s, g)
    assert (m.makespan, m.storage_peak, m.storage_ticks) == (6, 1, 2)


def test_shared_plan_schedule_by_hand() -> None:
    """Dispense S, B at 0; 1/2 mixes 1-3; both later mixes at 3-5 (their dispenses at 2-3);
    both detections at 5-9 on the two detectors; wastes at 5. Nothing waits."""
    g = build_opgraph(_dag(SHARED, SHARED_PLAN), SHARED, PLATFORM)
    s = ListScheduler().schedule(g, PLATFORM)
    validate_schedule(s, g, PLATFORM)
    assert [(x.start, x.end) for x in s.slots] == [
        (0, 1),
        (0, 1),
        (1, 3),
        (2, 3),
        (3, 5),
        (2, 3),
        (3, 5),
        (5, 9),
        (5, 9),
        (5, 5),
        (5, 5),
    ]
    assert s.slots[7].resource == ("detector", 0) and s.slots[8].resource == ("detector", 1)
    assert s.slots[3].resource == ("port", 0, 0) and s.slots[5].resource == ("port", 1, 0)
    m = schedule_metrics(s, g)
    assert (m.makespan, m.storage_ticks, m.detector_wait, m.changeovers) == (9, 0, 0, 0)
    assert (m.mixer_busy, m.port_busy, m.detector_busy) == (6, 4, 8)


def test_one_detector_waits_and_changes_over() -> None:
    """One detector: the second detection waits for the first (5-9) and a 2-tick wash."""
    one = replace(PLATFORM, detectors=1)
    g = build_opgraph(_dag(SHARED, SHARED_PLAN), SHARED, one)
    s = ListScheduler().schedule(g, one)
    validate_schedule(s, g, one)
    assert (s.slots[7].start, s.slots[8].start) == (5, 11)
    m = schedule_metrics(s, g)
    assert (m.makespan, m.detector_wait, m.changeovers) == (15, 6, 1)
    # the 1/4 droplet waits on the chip for the detector: 6 ticks
    assert (m.storage_peak, m.storage_ticks) == (1, 6)


def test_storage_full_raises_a_deadlock() -> None:
    """Capacity 1: two droplets must wait together for M3, so nothing can finish."""
    g = replace(_four_mixes(), storage_capacity=1)
    sched = ListScheduler()
    with pytest.raises(ScheduleDeadlock, match="four-mixes"):
        sched.schedule(replace(g, ops=g.ops[:3]), replace(TWO_MIXERS, mixers=1))


def test_same_input_same_schedule() -> None:
    """Acceptance M10 (c)."""
    inst = BenchmarkSet.read("val_s3_v2", BENCHMARKS_DIR).instances[2]
    dag = ShareSolver(_weights()).solve(inst).dag
    assert dag is not None
    g = build_opgraph(dag, inst, PLATFORM)
    assert ListScheduler().schedule(g, PLATFORM) == ListScheduler().schedule(g, PLATFORM)


# ---------------------------------------------------------------- validator


def _valid() -> tuple[Schedule, OpGraph]:
    g = build_opgraph(_dag(SHARED, SHARED_PLAN), SHARED, PLATFORM)
    return ListScheduler().schedule(g, PLATFORM), g


def _with(s: Schedule, op: int, slot: Slot) -> Schedule:
    slots = list(s.slots)
    slots[op] = slot
    return replace(s, slots=tuple(slots))


@pytest.mark.parametrize(
    ("op", "slot", "rule"),
    [
        (4, Slot(2, 4, ("mixer", 1)), ScheduleRule.PRECEDENCE),  # before the 1/2 mix ends
        (4, Slot(3, 4, ("mixer", 1)), ScheduleRule.COMPLETE),  # wrong duration
        (4, Slot(3, 5, ("detector", 0)), ScheduleRule.CAPACITY),  # wrong resource kind
        (4, Slot(3, 5, ("mixer", 9)), ScheduleRule.CAPACITY),  # no such mixer
        (6, Slot(3, 5, ("mixer", 1)), ScheduleRule.CAPACITY),  # overlaps op 4 on mixer 1
        (8, Slot(9, 13, ("detector", 0)), ScheduleRule.CHANGEOVER),  # no wash after 3/4
        (0, Slot(0, 1, ("port", 1, 0)), ScheduleRule.CAPACITY),  # S dispensed at B's port
    ],
)
def test_tampering_breaks_the_right_rule(op: int, slot: Slot, rule: ScheduleRule) -> None:
    s, g = _valid()
    s = _with(s, op, slot)
    if op == 6:  # make op 4 sit on mixer 1 too
        s = _with(s, 4, Slot(3, 5, ("mixer", 1)))
    with pytest.raises(ScheduleError) as info:
        validate_schedule(s, g, PLATFORM)
    assert rule in info.value.rules
    assert "test-dilution" in str(info.value)


def test_validator_checks_storage() -> None:
    s, g = _valid()
    late = _with(s, 8, Slot(20, 24, ("detector", 1)))  # the 1/4 droplet waits 5-20
    late = _with(late, 9, Slot(20, 20, None))
    assert not check_schedule(late, g, PLATFORM)
    issues = check_schedule(late, replace(g, storage_capacity=0), PLATFORM)
    assert {i.rule for i in issues} == {ScheduleRule.STORAGE}


# ---------------------------------------------------------------- exact check


def test_ilp_matches_the_doc_example() -> None:
    """Acceptance M10 (d), hand case: the optimum is schedule B's 4."""
    plat = replace(TWO_MIXERS, changeover_ticks=0)
    assert exact_makespan(_four_mixes(), plat, horizon=6).makespan == 4
    with pytest.raises(ValueError, match="changeover"):
        exact_makespan(_four_mixes(), TWO_MIXERS, horizon=6)


def test_ilp_never_worse_than_list_on_tiny_plans() -> None:
    """Acceptance M10 (d): the sharing pass's plans on tiny_exact_v1."""
    plat = replace(PLATFORM, changeover_ticks=0)
    w = _weights()
    for inst in BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR).instances:
        dag = ShareSolver(w).solve(inst).dag
        assert dag is not None
        g = build_opgraph(dag, inst, plat)
        listed = schedule_metrics(ListScheduler().schedule(g, plat), g).makespan
        result = exact_makespan(g, plat, horizon=listed)
        assert result.optimal and result.makespan <= listed


# ---------------------------------------------------------------- val sets


def _val_plans(per_set: int | None):  # type: ignore[no-untyped-def]
    w = _weights()
    for stage in range(1, 6):
        insts = BenchmarkSet.read(f"val_s{stage}_v2", BENCHMARKS_DIR).instances
        for inst in insts[:per_set]:
            dag = ShareSolver(w).solve(inst).dag
            assert dag is not None, inst.instance_id
            yield inst, dag


def _schedules_validate(per_set: int | None) -> None:
    for inst, dag in _val_plans(per_set):
        g = build_opgraph(dag, inst, PLATFORM)
        for priority in ListScheduler.PRIORITIES:
            s = ListScheduler(priority).schedule(g, PLATFORM)
            validate_schedule(s, g, PLATFORM)
            assert schedule_metrics(s, g).storage_peak <= g.storage_capacity


def test_val_plans_schedule_and_validate_sample() -> None:
    _schedules_validate(per_set=3)


@pytest.mark.slow
def test_val_plans_schedule_and_validate_all() -> None:
    """Acceptance M10 (b): every sharing-pass plan of val_s1_v2 ... val_s5_v2."""
    _schedules_validate(per_set=None)


def test_tight_storage_never_boxes_the_scheduler_in() -> None:
    """val_s5-006 (capacity 2) deadlocked before the plan-order safety check: critical-path
    order filled storage with droplets nothing could use up."""
    inst = next(
        i
        for i in BenchmarkSet.read("val_s5_v2", BENCHMARKS_DIR).instances
        if i.instance_id == "val_s5-006"
    )
    dag = ShareSolver(_weights()).solve(inst).dag
    assert dag is not None
    g = build_opgraph(dag, inst, PLATFORM)
    assert g.storage_capacity == 2
    for priority in ListScheduler.PRIORITIES:
        s = ListScheduler(priority).schedule(g, PLATFORM)
        validate_schedule(s, g, PLATFORM)


def test_ilp_out_of_time_reports_a_bound_not_an_error() -> None:
    """A time limit with no schedule found is an answer ("no answer, bound b"), not a crash."""
    inst = next(
        i
        for i in BenchmarkSet.read("val_s4_v2", BENCHMARKS_DIR).instances
        if i.instance_id == "val_s4-010"
    )
    dag = ShareSolver(_weights()).solve(inst).dag
    assert dag is not None
    plat = replace(PLATFORM, changeover_ticks=0)
    g = build_opgraph(dag, inst, plat)
    listed = schedule_metrics(ListScheduler().schedule(g, plat), g).makespan
    result = exact_makespan(g, plat, horizon=listed, time_limit=0.5)
    assert not result.optimal
    assert 0 <= result.lower_bound <= listed
    assert result.makespan is None or result.lower_bound <= result.makespan <= listed
