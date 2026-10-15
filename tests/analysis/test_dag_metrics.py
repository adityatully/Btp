"""DAG metrics on hand-worked plans (SPEC 12.1).

Every expected number below was worked out by hand from the plan, not read off the code.
"""

from __future__ import annotations

import pytest

from meda.analysis.dag_metrics import ScheduleConfig, dag_metrics, list_schedule
from meda.core.actions import Action, Mix, Output, Source
from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.solvers.baselines.naive import NaiveSolver
from tests.core.helpers import chip, default_weights, dilution

S, B = Source.reservoir(0), Source.reservoir(1)
UNLIMITED = ScheduleConfig(mix_duration=1, output_duration=1, max_concurrent_mixes=None)
SHARING = dilution([(3, 4), (1, 4)], chip_limits=chip("dmfb_m2", max_denominator=4))
# S+B -> 1/2 (node 2); 1 of it + S -> 3/4 (node 3); the other unit + B -> 1/4 (node 4).
SHARING_PLAN: list[Action] = [
    Mix(S, 1, B, 1),
    Mix(Source.droplet(2), 1, S, 1),
    Mix(Source.droplet(2), 1, B, 1),
    Output(Source.droplet(3), 0),
    Output(Source.droplet(4), 1),
]


def dag_of(instance: Instance, plan: list[Action]) -> MixingDAG:
    _, totals, deltas = replay(instance, plan)
    return MixingDAG.from_deltas(instance, deltas, totals, "hand")


def test_sharing_case_proxy_metrics() -> None:
    m = dag_metrics(dag_of(SHARING, SHARING_PLAN), SHARING, UNLIMITED)
    assert (m.mix_ops, m.outputs) == (3, 2)
    assert (m.sample_volume, m.buffer_volume, m.reagent_volume, m.total_reagent) == (2, 2, 0, 4)
    assert m.waste_volume == 2  # one unit each of 3/4 and 1/4 left at the end
    assert m.distinct_intermediates == 3
    assert m.sharing_count == 1  # only the 1/2 node reaches both targets


def test_naive_never_shares() -> None:
    result = NaiveSolver(default_weights()).solve(SHARING)
    assert result.dag is not None
    m = dag_metrics(result.dag, SHARING, UNLIMITED)
    assert m.mix_ops == 4
    assert m.sharing_count == 0


def test_sharing_case_schedule_unlimited() -> None:
    """1/2 in slot 0; 3/4 and 1/4 side by side in slot 1; both outputs in slot 2."""
    m = dag_metrics(dag_of(SHARING, SHARING_PLAN), SHARING, UNLIMITED)
    assert (m.makespan, m.critical_path) == (3, 3)
    assert m.peak_droplets == 2  # 3/4 and 1/4 in slot 1, after 1/2 is used up
    assert m.storage_slots == 0  # nothing waits


def test_sharing_case_schedule_one_mix_at_a_time() -> None:
    """1/4 waits a slot: makespan 4 against a critical path of 3, and the second unit of 1/2
    sits on chip for one slot."""
    cfg = ScheduleConfig(mix_duration=1, output_duration=1, max_concurrent_mixes=1)
    dag = dag_of(SHARING, SHARING_PLAN)
    sched = list_schedule(dag, cfg)
    assert [sched.start[i] for i in (2, 3, 4)] == [0, 1, 2]
    m = dag_metrics(dag, SHARING, cfg)
    assert (m.makespan, m.critical_path) == (4, 3)
    assert m.storage_slots == 1
    assert m.peak_droplets == 2


def test_durations_scale_the_schedule() -> None:
    cfg = ScheduleConfig(mix_duration=2, output_duration=1, max_concurrent_mixes=None)
    m = dag_metrics(dag_of(SHARING, SHARING_PLAN), SHARING, cfg)
    assert (m.makespan, m.critical_path) == (5, 5)  # two mixes deep (4 slots) + output


def test_unfinished_plan_keeps_its_droplet_to_the_end() -> None:
    """A failed episode's DAG has a MIX droplet with no consumer: it stays alive to the end."""
    m = dag_metrics(dag_of(SHARING, [Mix(S, 1, B, 1)]), SHARING, UNLIMITED)
    assert (m.mix_ops, m.outputs, m.waste_volume) == (1, 0, 0)
    assert (m.makespan, m.peak_droplets, m.storage_slots) == (1, 1, 0)


def test_schedule_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="keys"):
        ScheduleConfig.from_dict({"mix_duration": 1})
    with pytest.raises(ValueError, match=">= 1 slot"):
        ScheduleConfig(mix_duration=0, output_duration=1, max_concurrent_mixes=None)
    with pytest.raises(ValueError, match="max_concurrent_mixes"):
        ScheduleConfig(mix_duration=1, output_duration=1, max_concurrent_mixes=0)
