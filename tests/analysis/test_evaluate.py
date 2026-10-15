"""Evaluation harness: gaps on a hand-made benchmark, validation as a hard error, baseline gaps
and the summary (SPEC 12.2, 17.5). Expected gaps are worked out by hand."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from meda.agents.random_agent import RandomAgent
from meda.analysis.compare import add_baseline_gaps, summarize, tier_of, write_summary_md
from meda.analysis.dag_metrics import ScheduleConfig
from meda.analysis.evaluate import AgentProducer, SolverProducer, build_env, evaluate
from meda.core.composition import Rational
from meda.core.dag import DagValidationError
from meda.core.problem import Instance
from meda.envs.sample_prep_env import SamplePrepEnv
from meda.generators.benchmark_io import BenchmarkSet, InstanceReference
from meda.solvers.base import SolveBudget, SolveResult
from meda.solvers.baselines.naive import NaiveSolver
from tests.core.helpers import chip, default_weights, dilution, load_yaml

WEIGHTS = default_weights()
SCHEDULE = ScheduleConfig(mix_duration=1, output_duration=1, max_concurrent_mixes=None)
CFG = {"env": load_yaml("env/sample_prep.yaml"), "costs": load_yaml("costs/default.yaml")}


def m2(n: int, d: int, iid: str) -> Instance:
    inst = dilution([(n, d)], chip_limits=chip("dmfb_m2", max_denominator=d), step_limit=16)
    return replace(inst, instance_id=iid)


QUARTER, SEVEN, HALF = m2(1, 4, "quarter"), m2(7, 32, "seven-32"), m2(1, 2, "half")
BENCH = BenchmarkSet(
    "tiny_exact_handmade",
    1,
    (QUARTER, SEVEN, HALF),
    reference={
        # naive costs 15 here, which is the optimum: gap 0.
        "quarter": InstanceReference(Rational.of(15), "optimal", "exact_search"),
        # naive costs 83/2. Against [13, 39/2]: best known 44/39, lower bound 57/26, ratio 3/2.
        "seven-32": InstanceReference(
            Rational.of(39, 2), "feasible", "hand", lower_bound=Rational.of(13)
        ),
        # "half" has no reference: no gaps.
    },
)


def random_env() -> SamplePrepEnv:
    return build_env(CFG, list(BENCH.instances))


def naive() -> SolverProducer:
    return SolverProducer(NaiveSolver(WEIGHTS), SolveBudget())


def test_gaps_against_optimum_and_interval() -> None:
    df = evaluate(naive(), BENCH, SCHEDULE).set_index("instance_id")
    assert df.at["quarter", "cost_exact"] == "15"
    assert df.at["quarter", "gap_to_optimal"] == 0
    assert df.at["quarter", "gap_to_best_known"] == 0
    assert df.at["seven-32", "cost_exact"] == "83/2"
    assert pd.isna(df.at["seven-32", "gap_to_optimal"])  # never a gap to a bound in this column
    assert df.at["seven-32", "gap_to_best_known"] == pytest.approx(44 / 39, rel=1e-15)
    assert df.at["seven-32", "gap_to_lower_bound"] == pytest.approx(57 / 26, rel=1e-15)
    assert df.at["seven-32", "bound_ratio"] == 1.5
    assert df.loc["half", ["gap_to_optimal", "gap_to_best_known"]].isna().all()
    assert df["success"].all()
    assert (df.loc[["quarter", "seven-32"], "mix_ops"] == [2, 5]).all()


def test_an_invalid_dag_is_a_hard_error() -> None:
    class Tampering:
        name = "tampering"

        def produce(self, instance: Instance) -> SolveResult:
            r = NaiveSolver(WEIGHTS).solve(instance)
            assert r.dag is not None
            nodes = tuple(replace(n, volume=n.volume + 1) if n.id == 2 else n for n in r.dag.nodes)
            return replace(r, dag=replace(r.dag, nodes=nodes))

    with pytest.raises(DagValidationError):
        evaluate(Tampering(), BENCH, SCHEDULE)


def test_agent_failures_get_no_gap() -> None:
    env = build_env(CFG, list(BENCH.instances))
    df = evaluate(AgentProducer(RandomAgent(seed=3), env), BENCH, SCHEDULE)
    assert df["gap_to_best_known"][~df["success"]].isna().all()
    ok = df[df["success"] & df["reference_cost"].notna()]
    assert (ok["gap_to_best_known"] >= -1).all()
    assert set(df["status"]) <= {"feasible", "failed"}


def test_best_of_k_needs_sampling() -> None:
    env = build_env(CFG, list(BENCH.instances))
    with pytest.raises(ValueError, match="deterministic"):
        AgentProducer(RandomAgent(), env, deterministic=True, best_of_k=4)
    r = AgentProducer(RandomAgent(seed=0), env, deterministic=False, best_of_k=8).produce(HALF)
    assert r.status == "feasible"  # 1/2 is one mix away; 8 tries find it


def test_baseline_gaps_from_hand_made_tables() -> None:
    results = pd.DataFrame(
        {
            "instance_id": ["a", "b", "c"],
            "success": [True, True, False],
            "cost_exact": ["20", "39/2", "7"],
        }
    )
    base = pd.DataFrame(
        {"instance_id": ["a", "b", "c"], "success": [True, False, True], "cost_exact": ["40"] * 3}
    )
    out = add_baseline_gaps(results, {"naive": base})
    assert out["gap_to_naive"].tolist()[0] == -0.5
    assert out["gap_to_naive"].isna().tolist() == [False, True, True]


def test_summary_md_states_the_tier(tmp_path) -> None:
    df = pd.concat(
        [
            evaluate(naive(), BENCH, SCHEDULE),
            evaluate(AgentProducer(RandomAgent(seed=1), random_env()), BENCH, SCHEDULE),
        ]
    )
    summary = summarize(df)
    assert set(summary["producer"]) == {"naive", "random"}
    row = summary[(summary["producer"] == "naive") & (summary["metric"] == "mix_ops")].iloc[0]
    assert row["n"] == 3
    write_summary_md(tmp_path / "summary.md", df)
    text = (tmp_path / "summary.md").read_text()
    assert "0 Correctness" in text
    assert "83/2" in text
    with pytest.raises(ValueError, match="tier"):
        tier_of("mystery_v1")
