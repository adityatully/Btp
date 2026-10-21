"""The evaluation CLI end to end, and M7's acceptance criterion (SPEC 16)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from meda.core.composition import Rational
from meda.evaluate_cli import make_producer, run
from meda.generators.benchmark_io import BenchmarkSet
from tests.core.helpers import load_yaml


def config(**eval_overrides: Any) -> dict[str, Any]:
    return {
        "seed": 0,
        "costs": load_yaml("costs/default.yaml"),
        "env": load_yaml("env/sample_prep.yaml"),
        "eval": {**load_yaml("eval/default.yaml"), **eval_overrides},
    }


def test_cli_writes_results_summary_and_plots(tmp_path: Path) -> None:
    results = run(config(producers=["random", "naive"]), tmp_path)
    assert (tmp_path / "summary.md").exists()
    assert sorted(p.name for p in (tmp_path / "plots").iterdir()) == [
        "cost_vs_naive.png",
        "cost_vs_targets.png",
        "gap_histogram.png",
    ]
    on_disk = pd.read_parquet(tmp_path / "results.parquet")
    assert len(on_disk) == len(results) == 2 * 10
    assert (on_disk.loc[on_disk["producer"] == "naive", "gap_to_naive"] == 0).all()


def test_unknown_producer_is_named() -> None:
    bench = BenchmarkSet.read("tiny_exact_v1")
    with pytest.raises(ValueError, match="unknown producer 'ppo'"):
        make_producer("ppo", config(), bench)


@pytest.mark.slow
def test_m7_acceptance_random_naive_exact_on_tiny(tmp_path: Path) -> None:
    """SPEC 16, M7: results.parquet and summary.md for random, naive and exact search on
    tiny_exact_v1, with correct gaps. Exact search must land on every stored optimum (gap 0),
    and naive's gaps must equal the ones computed by hand from the stored costs."""
    results = run(config(producers=["random", "naive", "exact_search"]), tmp_path)
    exact = results[results["producer"] == "exact_search"]
    assert exact["success"].all()
    assert (exact["gap_to_optimal"] == 0).all()
    naive = results[results["producer"] == "naive"].set_index("instance_id")
    bench = BenchmarkSet.read("tiny_exact_v1")
    for iid, ref in bench.reference.items():
        cost = naive.at[iid, "cost_exact"]
        c, o = Rational.parse(cost), ref.cost
        assert naive.at[iid, "gap_to_optimal"] == pytest.approx(float(c - o) / float(o))
    assert (tmp_path / "summary.md").exists() and (tmp_path / "results.parquet").exists()


def test_every_cli_producer_has_a_plot_colour() -> None:
    """`tree` (0.12) was a producer without a colour slot, so plotting crashed on it."""
    from meda.analysis.plots import colour
    from meda.evaluate_cli import PRODUCERS

    for name in PRODUCERS:
        colour(name)
