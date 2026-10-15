"""`meda.train` grades tabular Q through the M7 harness (`analysis.evaluate`)."""

from __future__ import annotations

from pathlib import Path

from meda.train import train_tabular_q
from tests.core.helpers import load_yaml


def test_tabular_q_is_graded_by_evaluate(tmp_path: Path) -> None:
    cfg = {
        "seed": 0,
        "costs": load_yaml("costs/default.yaml"),
        "env": load_yaml("env/sample_prep.yaml"),
        "eval": load_yaml("eval/default.yaml"),
        "agent": {**load_yaml("agent/tabular_q.yaml"), "episodes": 3000},
        "train": {
            **load_yaml("train/default.yaml"),
            "instances": ["tiny-quarter-m2"],
            "log_every": 0,
        },
    }
    df = train_tabular_q(cfg, tmp_path)
    row = df.iloc[0]
    assert row["instance_id"] == "tiny-quarter-m2"
    assert row["status"] == "feasible"
    assert row["cost_exact"] == row["reference_cost"] == "15"
    assert row["gap_to_optimal"] == 0
    assert row["mix_ops"] == 2 and row["episodes"] == 3000
    assert (tmp_path / "checkpoints" / "tiny-quarter-m2.pkl").exists()
