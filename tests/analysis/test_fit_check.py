"""`scripts/fit_check.py` (Q-111): the split matches the copy step's, pair order counts sibling
positions only, the reading follows the rule fixed in advance, and a run end to end writes the
table."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from meda.agents.ppo.networks import SetEncoderNet
from meda.agents.ppo.policy import PPOAgent
from meda.agents.ppo.ppo import PPOConfig
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def load(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    import search_check  # noqa: F401  (registers the dataclass module first)

    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FC = load("fit_check")


def test_split_holds_out_the_last_5_percent_and_samples_training_per_stage() -> None:
    s4 = next(k for k, v in FC.stage_names().items() if v == "s4_targets")
    lines = [{"cell": s4 if i % 2 else "small-cell"} for i in range(100)]
    chosen = FC.split(lines, per_stage=10)
    assert [i for part, i in chosen if part == "held"] == list(range(95, 100))
    train = [i for part, i in chosen if part == "train"]
    assert len(train) == 20 and max(train) < 95


def test_pair_order_compares_siblings_only() -> None:
    problem = np.array([0, 0, 0, 1])
    depth = np.array([1, 1, 2, 1])
    target = np.array([-10.0, -20.0, -5.0, -7.0])
    assert FC.pair_order(problem, depth, target, np.array([-1.0, -2.0, 0.0, 0.0])) == (1, 1)
    assert FC.pair_order(problem, depth, target, np.array([-2.0, -1.0, 0.0, 0.0])) == (0, 1)
    same = np.array([-10.0, -10.05, -5.0, -7.0])  # within 1%: not a pair
    assert FC.pair_order(problem, depth, same, np.zeros(4)) == (0, 0)


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    def row(agree: float, pair: float) -> dict[str, float]:
        return {"agree": agree, "pair_order": pair}

    got = FC.reading(row(0.5, 0.65), row(0.5, 0.55))
    assert got[0].startswith("guide: cannot fit") and got[1].startswith("judge: cannot fit")
    assert "coin flip" in got[2]
    got = FC.reading(row(0.9, 0.9), row(0.6, 0.7))
    assert "does not generalise" in got[0] and "does not generalise" in got[1]
    got = FC.reading(row(0.8, 0.8), row(0.75, 0.75))
    assert "generalises" in got[0] and len(got) == 2


@pytest.fixture(scope="module")
def agent_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    net = SetEncoderNet(PPOConfig.from_dict(load("search_check")._load("agent/ppo.yaml")).net)
    path = tmp_path_factory.mktemp("agent") / "agent.pt"
    PPOAgent(net, torch.device("cpu")).save(path)
    return str(path)


def test_run_end_to_end(agent_path: str, tmp_path: Path) -> None:
    de = load("dead_ends")
    opts = {"agent": agent_path, "seconds": 60.0, "sims": 4, "reward_scale": 0.01, "factor": 2.0}
    demos = tmp_path / "demos.jsonl"
    demos.write_text(
        "\n".join(json.dumps(de.run_label((i.to_json(), opts))) for i in TINY.instances[:3])
    )
    out = tmp_path / "table.json"
    FC.main([f"agent={agent_path}", f"demos={demos}", "workers=2", "threads=1", f"out={out}"])
    table = json.loads(out.read_text())
    assert set(table) == {"train/small", "held/small"}
    assert table["held/small"]["problems"] == 1
    assert table["train/small"]["value_mae_cost"] is not None
