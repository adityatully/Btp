"""`scripts/resolution_check.py` (D-50, Q-121): the replay records each position's parent, cost
spent and planner finish; sibling groups compare whole-plan costs; the reading follows the
rule fixed in advance; a run end to end writes the table."""

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
DEMO = {"agent": "", "seed": 0, "samples": True, "min_visits": 2, "value_cap": 8}


def load(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    import search_check  # noqa: F401  (registers the dataclass module first)

    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RC = load("resolution_check")


@pytest.fixture(scope="module")
def demos(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("demos") / "demos.jsonl"
    recs = [RC.ei.run_task((i.to_json(), 16, 30.0, DEMO)) for i in TINY.instances[:3]]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return path


def test_replay_records_parents_spent_planner_and_bound(demos: Path) -> None:
    rec = json.loads(demos.read_text().split("\n")[0])
    store, spent, planner, bound = RC.replay((rec["instance"], rec["samples"], 0.01))
    assert len(store) == len(spent) == len(planner) == len(bound) == len(store.parent)
    assert store.parent[0] == -1 and spent[0] == 0.0
    for i, p in enumerate(store.parent):
        assert p < i and (p == -1 or store.depth[p] < store.depth[i])
        if p >= 0:
            assert spent[i] >= spent[p]
    assert not np.isnan(planner[0])
    # at the start the planner finish is the whole sharing-pass plan
    assert planner[0] == pytest.approx(rec["planner_cost"])
    # a lower bound: never above any way to finish
    for b, p, v in zip(bound, planner, store.value, strict=True):
        assert b >= 0 and b <= -v / 0.01 + 1e-9 and (np.isnan(p) or b <= p + 1e-9)


def rows(**kw: Any) -> dict[str, np.ndarray]:
    n = len(kw["target"])
    base = {
        "problem": np.zeros(n),
        "parent": np.array([-1] + [0] * (n - 1)),
        "n": np.full(n, 3),
        "tried": np.full(n, 1.0),
        "spent": np.zeros(n),
        "pred": np.zeros(n),
        "planner": np.full(n, np.nan),
        "bound": np.zeros(n),
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in (base | kw).items()}


def test_siblings_compare_whole_plans() -> None:
    # children 1 and 2: remaining 10 vs 12, but child 2 spent 5 less: child 2 is best
    got = RC.metrics(
        rows(
            target=[20.0, 10.0, 12.0, 30.0],
            spent=[0.0, 7.0, 2.0, 0.0],
            pred=[20.0, 10.0, 13.0, 31.0],
            planner=[20.0, 11.0, 12.0, 40.0],
        )
    )
    assert got["groups"] == 1
    assert got["runner_gap"] == pytest.approx(3.0)  # totals 17, 14, 30
    assert got["head_best"] == 1.0 and got["planner_best"] == 1.0
    assert got["chance_best"] == pytest.approx(1 / 3)
    assert got["candidates"] == 3 and got["tried"] == 1.0


def test_bound_and_planner_columns() -> None:
    got = RC.metrics(
        rows(
            target=[20.0, 10.0, 12.0, 30.0],
            spent=[0.0, 7.0, 2.0, 0.0],
            pred=[20.0, 10.0, 13.0, 31.0],
            planner=[20.0, 11.0, 12.0, 40.0],
            bound=[16.0, 9.0, 8.0, 20.0],
        )
    )
    assert got["beats_planner"] == 0.5 and got["beat_by"] == pytest.approx(5.5)
    assert got["excess"] == pytest.approx(4.0)  # 4, 1, 4, 10
    assert got["bound_best"] == 1.0  # spent + bound: 16, 10, 20
    # head error median 0.5 (0, 0, 1, 1) over median target 16, times excess 4
    assert got["projected"] == pytest.approx(0.125)


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    row = {"candidates": 5.0, "tried": 0.9, "runner_gap": 2.0, "head_err": 10.0}
    got = RC.reading(row | {"residual": 10.0, "target": 100.0})
    assert "little to steer" in got[0] and "cannot resolve" in got[1] and "smaller" in got[2]
    assert len(got) == 3  # no bound columns: no bound lines
    got = RC.reading(row | {"residual": 10.0, "target": 100.0, "excess": 20.0, "projected": 1.0})
    assert "carries most" in got[3] and "could resolve" in got[4]
    got = RC.reading(row | {"residual": 10.0, "target": 100.0, "excess": 60.0, "projected": 5.0})
    assert "loose" in got[3] and "too coarse" in got[4]
    got = RC.reading(
        row | {"candidates": 20.0, "runner_gap": 20.0, "residual": 60.0, "target": 100.0}
    )
    assert "has room" in got[0] and "not the problem" in got[1] and got[2].endswith("not smaller")


@pytest.fixture(scope="module")
def agent_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    net = SetEncoderNet(PPOConfig.from_dict(RC.sc._load("agent/ppo.yaml")).net)
    path = tmp_path_factory.mktemp("agent") / "agent.pt"
    PPOAgent(net, torch.device("cpu")).save(path)
    return str(path)


def test_run_end_to_end(agent_path: str, demos: Path, tmp_path: Path) -> None:
    out = tmp_path / "table.json"
    RC.main([f"agent={agent_path}", f"demos={demos}", "workers=2", "threads=1", f"out={out}"])
    table = json.loads(out.read_text())
    assert set(table) == {"train/small", "held/small"}
    assert table["train/small"]["groups"] > 0 and table["train/small"]["tried"] is not None
