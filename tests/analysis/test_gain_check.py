"""`scripts/gain_check.py` (D-52, Q-123): the labels are the planner finish minus the search's
cheapest finish, on the same samples, held-out problems last; they load as copy-step targets;
AUC and the sibling test behave at their extremes; the reading follows the rule fixed in
advance; both parts run end to end."""

from __future__ import annotations

import importlib.util
import json
import math
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


GC = load("gain_check")


@pytest.fixture(scope="module")
def demos(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("demos") / "demos.jsonl"
    recs = [GC.ei.run_task((i.to_json(), 16, 30.0, DEMO)) for i in TINY.instances[:4]]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return path


@pytest.fixture(scope="module")
def labels(demos: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("labels") / "labels.jsonl"
    GC.main(["part=label", f"demos={demos}", f"out={out}", "workers=2"])
    return out


def test_gains_are_planner_minus_search_on_the_same_samples(demos: Path) -> None:
    rec = json.loads(demos.read_text().split("\n")[0])
    before = GC.ei._unpack(rec["samples"])
    after, n, positive = GC.gains((rec["instance"], rec["samples"]))
    assert [(a, v) for a, v, _ in after] == [(a, v) for a, v, _ in before]
    assert n == sum(not math.isnan(g) for _, _, g in after) and 0 <= positive <= n
    root = next(i for i, (a, _, _) in enumerate(before) if not a)
    assert after[root][2] == pytest.approx(max(0.0, rec["planner_cost"] - before[root][2]))
    assert all(math.isnan(g) or g >= 0 for _, _, g in after)


def test_labels_keep_held_out_last_and_load_as_targets(demos: Path, labels: Path) -> None:
    src = [json.loads(x) for x in demos.read_text().split("\n") if x]
    got = [json.loads(x) for x in labels.read_text().split("\n") if x]
    assert len(got) == len(src)  # tiny: every training problem kept
    assert got[-1]["id"] == src[-1]["id"] and got[-1]["samples_cost"] == src[-1]["samples"]
    loaded = GC.ei.load_targets(labels)
    assert len(loaded) == len(got)


def test_auc_extremes_and_ties() -> None:
    pos = np.array([True, True, False, False])
    assert GC.auc(np.array([3.0, 4.0, 1.0, 2.0]), pos) == 1.0
    assert GC.auc(np.array([1.0, 2.0, 3.0, 4.0]), pos) == 0.0
    assert GC.auc(np.zeros(4), pos) == 0.5
    assert GC.auc(np.zeros(2), np.array([True, True])) is None


def rows(pred: list[float]) -> dict[str, np.ndarray]:
    # one parent, three children: planner says child 1 (12 + 0), but child 2 finishes at 9
    return {
        "parent": np.array([-1, 0, 0, 0], dtype=np.float64),
        "spent": np.array([0.0, 0.0, 1.0, 2.0]),
        "target": np.array([10.0, 12.0, 8.0, 11.0]),
        "planner": np.array([14.0, 12.0, 12.0, 11.0]),
        "pred": np.array(pred),
    }


def test_sibling_test_perfect_and_zero_gain() -> None:
    perfect = GC.metrics(rows([4.0, 0.0, 4.0, 0.0]))
    assert perfect["gain_best"] == 1.0 and perfect["planner_best"] == 0.0
    assert perfect["auc"] == 1.0 and perfect["with_gain"] == 0.5
    zero = GC.metrics(rows([0.0, 0.0, 0.0, 0.0]))
    assert zero["gain_best"] == zero["planner_best"] and zero["auc"] == 0.5


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    row = {"planner_best": 0.77, "gain_best": 0.83, "auc": 0.75}
    assert GC.reading(row).startswith("passes")
    assert "needs +0.05" in GC.reading(row | {"gain_best": 0.80})
    assert GC.reading(row | {"auc": 0.6}).startswith("fails: AUC")
    assert GC.reading(row | {"auc": None}) == "no data"


def test_check_end_to_end(labels: Path, tmp_path: Path) -> None:
    net = SetEncoderNet(PPOConfig.from_dict(GC.rc.sc._load("agent/ppo.yaml")).net)
    agent = tmp_path / "agent.pt"
    PPOAgent(net, torch.device("cpu")).save(agent)
    out = tmp_path / "table.json"
    GC.main(["part=check", f"agent={agent}", f"labels={labels}", "workers=1", f"out={out}"])
    table = json.loads(out.read_text())
    assert set(table) == {"small"} and table["small"]["positions"] > 0
