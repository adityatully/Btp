"""`scripts/dead_ends.py` (Q-115): the practice search's counts cover what `samples` keeps, the
replay of a no-planner game scores every position it reached, and the reading follows the rule
fixed in advance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from meda.agents.ppo.networks import SetEncoderNet
from meda.agents.ppo.policy import PPOAgent
from meda.agents.ppo.ppo import PPOConfig
from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet

ROOT = Path(__file__).resolve().parents[2]
TINY = BenchmarkSet.read("tiny_exact_v1", BENCHMARKS_DIR)


def dead_module() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    import search_check  # noqa: F401  (registers the dataclass module first)

    spec = importlib.util.spec_from_file_location("dead_ends", ROOT / "scripts/dead_ends.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dead_ends"] = module
    spec.loader.exec_module(module)
    return module


DE = dead_module()


@pytest.mark.parametrize("instance", TINY.instances[:2], ids=lambda i: i.instance_id)
def test_counts_cover_what_samples_keeps(instance: Any) -> None:
    import numpy as np

    from meda.analysis.evaluate import build_env

    search = DE.CountingSearch(build_env(DE._cfg(), [instance]), instance, None, 1e18, collect=4)
    search.run(8)
    kept = search.samples(10**6, np.random.default_rng(0))
    assert search.counts["kept"] == len(kept)
    assert sum(search.counts.values()) > 0


@pytest.fixture(scope="module")
def agent_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    net = SetEncoderNet(PPOConfig.from_dict(DE.sc._load("agent/ppo.yaml")).net)
    path = tmp_path_factory.mktemp("agent") / "agent.pt"
    PPOAgent(net, torch.device("cpu")).save(path)
    return str(path)


def test_replay_scores_every_position_of_a_no_planner_game(agent_path: str) -> None:
    inst = TINY.instances[0]
    task = {"set": "C3", "id": inst.instance_id, "instance": inst.to_json()}
    opts = {"agent": agent_path, "seconds": 60.0, "sims": 4, "reward_scale": 0.01}
    rec = DE.run_zero((task, opts))
    assert rec["end"] in {"done", "dead end", "step limit", "stopped"}
    assert len(rec["positions"]) == rec["steps"]
    assert rec["positions"][0][1] is not None  # the planner finishes the start of a tiny problem
    assert all(head >= 0 for head, _ in rec["positions"])
    json.dumps(rec)


def test_cliff_is_the_jump_at_the_first_unfinishable_position() -> None:
    assert DE.cliff([[10.0, 12.0], [8.0, 9.0], [4.0, None], [2.0, None]]) == pytest.approx(0.5)
    assert DE.cliff([[10.0, 12.0], [8.0, 9.0]]) is None


def test_reading_follows_the_rule_fixed_in_advance(tmp_path: Path) -> None:
    practice = tmp_path / "p.jsonl"
    practice.write_text(json.dumps({"kept": 995, "dropped": 5, "ended": 3}) + "\n")
    assert DE.summarise(practice).endswith("False")
    practice.write_text(json.dumps({"kept": 900, "dropped": 100, "ended": 3}) + "\n")
    assert DE.summarise(practice).endswith("True")
    zero = tmp_path / "z.jsonl"
    blind = {"finished": False, "end": "step limit", "positions": [[10, 12], [9, None]]}
    zero.write_text(json.dumps(blind) + "\n")
    assert DE.summarise(zero).endswith("True")
    sees = {"finished": False, "end": "step limit", "positions": [[10, 12], [40, None]]}
    zero.write_text(json.dumps(sees) + "\n")
    assert DE.summarise(zero).endswith("False")


def test_labels_give_planner_cost_then_the_failure_penalty() -> None:
    path = ("a", "b", "c", "d")
    positions = [(5.0, 12.0), (5.0, 9.0), (4.0, None), (3.0, None)]
    got = DE.label(path, positions, 2.0)
    assert got == [
        ((), None, 12.0),
        (("a",), None, 9.0),
        (("a", "b"), None, 18.0),
        (("a", "b", "c"), None, 18.0),
    ]


def test_label_records_load_as_value_targets(agent_path: str, tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import expert_iteration as ei
    import imitation_check as ic

    inst = TINY.instances[0]
    opts = {"agent": agent_path, "seconds": 60.0, "sims": 4, "reward_scale": 0.01, "factor": 2.0}
    rec = DE.run_label((inst.to_json(), opts))
    assert rec["n_value"] > 0 and rec["n_policy"] == 0
    demos = tmp_path / "dead.jsonl"
    demos.write_text(json.dumps(rec) + "\n")
    [(loaded, samples)] = ei.load_targets(demos)
    store = ic.SearchTargets()
    store.add_problem(DE.build_env(DE._cfg(), [loaded]), loaded, samples, 0.01)
    assert len(store) == rec["n_value"]
    assert all(p is None for p in store.pi)
