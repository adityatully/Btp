"""`scripts/zero_loop.py` (D-49): self-play keeps targets only on the positions played, from the
search's own outcomes (or the planner-free failure penalty); the records load as training
targets; the reading follows the rule fixed in advance."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
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


def load(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    import search_check  # noqa: F401  (registers the dataclass module first)

    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ZL = load("zero_loop")


@pytest.fixture(scope="module")
def agent_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    net = SetEncoderNet(PPOConfig.from_dict(ZL.sc._load("agent/ppo.yaml")).net)
    path = tmp_path_factory.mktemp("agent") / "agent.pt"
    PPOAgent(net, torch.device("cpu")).save(path)
    return str(path)


def opts(agent: str) -> dict[str, Any]:
    return {
        "agent": agent,
        "turn": 1,
        "sims": 8,
        "seconds": 60.0,
        "reward_scale": 0.01,
        "alpha": 0.3,
        "eps": 0.25,
    }


@pytest.mark.parametrize("instance", TINY.instances[:3], ids=lambda i: i.instance_id)
def test_targets_are_the_played_positions_only(agent_path: str, instance: Any) -> None:
    rec = ZL.play((instance.to_json(), opts(agent_path)))
    samples = ZL.ei._unpack(rec["samples"])
    samples = sorted(samples, key=lambda s: len(s[0]))
    paths = [actions for actions, _, _ in samples]
    assert [len(p) for p in paths] == list(range(len(paths)))  # one per move, from the start
    for a, b in itertools.pairwise(paths):
        assert b[: len(a)] == a  # each extends the one before: a single line of play
    penalty = rec["penalty"]
    assert penalty >= ZL._cfg()["env"]["failure_penalty_factor"] * rec["construction"]
    for _, visits, remaining in samples:
        assert visits is not None and sum(visits) > 0
        assert remaining == penalty or (0 <= remaining < penalty)
    if rec["finished"]:
        assert samples[0][2] <= rec["cost"] + 1e-9


def test_records_load_as_training_targets(agent_path: str, tmp_path: Path) -> None:
    ic = load("imitation_check")
    inst = TINY.instances[0]
    rec = ZL.play((inst.to_json(), opts(agent_path)))
    demos = tmp_path / "play1.jsonl"
    demos.write_text(json.dumps(rec) + "\n")
    [(loaded, samples)] = ZL.ei.load_targets(demos)
    store = ic.SearchTargets()
    store.add_problem(ZL.build_env(ZL._cfg(), [loaded]), loaded, samples, 0.01)
    assert len(store) == rec["n_value"] and all(p is not None for p in store.pi)


def test_the_same_turn_plays_the_same_game(agent_path: str) -> None:
    inst = TINY.instances[1]
    a = ZL.play((inst.to_json(), opts(agent_path)))
    b = ZL.play((inst.to_json(), opts(agent_path)))
    assert a["samples"] == b["samples"]


def test_practice_problems_avoid_test_problems_and_overlap() -> None:
    with pytest.raises(ValueError):
        ZL.problems(1, 101, 0)
    held = ZL.ei.search_test_hashes()
    probs = ZL.problems(1, 1, 5)
    assert probs and not {ZL.instance_hash(i) for i in probs} & held


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    first = {"below": 30, "unfinished": 20}
    assert ZL.reading(first, {"below": 36, "unfinished": 20}).startswith("climbs")
    assert ZL.reading(first, {"below": 35, "unfinished": 10}).startswith("learns to finish")
    assert ZL.reading(first, {"below": 40, "unfinished": 21}).startswith("does not climb")
    assert ZL.reading({"below": 2, "unfinished": 5}, {"below": 7, "unfinished": 5}).startswith(
        "climbs"
    )


def test_eval_stats_and_summary(tmp_path: Path) -> None:
    recs = [
        {"set": "A", "search_cost": 9.0, "planner": 10.0},
        {"set": "A", "search_cost": None, "planner": 10.0},
        {"set": "C3", "search_cost": 12.0, "planner": 10.0},
    ]
    st = ZL.eval_stats(recs)
    assert st["A"] == {"n": 2, "below": 1, "unfinished": 1, "ratio": 0.9}
    assert st["all"]["below"] == 1 and st["all"]["unfinished"] == 1
    assert st["B"]["ratio"] is None and not math.isnan(st["all"]["ratio"])
    (tmp_path / "loop.jsonl").write_text(
        "\n".join(json.dumps({"turn": t, **st}) for t in (0, 1)) + "\n"
    )
    text = ZL.summary_loop(tmp_path)
    assert "| 1 |" in text and "Reading" in text
