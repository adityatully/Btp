"""`scripts/scaling_check.py` (D-53, Q-124): labels are the planner finish at sibling groups on
the network's line of play; the training files are nested and load as copy-step targets; the
sibling test and the reading behave as fixed in advance; both parts run end to end."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from meda.agents.ppo.networks import SetEncoderNet
from meda.agents.ppo.policy import PPOAgent
from meda.agents.ppo.ppo import PPOConfig
from meda.analysis.evaluate import build_env
from meda.core.problem import Instance
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


SC = load("scaling_check")


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    net = SetEncoderNet(PPOConfig.from_dict(SC.sc._load("agent/ppo.yaml")).net)
    path = tmp_path_factory.mktemp("agent") / "agent.pt"
    PPOAgent(net, torch.device("cpu")).save(path)
    return path


def tiny(seed: int) -> Iterator[Instance]:
    """The tiny problems over and over, with ids that say which draw they came from."""
    i = 0
    while True:
        inst = TINY.instances[i % len(TINY.instances)]
        yield Instance(
            f"scale-{seed}-{i}", inst.fluids, inst.targets, inst.chip, inst.step_limit, {}
        )
        i += 1


@pytest.fixture(scope="module")
def labelled(agent: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("labels")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SC, "problems", tiny)
        SC.main(
            ["part=label", f"agent={agent}", f"out={out}", "held=3", "sizes=20,60", "workers=2"]
        )
    return out


def test_labels_are_the_planner_finish_in_sibling_groups(agent: Path) -> None:
    inst = TINY.instances[0]
    rec = SC.play((inst.to_json(), str(agent), 3, 4, 0))
    samples = SC.ei._unpack(rec["samples"])
    assert rec["positions"] == len(samples) > 0
    env = build_env(SC.fc._cfg(), [inst])
    for actions, visits, remaining in samples:
        assert visits is None
        env.reset(options={"instance": inst})
        for a in actions:
            env.step(env.legal.index(a))
        assert remaining == SC.rc.planner_finish(env, inst)
    lines = {a for a, _, _ in samples}
    parents = {a[:-1] for a in lines if a and a[:-1] in lines}
    assert parents  # at least one labelled position with labelled children
    assert all(sum(a[:-1] == p for a in lines if a) <= 4 for p in parents)


def test_training_files_are_nested_and_load_as_targets(labelled: Path) -> None:
    small, big = (
        [json.loads(x) for x in (labelled / f"train_{n}.jsonl").read_text().split("\n") if x]
        for n in (20, 60)
    )
    assert [r["id"] for r in small] == [r["id"] for r in big[: len(small)]]
    assert sum(r["positions"] for r in small[:-1]) < 20 <= sum(r["positions"] for r in small)
    assert sum(r["positions"] for r in big) >= 60
    held = [json.loads(x) for x in (labelled / "held.jsonl").read_text().split("\n") if x]
    assert len(held) == 3 and all(r["id"].startswith("scale-1-") for r in held)
    assert len(SC.ei.load_targets(labelled / "train_60.jsonl")) == len(big)


def test_a_stopped_run_resumes_where_it_stopped(
    labelled: Path, agent: Path, tmp_path: Path
) -> None:
    lines = (labelled / "train_60.jsonl").read_text().split("\n")
    (tmp_path / "held.jsonl").write_text((labelled / "held.jsonl").read_text())
    (tmp_path / "train_60.jsonl").write_text("\n".join(lines[:2]) + "\n" + lines[2][:30])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SC, "problems", tiny)
        SC.main(["part=label", f"agent={agent}", f"out={tmp_path}", "sizes=20,60", "workers=2"])
    for n in (20, 60):
        got = (tmp_path / f"train_{n}.jsonl").read_text()
        assert got == (labelled / f"train_{n}.jsonl").read_text()


def rows(pred: list[float]) -> dict[str, np.ndarray]:
    # one parent, three children: the planner's choice is child 2 (1 + 8)
    return {
        "parent": np.array([-1, 0, 0, 0], dtype=np.float64),
        "spent": np.array([0.0, 0.0, 1.0, 2.0]),
        "target": np.array([10.0, 12.0, 8.0, 11.0]),
        "pred": np.array(pred),
        "bound": np.array([5.0, 3.0, 4.0, 5.0]),
    }


def test_sibling_test_counts_the_planners_choice() -> None:
    right = SC.metrics(rows([10.0, 12.0, 8.0, 11.0]))
    assert right["head_best"] == 1.0 and right["head_err"] == 0.0 and right["groups"] == 1
    assert right["runner_gap"] == 3.0 and right["bound_best"] == 0.0
    assert right["chance_best"] == pytest.approx(1 / 3)
    assert SC.metrics(rows([10.0, 1.0, 8.0, 11.0]))["head_best"] == 0.0


def test_reading_follows_the_rule_fixed_in_advance() -> None:
    assert SC.reading([0.5, 0.56, 0.62, 0.7]).startswith("data is the bottleneck")
    assert SC.reading([0.5, 0.51, 0.5, 0.515]).startswith("not data")
    assert SC.reading([0.5, 0.52, 0.56, 0.58]).startswith("data helps, slowly")


def test_check_end_to_end(labelled: Path, agent: Path, tmp_path: Path) -> None:
    out = tmp_path / "table.json"
    SC.main(
        [
            "part=check",
            f"held={labelled / 'held.jsonl'}",
            f"agents={agent},{agent}",
            "workers=1",
            f"out={out}",
        ]
    )
    table = json.loads(out.read_text())
    assert list(table) == [agent.parent.name] and table[agent.parent.name]["positions"] > 0
