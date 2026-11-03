"""`scripts/head_check.py` (D-54, Q-126): the move-reading head sees the candidates, the ranking
loss prefers the right order, chunking does not change a step, and the check runs end to end on
labels from `scaling_check.py`."""

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
from meda.agents.ppo.policy import PPOAgent, batch_obs
from meda.agents.ppo.ppo import PPOConfig
from meda.agents.ppo.rollout import pad_obs
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


HC = load("head_check")


def tiny(seed: int) -> Iterator[Instance]:
    i = 0
    while True:
        inst = TINY.instances[i % len(TINY.instances)]
        yield Instance(
            f"scale-{seed}-{i}", inst.fluids, inst.targets, inst.chip, inst.step_limit, {}
        )
        i += 1


@pytest.fixture(scope="module", autouse=True)
def registered_scaling_check() -> Iterator[None]:
    """`test_scaling_check.py` loads its own copy of `scaling_check`; workers pickle functions
    by the registered module's name, so use whichever copy is registered."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HC, "sc", sys.modules["scaling_check"])
        yield


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    net = SetEncoderNet(PPOConfig.from_dict(HC.sc.sc._load("agent/ppo.yaml")).net)
    path = tmp_path_factory.mktemp("agent") / "agent.pt"
    PPOAgent(net, torch.device("cpu")).save(path)
    return path


@pytest.fixture(scope="module")
def labelled(agent: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("labels")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HC.sc, "problems", tiny)
        HC.sc.main(
            ["part=label", f"agent={agent}", f"out={out}", "held=3", "sizes=60", "workers=2"]
        )
    return out


@pytest.fixture(scope="module")
def data(labelled: Path) -> Any:
    return HC.load(str(labelled / "train_60.jsonl"), 2, 0.01)


def test_positions_keep_parents_and_groups(data: Any) -> None:
    n = len(data.obs)
    assert n == len(data.target) == len(data.spent) == len(data.bound) == len(data.parent)
    groups = data.groups()
    assert groups and all(len(g) >= 2 for g in groups)
    for g in groups:
        assert len({data.parent[int(i)] for i in g}) == 1  # siblings share a parent
        assert all(data.parent[int(i)] < int(i) for i in g)


def test_only_the_move_head_reads_the_candidates(agent: Path, data: Any) -> None:
    net = PPOAgent.load(agent).net
    i = int(np.argmax(data.widths()))
    obs = pad_obs([data.obs[i]])
    changed = {k: v.copy() for k, v in obs.items()}
    changed["candidates"] = changed["candidates"] + 1.0
    for moves, differs in ((False, False), (True, True)):
        head = HC.Head(net, moves, start=-1.0).eval()
        with torch.no_grad():
            a = head(batch_obs(obs, torch.device("cpu")))
            b = head(batch_obs(changed, torch.device("cpu")))
        assert bool(torch.isfinite(a).all())
        assert (not torch.allclose(a, b)) == differs


def test_rank_loss_prefers_the_right_order() -> None:
    spent = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([12.0, 8.0, 11.0])  # child 1 is cheapest (9)
    slices = [slice(0, 3)]

    def loss(pred: list[float]) -> float:
        v = -torch.tensor(pred) * 0.01
        return float(HC.rank_loss(v, spent, target, slices, 1.0, 0.01))

    assert loss([12.0, 8.0, 11.0]) < loss([5.0, 8.0, 11.0])
    # only differences inside a group matter
    assert loss([12.0, 8.0, 11.0]) == pytest.approx(loss([112.0, 108.0, 111.0]), abs=1e-5)
    tie = HC.rank_loss(-torch.zeros(2), torch.zeros(2), torch.ones(2), [slice(0, 2)], 1.0, 0.01)
    assert float(tie) == pytest.approx(np.log(2), abs=1e-6)


def test_chunking_does_not_change_a_step(agent: Path, data: Any) -> None:
    idx = np.arange(min(12, len(data.obs)))
    target = -torch.as_tensor(data.target, dtype=torch.float32)[idx] * 0.01
    params = []
    for rows in (10**9, 1):  # one chunk, then one observation per chunk
        torch.manual_seed(0)
        head = HC.Head(PPOAgent.load(agent).net, True, start=-1.0)
        opt = torch.optim.SGD(head.parameters(), lr=0.1)
        HC.step(
            head,
            opt,
            data,
            idx,
            lambda v: ((v - target) ** 2).mean(),
            torch.device("cpu"),
            rows,
        )
        params.append(torch.cat([p.detach().flatten() for p in head.parameters()]))
    assert torch.allclose(params[0], params[1], atol=1e-5)


def test_top2_and_reading() -> None:
    rows = {
        "spent": np.array([0.0, 0.0, 1.0, 2.0]),
        "target": np.array([10.0, 12.0, 8.0, 11.0]),
        "pred": np.array([10.0, 9.0, 8.5, 20.0]),  # head's order: child 1, child 2
    }
    assert HC.top2(rows, [np.array([1, 2, 3])]) == 1.0
    rows["pred"] = np.array([10.0, 9.0, 30.0, 8.0])  # head's order: child 1, child 3
    assert HC.top2(rows, [np.array([1, 2, 3])]) == 0.0
    assert HC.reading({"global": {"head_best": 0.4}})[0].startswith("no arm")
    assert HC.reading({"rank": {"head_best": 0.6}, "global": {"head_best": 0.4}}) == [
        "rank: useful (picks the cheapest sibling 0.60)"
    ]


def test_check_end_to_end(labelled: Path, agent: Path, tmp_path: Path) -> None:
    HC.main(
        [
            f"train={labelled / 'train_60.jsonl'}",
            f"held={labelled / 'held.jsonl'}",
            f"init={agent}",
            "arms=global,moves_rank",
            "epochs=1",
            "workers=1",
            "threads=1",
            f"out={tmp_path}",
        ]
    )
    table = json.loads((tmp_path / "table.json").read_text())
    assert list(table) == ["global", "moves_rank"]
    assert all(row["groups"] > 0 for row in table.values())
    assert (tmp_path / "moves_rank.pt").exists()
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
