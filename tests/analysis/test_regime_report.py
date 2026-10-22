"""`scripts/regime_report.py`: grading regime runs at their end or at a shared validation round."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
STAGES = ["s1", "s2"]
BATCH = 100  # env steps per update in these fake runs


def report_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "regime_report", ROOT / "scripts" / "regime_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(
    root: Path,
    regime: str,
    seed: int,
    rounds: list[dict[str, tuple[float, float | None]]],
    events: list[dict[str, Any]] = (),  # type: ignore[assignment]
    finished: bool = False,
) -> Path:
    """A run directory as `train_ppo_regime` leaves it: one validation round per update."""
    run = root / f"{regime}_s{seed}"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text(
        json.dumps(
            {
                "seed": seed,
                "train": {"source": "regime"},
                "stages": {"final_weights": {"s1": 1, "s2": 3}},
            }
        )
    )
    lines = []
    for update, results in enumerate(rounds, start=1):
        lines.append({"prefix": "train", "step": update, "global_step": update * BATCH})
        for stage, (success, gap) in results.items():
            lines.append(
                {
                    "prefix": f"eval/{stage}",
                    "step": update,
                    "success_rate": success,
                    "gap_to_best_known": gap,
                }
            )
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    state = {"regime": regime, "stages": STAGES, "events": list(events)}
    if finished:
        (run / "regime.json").write_text(json.dumps(state))
    else:
        (run / "checkpoints").mkdir()
        torch.save(
            {"extra": {"regime": state}, "global_step": len(rounds) * BATCH},
            run / "checkpoints" / "last.pt",
        )
    return run


def test_unfinished_runs_grade_at_a_shared_round(tmp_path: Path) -> None:
    rr = report_module()
    promoted = {"env_step": 100, "stage": "s1", "event": "promoted", "forced": False}
    later = {"env_step": 300, "stage": "s2", "event": "finished", "forced": True}
    fake_run(
        tmp_path,
        "expanding",
        0,
        [
            {"s1": (0.5, 0.4), "s2": (0.0, None)},
            {"s1": (1.0, 0.1), "s2": (0.9, 0.3)},
            {"s1": (0.8, 0.1), "s2": (1.0, 0.2)},
        ],
        events=[promoted, later],
    )
    fake_run(
        tmp_path,
        "mixed",
        0,
        [{"s1": (0.2, 0.5), "s2": (0.1, 0.9)}, {"s1": (0.6, 0.3), "s2": (0.5, 0.6)}],
    )
    dirs = rr.run_dirs([tmp_path])
    with pytest.raises(SystemExit, match="unfinished"):
        rr.read_runs(dirs, None)

    runs = {r["regime"]: r for r in rr.read_runs(dirs, "common")}  # mixed has reached step 200
    exp = runs["expanding"]
    assert exp["env_steps"] == runs["mixed"]["env_steps"] == 200
    assert exp["final"] == {"s1": (1.0, 0.1), "s2": (0.9, 0.3)}
    assert exp["weighted_gap"] == pytest.approx((1 * 0.1 + 3 * 0.3) / 4)
    assert exp["events"] == [promoted]  # the step-300 event is after the graded round
    assert exp["forgetting"] == {"s1": 0.0, "s2": 0.0}

    # At step 250 both runs grade at their round at 200; at 300 forgetting shows on s1.
    assert {r["env_steps"] for r in rr.read_runs(dirs, "250")} == {200}
    with pytest.raises(SystemExit, match="different validation rounds"):
        rr.read_runs(dirs, "300")
    alone = rr.read_run(dirs[0], 300)
    assert alone["forgetting"]["s1"] == pytest.approx(0.2)
    assert "env step 200" in rr.report(rr.read_runs(dirs, "common"), 0.95, "common")


def test_a_round_the_checkpoint_does_not_cover_is_refused(tmp_path: Path) -> None:
    rr = report_module()
    run = fake_run(tmp_path, "mixed", 1, [{"s1": (1.0, 0.0), "s2": (1.0, 0.0)}] * 2)
    torch.save(
        {
            "extra": {"regime": {"regime": "mixed", "stages": STAGES, "events": []}},
            "global_step": 100,
        },
        run / "checkpoints" / "last.pt",
    )
    with pytest.raises(SystemExit, match="covers env steps up to 100"):
        rr.read_run(run, 200)
    assert rr.read_run(run, 100)["env_steps"] == 100


def test_only_regime_runs_are_found(tmp_path: Path) -> None:
    rr = report_module()
    fake_run(tmp_path, "sequential", 0, [{"s1": (1.0, 0.0), "s2": (1.0, 0.0)}], finished=True)
    other = tmp_path / "plain"
    (other / ".hydra").mkdir(parents=True)
    (other / ".hydra" / "config.yaml").write_text(json.dumps({"train": {"source": "generator"}}))
    (other / "metrics.jsonl").write_text("")
    assert [d.name for d in rr.run_dirs([tmp_path])] == ["sequential_s0"]
