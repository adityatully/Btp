"""Training metrics: TensorBoard, a JSON-lines file and one console line per update (SPEC 13).

`metrics.jsonl` (one object per update) is the record tests and analysis read; TensorBoard is
for watching a run. Weights & Biases stays off (`train.use_wandb`), as SPEC 3 allows.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONSOLE_KEYS = ("success_rate", "plan_cost", "episode_return", "entropy", "approx_kl", "value_loss")


class MetricsLogger:
    def __init__(self, out_dir: str | Path, tensorboard: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.out_dir / "metrics.jsonl"
        self.writer: Any = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.out_dir / "tensorboard"))

    def log(self, step: int, metrics: Mapping[str, float], prefix: str = "train") -> None:
        clean = {
            k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()
        }
        with self.jsonl.open("a") as f:
            f.write(json.dumps({"prefix": prefix, "step": step, **clean}) + "\n")
        if self.writer is not None:
            for key, value in clean.items():
                if value is not None:
                    self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def console(self, update: int, total: int, metrics: Mapping[str, float]) -> None:
        parts = [f"{k}={metrics[k]:.3g}" for k in CONSOLE_KEYS if k in metrics]
        sps = metrics.get("env_steps_per_second")
        if sps is not None:
            parts.append(f"sps={sps:.0f}")
        print(f"update {update}/{total}  " + "  ".join(parts), flush=True)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
