"""Generator-test helpers: load every real generator stage from `configs/`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from meda.core.chip import ChipLimits
from meda.generators.sampler import GeneratorConfig
from tests.core.helpers import CONFIGS, chip, load_yaml

STAGE_NAMES = (
    "tiny_dilution",
    "dilution_single",
    "dilution_multi",
    "mixture_single",
    "mixture_multi",
)


def stage_files() -> list[Path]:
    return sorted((CONFIGS / "generator").glob("*.yaml"))


def load_stage(name: str) -> tuple[GeneratorConfig, ChipLimits]:
    """A generator stage and the chip it runs on (honouring its `defaults` override)."""
    data = OmegaConf.to_container(
        OmegaConf.load(CONFIGS / "generator" / f"{name}.yaml"), resolve=True
    )
    assert isinstance(data, dict)
    chip_name = "meda_m8"
    for entry in data.pop("defaults", []) or []:
        if isinstance(entry, dict):
            for key, value in entry.items():
                if "chip" in str(key):
                    chip_name = str(value)
    return GeneratorConfig.from_dict(data), chip(chip_name)  # type: ignore[arg-type]


def stage_dict(name: str, **overrides: Any) -> dict[str, Any]:
    """The raw stage dict with `defaults` stripped, for config-validation tests."""
    data = OmegaConf.to_container(
        OmegaConf.load(CONFIGS / "generator" / f"{name}.yaml"), resolve=True
    )
    assert isinstance(data, dict)
    data.pop("defaults", None)
    data.update(overrides)
    return data  # type: ignore[return-value]


def env_sizes() -> dict[str, Any]:
    return load_yaml("env/sample_prep.yaml")
