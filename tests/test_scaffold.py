"""M0 checks: package layout imports, import-boundary rules, config composition, CLI entry point."""

import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import meda

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "meda"
CONFIGS = ROOT / "configs"

# SPEC rule 2.3: these packages must stay usable without any ML library.
ML_LIBS = {"torch", "tensorflow", "jax", "stable_baselines3", "sb3_contrib"}
ML_FREE_PACKAGES = ["core", "generators", "solvers"]


def all_modules() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(meda.__path__, prefix="meda.")]


@pytest.mark.parametrize("name", all_modules())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("package", ML_FREE_PACKAGES)
def test_no_ml_imports(package: str) -> None:
    for path in (SRC / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert not roots & ML_LIBS, f"{path} imports {roots & ML_LIBS}"


def test_config_composes_with_spec_chip_defaults() -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="config", overrides=["generator=dilution_multi"])
    chip = cfg.chip
    assert chip.name == "meda_m8"
    assert chip.max_mix_volume == 8
    assert chip.min_mix_volume == 2
    assert chip.unit_volume == 1
    assert chip.min_droplet_volume == 1
    assert chip.allow_partial_take is True
    assert chip.storage_capacity == 10
    assert chip.max_denominator == 256
    assert chip.allow_same_composition_merge is False


def test_tiny_dilution_selects_dmfb_m2_chip() -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="config", overrides=["generator=tiny_dilution"])
    assert cfg.chip.name == "dmfb_m2"
    assert cfg.chip.max_mix_volume == 2


@pytest.mark.parametrize("group", ["agent", "generator", "chip"])
def test_every_option_in_group_composes(group: str) -> None:
    for option in sorted((CONFIGS / group).glob("*.yaml")):
        with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
            cfg = compose(config_name="config", overrides=[f"{group}={option.stem}"])
        assert cfg[group].name == option.stem


@pytest.mark.parametrize("module", ["meda.train", "meda.evaluate_cli"])
def test_cli_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
