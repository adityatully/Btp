"""Chip limits, cost weights and instances built from config / JSON (SPEC 5.2, 5.3, 6.3)."""

from dataclasses import replace

import pytest

from meda.core.chip import ChipLimits
from meda.core.composition import Rational
from meda.core.costs import CostWeights, StepCost
from meda.core.problem import Instance
from tests.core.helpers import SB, chip, default_weights, dilution, load_yaml, mixture


@pytest.mark.parametrize("name", ["meda_m8", "dmfb_m2"])
def test_chip_configs_load(name: str) -> None:
    c = ChipLimits.from_dict(load_yaml(f"chip/{name}.yaml"))
    assert c.name == name
    assert c.max_mix_volume == (8 if name == "meda_m8" else 2)
    assert ChipLimits.from_dict(c.to_dict()) == c


def test_chip_rejects_bad_config() -> None:
    data = load_yaml("chip/meda_m8.yaml")
    with pytest.raises(ValueError, match="missing"):
        ChipLimits.from_dict({k: v for k, v in data.items() if k != "max_mix_volume"})
    with pytest.raises(ValueError, match="unknown"):
        ChipLimits.from_dict({**data, "extra": 1})
    with pytest.raises(TypeError, match="max_mix_volume"):
        ChipLimits.from_dict({**data, "max_mix_volume": 8.0})
    with pytest.raises(ValueError, match="max_mix_volume < min_mix_volume"):
        ChipLimits.from_dict({**data, "max_mix_volume": 1})


def test_cost_weights_are_exact() -> None:
    w = default_weights()
    assert (w.w_sample, w.w_reagent, w.w_buffer) == (
        Rational.of(10),
        Rational.of(2),
        Rational.of(1, 2),
    )
    assert (w.w_mix, w.w_waste) == (Rational.of(1), Rational.of(1))
    cost = StepCost((1, 3), 1, 3, 1)
    assert w.scalar(cost, SB) == Rational.of(31, 2)
    assert float(w.scalar(cost, SB)) == 15.5


def test_cost_weights_accept_fraction_strings_and_reject_negative() -> None:
    data = load_yaml("costs/default.yaml")
    data["w_fluid"]["BUFFER"] = "1/3"
    assert CostWeights.from_dict(data).w_buffer == Rational.of(1, 3)
    data["w_mix"] = -1
    with pytest.raises(ValueError, match="w_mix"):
        CostWeights.from_dict(data)


@pytest.mark.parametrize(
    "inst",
    [dilution([(1, 4), (3, 10)], demand=2), mixture([(2, 1, 1)], tolerance=Rational.of(1, 64))],
)
def test_instance_json_round_trip(inst: Instance) -> None:
    inst = replace(inst, metadata={"generator": "test", "seed": 3})
    back = Instance.from_json(inst.to_json())
    assert back == inst
    assert back.metadata == inst.metadata


def test_instance_validation_names_instance() -> None:
    inst = dilution([(1, 4)])
    with pytest.raises(ValueError, match="test-dilution"):
        replace(inst, step_limit=0)
    with pytest.raises(ValueError, match="test-dilution"):
        replace(inst, targets=(replace(inst.targets[0], demand=0),))
    with pytest.raises(ValueError, match="fluids"):
        replace(inst, chip=chip(), fluids=inst.fluids[:1])
