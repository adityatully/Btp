"""Stage design over constraint cells, the mixture source, gradient targets and the frozen
validation sets of `track2_v1` (SPEC 8.1, 8.3, 8.4)."""

from __future__ import annotations

import importlib.util
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meda.core.costs import CostWeights
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet, problem_hash
from meda.generators.sampler import GeneratorConfig, GeneratorSource, sample_instance
from meda.generators.stages import AXES, MixtureSource, StageDesign
from tests.core.helpers import CONFIGS, chip, load_yaml
from tests.generators.helpers import stage_dict

GEOMETRIC = CostWeights.from_dict(load_yaml("costs/geometric.yaml"))
ROOT = Path(__file__).resolve().parents[2]


def design_dict(**overrides: Any) -> dict[str, Any]:
    return {**load_yaml("stages/track2_v1.yaml"), **overrides}


def design() -> StageDesign:
    return StageDesign.from_dict(design_dict())


def test_each_stage_adds_one_axis_value_and_the_held_out_cell_is_never_a_stage_cell() -> None:
    d = design()
    assert [s.name for s in d.stages] == [
        "s1_dilution",
        "s2_mixture",
        "s3_demand",
        "s4_targets",
        "s5_storage",
    ]
    assert [len(s.cells) for s in d.stages] == [1, 1, 2, 4, 7]
    assert [c.name for c in d.held_out] == ["dilution-many-multi-limited"]
    names = [c.name for c in d.cells]
    assert len(names) == len(set(names)) == 15
    assert "dilution-many-multi-limited" not in names
    # Every combination of the four axes is either a stage cell or held out.
    assert len(names) + len(d.held_out) == 2 ** len(AXES)
    # A stage's cells all carry the value it adds.
    for s in d.stages[1:]:
        assert s.adds is not None
        axis, value = s.adds
        assert all(c.values[AXES.index(axis)] == value for c in s.cells)


def test_cell_configs_take_the_axis_values_and_overrides() -> None:
    d = design()
    cells = {c.name: c.config for c in d.cells}
    dil = cells["dilution-one-single-loose"]
    assert dil.max_fluids == 2 and dil.demand == (1, 1) and dil.storage_capacity == (32, 32)
    assert dil.target_structure == ("independent",)
    mix = cells["mixture-many-multi-limited"]
    assert mix.reagents == (1, 4) and mix.support_size == (3, 6)
    assert mix.targets_per_experiment == (2, 10) and mix.demand == (2, 4)
    assert mix.storage_capacity == (2, 16) and mix.max_mix_volume == (3, 10)
    assert cells["dilution-many-single-loose"].target_structure == ("independent", "gradient")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"surprise": 1}, "unknown keys"),
        ({"final_weights": {"s1_dilution": 1.0}}, "final_weights"),
        (
            {"start": {"fluids": "gel", "targets": "one", "demand": "single", "storage": "loose"}},
            "no value 'gel'",
        ),
    ],
)
def test_bad_designs_raise(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises((ValueError, KeyError), match=match):
        StageDesign.from_dict(design_dict(**overrides))


def test_a_pilot_keeps_the_first_stages_with_final_weights_renormalised() -> None:
    pilot = design().first(3)
    assert [s.name for s in pilot.stages] == ["s1_dilution", "s2_mixture", "s3_demand"]
    assert sum(pilot.final_weights.values()) == pytest.approx(1.0)
    assert pilot.final_weights["s3_demand"] == pytest.approx(0.2 / 0.3)
    with pytest.raises(ValueError, match="has 5 stages"):
        design().first(6)


def test_cell_weights_split_each_stage_evenly() -> None:
    d = design()
    w = d.cell_weights({"s3_demand": 0.5, "s4_targets": 0.5})
    assert w["dilution-one-multi-loose"] == w["mixture-one-multi-loose"] == 0.25
    assert w["mixture-many-single-loose"] == 0.125
    assert w["dilution-one-single-loose"] == 0.0


def test_mixture_source_draws_only_weighted_cells_and_never_a_held_out_problem() -> None:
    d = design()
    held = frozenset({problem_hash(i) for i in BenchmarkSet.read("val_s1_v2").instances})
    source = d.source({"s1_dilution": 0.25, "s2_mixture": 0.75}, chip(), GEOMETRIC, held)
    assert source.weights == {
        "dilution-one-single-loose": 0.25,
        "mixture-one-single-loose": 0.75,
    }
    rng = np.random.default_rng(0)
    drawn = [source(rng) for _ in range(80)]
    cells = Counter(i.metadata["cell"] for i in drawn)
    assert set(cells) == {"dilution-one-single-loose", "mixture-one-single-loose"}
    assert cells["mixture-one-single-loose"] > cells["dilution-one-single-loose"]
    assert all(problem_hash(i) not in held for i in drawn)
    again = d.source({"s1_dilution": 0.25, "s2_mixture": 0.75}, chip(), GEOMETRIC, held)
    rng = np.random.default_rng(0)
    assert [again(rng).to_json() for _ in range(80)] == [i.to_json() for i in drawn]
    with pytest.raises(ValueError, match="weight > 0"):
        MixtureSource(())


def test_every_cell_draws_witnessed_problems_in_the_1_3_ranges() -> None:
    rng = np.random.default_rng(1)
    for cell in design().cells:
        for _ in range(4):
            inst = GeneratorSource(cell.config, chip(), GEOMETRIC)(rng)
            assert 3 <= inst.chip.max_mix_volume <= 10
            assert all(t.droplet_volume <= inst.chip.max_mix_volume for t in inst.targets)
            final, _, _ = replay(inst, _witness(inst))
            assert final.is_success, (cell.name, inst.instance_id)


def _witness(inst: Any) -> Any:
    from meda.solvers.fallback import cheapest_construction

    found = cheapest_construction(inst, GEOMETRIC)
    assert found is not None
    return found[0]


# --------------------------------------------------------------------------- gradient targets


def gradient_config(**overrides: Any) -> GeneratorConfig:
    data = stage_dict(
        "dilution_multi",
        composition_sampler="dirichlet_snapped",
        accuracy_denominator=[8, 64],
        num_experiments=1,
        targets_per_experiment=[3, 6],
        max_mix_volume=[3, 8],
        target_structure="gradient",
        step_limit_rule="depth",
    )
    data.pop("step_limit_multiple")
    data.pop("walk_mixes", None)
    return GeneratorConfig.from_dict({**data, **overrides})


def test_a_gradient_is_one_reachable_denominator_with_evenly_spaced_sample_shares() -> None:
    cfg = gradient_config()
    rng = np.random.default_rng(2)
    for _ in range(20):
        inst = sample_instance(cfg, chip(), rng, GEOMETRIC)
        assert inst.metadata["target_structure"] == "gradient"
        den = math.lcm(*(t.composition.den for t in inst.targets))
        shares = sorted(t.composition.num[0] * den // t.composition.den for t in inst.targets)
        steps = {b - a for a, b in itertools.pairwise(shares)}
        assert len(steps) == 1 and shares[0] > 0 and shares[-1] < den
        final, _, _ = replay(inst, _witness(inst))
        assert final.is_success


def test_gradient_needs_a_plain_dilution_and_enters_the_hash_only_when_set() -> None:
    with pytest.raises(ValueError, match="gradient is a dilution"):
        gradient_config(num_fluids={"samples": 1, "reagents": 1, "buffers": 1})
    with pytest.raises(ValueError, match="target_structure"):
        gradient_config(target_structure="ladder")
    plain = GeneratorConfig.from_dict(stage_dict("dilution_multi"))
    explicit = GeneratorConfig.from_dict(
        stage_dict("dilution_multi", target_structure="independent")
    )
    assert plain.config_hash() == explicit.config_hash()
    assert (
        gradient_config().config_hash()
        != gradient_config(target_structure="independent").config_hash()
    )


# --------------------------------------------------------------------------- frozen sets


def make_benchmark() -> Any:
    spec = importlib.util.spec_from_file_location(
        "make_benchmark", ROOT / "scripts" / "make_benchmark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_design_sets_rebuild_identically_and_cover_their_cells() -> None:
    builders = make_benchmark().design_builders("track2_v1")
    d = design()
    expected = {s.val: s for s in d.stages}
    assert set(builders) == {*expected, d.held_out_val}
    for name in ("val_s1_v2", "val_s4_v2", d.held_out_val):
        stored = BenchmarkSet.read(name)
        rebuilt = builders[name]()
        assert [i.to_json() for i in rebuilt.instances] == [i.to_json() for i in stored.instances]
        assert rebuilt.reference == stored.reference
        cells = {c.name for c in (expected[name].cells if name in expected else d.held_out)}
        assert {i.metadata["cell"] for i in stored.instances} == cells
        for inst in stored.instances:
            ref = stored.reference[inst.instance_id]
            assert ref.lower_bound is None or ref.lower_bound < ref.cost


def test_design_sets_are_disjoint() -> None:
    d = design()
    names = [s.val for s in d.stages] + [d.held_out_val]
    seen: dict[str, str] = {}
    for name in names:
        for inst in BenchmarkSet.read(name).instances:
            h = problem_hash(inst)
            assert h not in seen, (name, seen.get(h))
            seen[h] = name
    assert (CONFIGS / "stages" / "track2_v1.yaml").exists()


def test_mixture_step_limits_leave_room_beyond_the_known_plan() -> None:
    """SPEC 8.1 `depth_support`: the base counts at least (required fluids - 1) mixes per
    delivery, and the limit is floored at WITNESS_FLOOR x the witness. Under plain `depth`, 13
    of 40 stage-2 validation problems allowed exactly the witness plan and nothing more."""
    from meda.generators.sampler import WITNESS_FLOOR, required_fluids, step_limit_base

    d = design()
    assert all(c.config.step_limit_rule == "depth_support" for c in d.cells)
    (cell,) = d.stage("s2_mixture").cells
    rng = np.random.default_rng(5)
    for _ in range(15):
        inst = sample_instance(cell.config, chip(d.chip), rng, GEOMETRIC)
        meta = inst.metadata
        assert meta["step_limit_rule"] == "depth_support"
        assert step_limit_base(inst, "depth_support") >= sum(
            t.demand * required_fluids(t) for t in inst.targets
        )
        assert step_limit_base(inst, "depth_support") >= step_limit_base(inst, "depth")
        assert inst.step_limit >= math.ceil(WITNESS_FLOOR * meta["witness_actions"])
        assert inst.step_limit == max(
            math.ceil(4 * meta["step_limit_base"]),
            math.ceil(WITNESS_FLOOR * meta["witness_actions"]),
        )
