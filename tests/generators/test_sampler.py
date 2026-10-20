"""Sampler, synthetic recipes and curriculum (SPEC 8.1-8.3; milestone M4).

The M4 acceptance criteria live here: the same seed gives identical instances, every sampled
instance is solvable, and training draws can be kept off a frozen set.
"""

from __future__ import annotations

import numpy as np
import pytest

from meda.core.composition import Composition, Rational
from meda.core.dag import MixingDAG, validate
from meda.core.fluids import FluidKind
from meda.core.problem import Instance, Target
from meda.core.rules import replay
from meda.envs.encoding import Encoder
from meda.envs.sample_prep_env import EnvConfig
from meda.generators.benchmark_io import instance_hash
from meda.generators.curriculum import Curriculum, Stage, StageStats
from meda.generators.sampler import (
    GeneratorConfig,
    dilution_grid,
    make_fluids,
    sample_instance,
    sample_instances,
    snap_to_denominator,
)
from meda.generators.synthetic import (
    Blend,
    Draw,
    composition_of,
    random_recipe,
    support_size_of,
    synthetic_instance,
    witness_plan,
)
from meda.solvers.baselines.naive import naive_plan
from tests.core.helpers import SB, chip, default_weights
from tests.generators.helpers import STAGE_NAMES, env_sizes, load_stage, stage_dict, stage_files

stages = pytest.mark.parametrize("name", STAGE_NAMES)
WEIGHTS = default_weights()


def witness_of(instance: Instance) -> tuple[object, ...]:
    """Re-derive the instance's witness plan, from its recipes or the naive construction."""
    from meda.generators.synthetic import recipe_from_dict

    recipes = instance.metadata.get("recipes")
    if recipes is not None:
        return witness_plan(instance, [recipe_from_dict(r) for r in recipes])
    plan = naive_plan(instance)
    assert plan is not None, f"no naive plan for {instance.instance_id}"
    return plan


# --------------------------------------------------------------------------- config


def test_every_stage_config_loads() -> None:
    assert {p.stem for p in stage_files()} == set(STAGE_NAMES)
    for name in STAGE_NAMES:
        cfg, _ = load_stage(name)
        assert cfg.name == name
        assert cfg.config_hash() == load_stage(name)[0].config_hash()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"composition_sampler": "magic"}, "composition_sampler"),
        ({"support_size": 1}, "support_size"),
        ({"support_size": 9}, "support_size"),
        ({"accuracy_denominator": 1}, "accuracy_denominator"),
        ({"demand": [3, 1]}, "demand"),
        ({"droplet_volume": 0}, "droplet_volume"),
        ({"tolerance": -1}, "tolerance"),
        ({"num_fluids": {"samples": 0, "buffers": 2}}, "SAMPLE"),
        ({"num_fluids": {"samples": 1, "buffers": 0}}, "at least 2 fluids"),
    ],
)
def test_bad_stage_config_raises(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GeneratorConfig.from_dict(stage_dict("tiny_dilution", **overrides))


def test_unknown_config_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        GeneratorConfig.from_dict(stage_dict("tiny_dilution", mystery=1))


def test_fixed_list_needs_compositions() -> None:
    with pytest.raises(ValueError, match="fixed_list"):
        GeneratorConfig.from_dict(stage_dict("tiny_dilution", composition_sampler="fixed_list"))


@stages
def test_every_stage_fits_the_env_config(name: str) -> None:
    """The env padding must cover the largest instance a stage can make (SPEC 9.2)."""
    cfg, chip_limits = load_stage(name)
    sizes = env_sizes()
    assert cfg.max_fluids <= sizes["F_pad"], f"{name}: F_pad too small"
    max_targets = cfg.num_experiments[1] * cfg.targets_per_experiment[1]
    assert max_targets <= sizes["T_pad"], f"{name}: T_pad too small for {max_targets} targets"
    max_storage = cfg.storage_capacity[1] if cfg.storage_capacity else chip_limits.storage_capacity
    assert max_storage <= sizes["N_pad"], f"{name}: N_pad too small"


@stages
def test_existing_stage_hashes_are_unchanged(name: str) -> None:
    """The chip-input keys are written into the config hash only when set, so stages that do
    not use them keep the hash that names their instances in frozen sets."""
    if name == "tiny_dilution":
        assert load_stage(name)[0].config_hash() == "686c78262fb2b9da"  # val_stage1_v2


def ranged_dilution(**overrides: object) -> GeneratorConfig:
    ranges = {"max_mix_volume": [4, 10], "storage_capacity": [3, 8], "droplet_volume": [1, 2]}
    return GeneratorConfig.from_dict(stage_dict("dilution_multi", **{**ranges, **overrides}))


def test_chip_inputs_and_droplet_volume_are_drawn_per_instance() -> None:
    """SPEC 1.3: M, storage capacity and droplet volume vary per instance within the ranges,
    every draw is witnessed, and the same seed gives the same instances."""
    cfg, base = ranged_dilution(), chip("meda_m8")
    drawn = sample_instances(cfg, base, np.random.default_rng(4), 40, WEIGHTS)
    ms = {i.chip.max_mix_volume for i in drawn}
    caps = {i.chip.storage_capacity for i in drawn}
    volumes = {t.droplet_volume for i in drawn for t in i.targets}
    assert len(ms) >= 4 and ms <= set(range(4, 11))
    assert len(caps) >= 3 and caps <= set(range(3, 9))
    assert volumes == {1, 2}
    for inst in drawn:
        m, cap = inst.chip.max_mix_volume, inst.chip.storage_capacity
        assert inst.chip.name == f"meda_m8-M{m}-S{cap}"
        assert inst.chip.min_mix_volume == base.min_mix_volume  # platform constants untouched
        final, _, _ = replay(inst, witness_of(inst))
        assert final.is_success, inst.instance_id
    again = sample_instances(cfg, base, np.random.default_rng(4), 40, WEIGHTS)
    assert [i.to_json() for i in again] == [i.to_json() for i in drawn]


def test_droplet_volume_above_the_drawn_m_raises() -> None:
    cfg = ranged_dilution(max_mix_volume=[2, 2], droplet_volume=[3, 3])
    with pytest.raises(ValueError, match="droplet_volume 3 > max_mix_volume 2"):
        sample_instance(cfg, chip("meda_m8"), np.random.default_rng(0), WEIGHTS)


@pytest.mark.parametrize(
    ("key", "value"), [("max_mix_volume", [1, 4]), ("storage_capacity", [5, 2])]
)
def test_bad_chip_input_ranges_raise(key: str, value: list[int]) -> None:
    with pytest.raises(ValueError, match=key):
        GeneratorConfig.from_dict(stage_dict("dilution_multi", **{key: value}))


# --------------------------------------------------------------------------- M4 acceptance


@stages
def test_same_seed_gives_identical_instances(name: str) -> None:
    cfg, chip_limits = load_stage(name)
    first = sample_instances(cfg, chip_limits, np.random.default_rng(7), 4, WEIGHTS)
    second = sample_instances(cfg, chip_limits, np.random.default_rng(7), 4, WEIGHTS)
    assert first == second
    assert [i.instance_id for i in first] == [i.instance_id for i in second]
    assert [dict(i.metadata) for i in first] == [dict(i.metadata) for i in second]
    other = sample_instances(cfg, chip_limits, np.random.default_rng(8), 4, WEIGHTS)
    assert [i.instance_id for i in other] != [i.instance_id for i in first]


@stages
def test_every_sampled_instance_is_solvable(name: str) -> None:
    """Acceptance: the witness is a legal plan that meets all demand and validates as a DAG."""
    cfg, chip_limits = load_stage(name)
    for inst in sample_instances(cfg, chip_limits, np.random.default_rng(3), 5, WEIGHTS):
        plan = witness_of(inst)
        assert len(plan) == inst.metadata["witness_actions"]
        assert len(plan) <= inst.step_limit
        state, totals, deltas = replay(inst, plan)  # type: ignore[arg-type]
        assert state.is_success
        dag = MixingDAG.from_deltas(inst, deltas, totals, "witness")
        validate(dag, inst).raise_if_invalid()
        assert Rational.parse(inst.metadata["known_upper_bound"]) == WEIGHTS.scalar(
            totals, inst.fluids
        )


@stages
def test_sampled_instances_respect_their_stage(name: str) -> None:
    cfg, chip_limits = load_stage(name)
    for inst in sample_instances(cfg, chip_limits, np.random.default_rng(11), 5, WEIGHTS):
        assert cfg.min_fluids <= inst.num_fluids <= cfg.max_fluids
        assert inst.fluids[0].kind is FluidKind.SAMPLE
        assert inst.chip == chip_limits
        experiments = {t.experiment_id for t in inst.targets}
        assert len(experiments) <= cfg.num_experiments[1]
        assert cfg.num_experiments[0] <= len(experiments)
        for t in inst.targets:
            assert cfg.demand[0] <= t.demand <= cfg.demand[1]
            assert cfg.droplet_volume[0] <= t.droplet_volume <= cfg.droplet_volume[1]
            assert t.tolerance == cfg.tolerance
            assert t.composition.den <= chip_limits.max_denominator
            assert support_size_of(t.composition) >= min(cfg.support_size, inst.num_fluids)
            assert t.composition.num[0] > 0 or support_size_of(t.composition) >= 2


@stages
def test_sampled_instances_run_in_the_encoder(name: str) -> None:
    cfg, chip_limits = load_stage(name)
    sizes = env_sizes()
    encoder = Encoder(EnvConfig.from_dict(sizes).sizes)
    for inst in sample_instances(cfg, chip_limits, np.random.default_rng(5), 5, WEIGHTS):
        encoder.check_instance(inst)


def test_exclude_keeps_training_off_a_frozen_set() -> None:
    """Acceptance: no overlap between training draws and a frozen set."""
    cfg, chip_limits = load_stage("tiny_dilution")
    rng = np.random.default_rng(0)
    held_out = {instance_hash(i) for i in sample_instances(cfg, chip_limits, rng, 4, WEIGHTS)}
    drawn = sample_instances(
        cfg, chip_limits, np.random.default_rng(1), 6, WEIGHTS, exclude=held_out
    )
    assert held_out.isdisjoint({instance_hash(i) for i in drawn})


def test_exhausted_stage_raises_instead_of_looping() -> None:
    """Stage 1 holds 31 distinct problems (the reduced fractions n/32); asking for more must say
    so instead of looping or repeating silently (rule 9)."""
    cfg, chip_limits = load_stage("tiny_dilution")
    assert len(sample_instances(cfg, chip_limits, np.random.default_rng(0), 31, WEIGHTS)) == 31
    with pytest.raises(RuntimeError, match="too small to sample"):
        sample_instances(cfg, chip_limits, np.random.default_rng(0), 32, WEIGHTS)
    everything = {
        instance_hash(i)
        for i in sample_instances(cfg, chip_limits, np.random.default_rng(0), 31, WEIGHTS)
    }
    with pytest.raises(RuntimeError, match="excluded set"):
        sample_instance(cfg, chip_limits, np.random.default_rng(0), exclude=everything)


def test_unwitnessable_stage_raises_with_guidance() -> None:
    """A denominator the chip cannot factorise has no naive witness (rule 9)."""
    cfg = GeneratorConfig.from_dict(
        stage_dict("tiny_dilution", accuracy_denominator=7, support_size=2)
    )
    with pytest.raises(ValueError, match="no feasibility witness"):
        sample_instance(cfg, chip("dmfb_m2"), np.random.default_rng(0))


# --------------------------------------------------------------------------- composition samplers


def test_snap_to_denominator_hits_the_support_and_accuracy() -> None:
    cfg = GeneratorConfig.from_dict(
        stage_dict("mixture_multi", composition_sampler="dirichlet_snapped")
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        comp = snap_to_denominator(rng, cfg, 4)
        assert support_size_of(comp) == cfg.support_size
        assert comp.num[0] > 0  # fluid 0 (a SAMPLE) is always in the support
        assert cfg.accuracy_denominator % comp.den == 0


def test_snap_rejects_support_above_accuracy() -> None:
    cfg = GeneratorConfig.from_dict(
        stage_dict("mixture_multi", accuracy_denominator=2, composition_sampler="dirichlet_snapped")
    )
    with pytest.raises(ValueError, match="support_size"):
        snap_to_denominator(np.random.default_rng(0), cfg, 4)


def test_fixed_list_sampler() -> None:
    cfg = GeneratorConfig.from_dict(
        stage_dict(
            "tiny_dilution",
            composition_sampler="fixed_list",
            fixed_compositions=[[1, 3], [1, 1]],
        )
    )
    allowed = {Composition.from_ratio((1, 3)), Composition.from_ratio((1, 1))}
    for inst in sample_instances(cfg, chip("dmfb_m2"), np.random.default_rng(0), 2, WEIGHTS):
        assert {t.composition for t in inst.targets} <= allowed


def test_make_fluids_orders_kinds() -> None:
    cfg, _ = load_stage("mixture_multi")
    fluids = make_fluids(cfg, np.random.default_rng(0))
    kinds = [f.kind for f in fluids]
    assert kinds[0] is FluidKind.SAMPLE
    assert kinds[-1] is FluidKind.BUFFER
    assert [f.id for f in fluids] == list(range(len(fluids)))
    assert kinds == sorted(kinds, key=[FluidKind.SAMPLE, FluidKind.REAGENT, FluidKind.BUFFER].index)


# --------------------------------------------------------------------------- recipes


def test_recipe_composition_and_volume() -> None:
    recipe = Blend(Blend(Draw(0), 2, Draw(1), 3), 3, Draw(1), 1)  # the SPEC 17.3 case-3 plan
    assert composition_of(recipe, 2) == Composition.from_ints((3, 7), 10)
    target = Target(0, 0, Composition.from_ints((3, 7), 10), Rational.of(0), 1, 1)
    inst = Instance("recipe-3-10", SB, (target,), chip(), 20)
    plan = witness_plan(inst, [recipe])
    state, totals, deltas = replay(inst, plan)
    assert state.is_success
    assert totals.mix_ops == 2
    validate(MixingDAG.from_deltas(inst, deltas, totals, "recipe"), inst).raise_if_invalid()


@pytest.mark.parametrize("chip_name", ["dmfb_m2", "meda_m8"])
def test_random_recipes_are_reachable(chip_name: str) -> None:
    chip_limits = chip(chip_name)
    rng = np.random.default_rng(0)
    for _ in range(25):
        recipe = random_recipe(rng, 3, chip_limits, num_mixes=3)
        comp = composition_of(recipe, 3)
        assert comp.den <= chip_limits.max_denominator
        assert sum(comp.num) == comp.den


def test_random_recipe_rejects_zero_mixes() -> None:
    with pytest.raises(ValueError, match="num_mixes"):
        random_recipe(np.random.default_rng(0), 2, chip(), num_mixes=0)


def test_synthetic_instance_has_a_known_upper_bound() -> None:
    inst, recipes = synthetic_instance(
        "synth-1", SB, chip(), np.random.default_rng(0), num_targets=2, num_mixes=2, demand=2
    )
    plan = witness_plan(inst, recipes)
    state, _, _ = replay(inst, plan)
    assert state.is_success
    assert inst.metadata["witness_actions"] == len(plan)
    assert inst.step_limit >= len(plan)
    for target, recipe in zip(inst.targets, recipes, strict=True):
        assert target.composition == composition_of(recipe, 2)


def test_witness_plan_reports_a_mismatched_recipe() -> None:
    inst, _ = synthetic_instance(
        "synth-2", SB, chip(), np.random.default_rng(1), num_targets=1, num_mixes=1
    )
    wrong = Blend(Draw(0), 1, Draw(1), 7)
    if composition_of(wrong, 2) != inst.targets[0].composition:
        with pytest.raises(ValueError, match="outside tolerance"):
            witness_plan(inst, [wrong])


# --------------------------------------------------------------------------- curriculum


def make_curriculum(window: int = 4, reference: Rational | None = None) -> Curriculum:
    stage_configs = [load_stage(n) for n in ("tiny_dilution", "dilution_single")]
    stages_list = [
        Stage(cfg, chip_limits, min_success_rate=0.75, window=window, reference_cost=reference)
        for cfg, chip_limits in stage_configs
    ]
    return Curriculum(stages_list, np.random.default_rng(0), WEIGHTS)


def test_curriculum_promotes_on_success_rate() -> None:
    cur = make_curriculum(window=4)
    assert cur.stage.name == "tiny_dilution"
    for _ in range(3):
        cur.record({"success": True, "plan_cost": 10.0})
    assert cur.stage.name == "tiny_dilution"  # window not full yet
    cur.record({"success": True, "plan_cost": 10.0})
    assert cur.stage.name == "dilution_single"
    assert cur.promotions[0]["from"] == "tiny_dilution"


def test_curriculum_holds_on_failures_and_on_cost() -> None:
    cur = make_curriculum(window=4)
    for success in (True, False, False, True):
        cur.record({"success": success, "plan_cost": 10.0})
    assert cur.stage.name == "tiny_dilution"

    costly = make_curriculum(window=2, reference=Rational.of(10))
    for _ in range(2):
        costly.record({"success": True, "plan_cost": 100.0})
    assert costly.stage.name == "tiny_dilution"  # successful but far above the reference
    for _ in range(2):
        costly.record({"success": True, "plan_cost": 10.0})
    assert costly.stage.name == "dilution_single"


def test_curriculum_stops_at_the_last_stage() -> None:
    cur = make_curriculum(window=1)
    for _ in range(10):
        cur.record({"success": True, "plan_cost": 1.0})
    assert cur.stage.name == "dilution_single"
    assert cur.finished_all
    assert len(cur.promotions) == 1


def test_curriculum_source_draws_from_the_current_stage() -> None:
    cur = make_curriculum(window=1)
    inst = cur.source()(np.random.default_rng(0))
    assert inst.chip.name == "dmfb_m2"
    cur.record({"success": True, "plan_cost": 1.0})
    assert cur.source()(np.random.default_rng(0)).chip.name == "meda_m8"


def test_curriculum_state_round_trip() -> None:
    cur = make_curriculum(window=1)
    cur.record({"success": True, "plan_cost": 1.0})
    restored = make_curriculum(window=1)
    restored.load_state_dict(cur.state_dict())
    assert restored.stage.name == cur.stage.name
    with pytest.raises(ValueError, match="stage_index"):
        restored.load_state_dict({"stage_index": 9})


def test_curriculum_record_needs_a_finished_episode() -> None:
    cur = make_curriculum()
    with pytest.raises(KeyError, match="success"):
        cur.record({"legal_count": 3})


def test_stage_stats_window() -> None:
    stats = StageStats(window=3)
    for _ in range(5):
        stats.record(True, 2.0)
    assert stats.finished == 3
    assert stats.success_rate == 1.0
    assert stats.mean_success_cost == 2.0
    stats.record(False, float("nan"))
    assert stats.success_rate == pytest.approx(2 / 3)


def test_curriculum_needs_stages() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        Curriculum([], np.random.default_rng(0))


# --------------------------------------------------------------------------- dilution_grid


def test_dilution_grid_is_deduplicated() -> None:
    grid = dilution_grid(chip("dmfb_m2"), 8, step_limit=20)
    comps = [i.targets[0].composition for i in grid]
    assert len(set(comps)) == len(comps) == 7  # n/8 for n = 1..7, all already reduced
    assert dilution_grid(chip(), 16, 20) == dilution_grid(chip(), 16, 20)
    with pytest.raises(ValueError, match="denominator"):
        dilution_grid(chip(), 1, 20)


# --------------------------------------------------------------------------- M4 acceptance (slow)


@pytest.mark.slow
@stages
def test_many_sampled_instances_are_all_solvable(name: str) -> None:
    """Acceptance at scale: 200 instances per stage, every one with a validated witness."""
    cfg, chip_limits = load_stage(name)
    rng = np.random.default_rng(0)
    for n in range(200):
        inst = sample_instance(cfg, chip_limits, rng, WEIGHTS)
        plan = witness_of(inst)
        state, totals, deltas = replay(inst, plan)  # type: ignore[arg-type]
        assert state.is_success, f"{name} draw {n}: {inst.instance_id} unmet"
        assert len(plan) <= inst.step_limit
        validate(MixingDAG.from_deltas(inst, deltas, totals, "witness"), inst).raise_if_invalid()


def test_curriculum_never_draws_a_held_out_problem() -> None:
    """Training must not see frozen-set problems, including `tiny_exact_v1`'s, which differ
    from stage-1 draws only in `max_denominator`."""
    from meda.generators.benchmark_io import BENCHMARKS_DIR, BenchmarkSet, check_no_overlap

    frozen = [BenchmarkSet.read(n, BENCHMARKS_DIR) for n in ("tiny_exact_v1", "val_stage1_v1")]
    held_out: set[str] = set()
    for bench in frozen:
        held_out |= bench.hashes()
    stage_configs = [load_stage("tiny_dilution")]
    stages_list = [Stage(cfg, chip_limits) for cfg, chip_limits in stage_configs]
    cur = Curriculum(stages_list, np.random.default_rng(0), WEIGHTS, exclude=held_out)
    draw = cur.source()
    rng = np.random.default_rng(1)
    drawn = [draw(rng) for _ in range(200)]
    check_no_overlap(drawn, frozen)

    unguarded = Curriculum(stages_list, np.random.default_rng(0), WEIGHTS).source()
    leaked = [unguarded(rng) for _ in range(200)]
    with pytest.raises(ValueError, match="overlap"):
        check_no_overlap(leaked, frozen)
