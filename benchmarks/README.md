# Benchmarks

Frozen, versioned instance sets (SPEC 8.4), one JSON file per set: `name`, `version`,
generator config, seeds and instances.

- Never regenerate a published version; create a new version instead.
- Evaluation uses only these frozen sets. Training draws from the sampler.

Planned: `tiny_exact_v1`, `val_<stage>_v1`, `unseen_ratios_v1`, `larger_v1`, `literature_v1`.
