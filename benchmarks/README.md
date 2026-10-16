# Benchmarks

Frozen, versioned instance sets (SPEC 8.4), one JSON file per set: `name`, `version`,
generator config, seeds and instances.

- Never regenerate a published version; create a new version instead.
- Evaluation uses only these frozen sets. Training draws from the sampler.

Sets are organised in tiers (SPEC 8.4). Each tier answers one question, and its numbers are not
used as evidence for another tier's question.

| Tier | Question | Sets |
|---|---|---|
| 0 Correctness | proved optimum reached? | `tiny_exact_v1` (built) |
| 1 In-distribution | training stage learned? | `val_stage1_v1` (naive costs only), `val_stage1_v2` (same instances: 3 optima, 5 intervals), `val_stage<k>_v<n>` |
| 2 Engine check (DMFB) | core matches published 1:1 results? never a "win" | `dmfb_sweep_v1` |
| 3 Literature | reproduce, then beat, MEDA methods on their own problems | `literature_<paper>_v1` |
| 4 Generalisation | unseen and larger problems | `unseen_ratios_v1`, `larger_v1` |
| 5 Frontier | MTMD + N >= 3 + demand + binding storage; scaling axes | `frontier_v1` |
| 6 Assays | ratios from real protocols | `assays_v1` |

Where exact search cannot finish, the answer key is an interval [lower bound, best known]
(SPEC 8.4, 10.2, 12.2).
