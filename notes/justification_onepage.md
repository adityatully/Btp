# Every number in the method, on one page (2026-11-20)

How each number is justified in the paper (D-101): **(a) converged**: the curve flattens and our
value is at or past that point; **(b) a time trade**: a larger value is still cheaper in the
planner alone, but costs time and the full method does not need it; **(c) argued or cited**:
not a tunable number. Details: `justification_audit.md`; numbers: `paper_log.md`.

CPU-h ÷ workers = wall hours (18 free cores on coe9: 27 CPU-h ≈ 1.5 h).

## Done

| Number | Ours | Evidence | Case |
|---|---|---|---|
| Lookahead k (baselines) | 3 | Flattens at 2; k = 2-10 within 0.04%; k ≥ 4 tries every candidate (E-109) | (a) |
| Split-tree search budget | 20,000 | Never runs out on 322 val problems (E-109) | (a) |
| Recipes | tree, direct, one mix, chain (+ MRCM) | Each up to the chain needed; the chain gives 13-27% (E-109) | (a), MRCM recipe still open |
| Repeats per problem | 1 | Identical runs within 0.2% (E-106) | (a) |
| Tie order | candidate order | Random tie-breaking 3% dearer (E-106) | part of the method |

## Running tonight (`night1`)

| Number | Ours | Run | Turns into |
|---|---|---|---|
| Simulations per move | 256 | 32 … 2048, 900 s each | (a) if 256 is on the plateau |
| Chain depth | 3 | depth 4 inside the search (`leaf_chain4`) | (b) if the search does not gain; planner alone: 4 is 4.8% cheaper at 10x time (E-109) |

## Next, in order

| # | Number | Ours | Run | Cost |
|---|---|---|---|---|
| 1 | Time budget; verdict thresholds | 900 s; 10% losses | J0, J9: offline, from the final runs' records | minutes |
| 2 | Target orders | 4 rules | `more_orders4`, `more_orders12` inside the search (J7m). Planner alone: 4.2% from the best of 36 on > 4 targets (E-109) | ~3 h |
| 3 | C_PUCT | 1.5 | 0.25 … 8 at 256 sims (J2's second curve) | ~10 h, one night |
| 4 | Finisher switch | 20 s, 3 leaves, 0.5 | J5, only where it fires | ~4 h |
| 5 | Blend partners; split-tree steps; MRCM budget and recipe | 4; 4; 2,000,000 | J6 curves not run yet (MRCM base) | ~3-6 h |

## Argued or cited (no run planned)

| Number or choice | Ours | Argument |
|---|---|---|
| Fluid weights | 64 / 8 / 1 | From e-MRCM (cite); J8 sensitivity optional (~16 h) |
| Mix and waste weights | 1, 1 | Our choice; J8 sensitivity optional, or state it |
| Step limit floor | 2x witness | Set for the RL agent (SPEC 8.1); J8 optional |
| Value scale, floor −1, dead end −1, mean backup, first-play value, most-visited move | standard PUCT | Standard choices (cite AlphaZero / MCTS surveys); J3 (~14 h) only if a reviewer asks |
| Two-input k:l mixes, storage, partial takes | task definition | SPEC 1.3, SPEC 15 (platform) |
