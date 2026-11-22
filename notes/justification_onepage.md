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
| Recipes (planner) | direct, one mix, chain, split tree, MRCM | Planner: removing chain / split tree / MRCM costs 13% / 19% / 21%. Method: no chain 6.1%, no MRCM 13.4%, no split tree = no plan on 77/110. Direct never legal, one mix ties a chain: argued (E-109, E-113, E-114) | (a) + (c) |
| Repeats per problem | 1 | Identical runs within 0.2% (E-106) | (a) |
| Tie order | candidate order | Random tie-breaking 3% dearer (E-106) | part of the method |
| Target orders | 4 rules | Planner alone: 4.2% from the best of 36 (E-109); inside the search +4 / +12 orders are 0.8-1.0% dearer, fewer leaves (E-111) | (b) |
| Finisher switch | 20 s, 3 leaves, 0.5 | 5/10/40 s, 1/10 leaves, 0.25/0.75 share all level; none > 0.6% cheaper, the far ends 2-3% dearer on mixtures (E-116) | (a) |
| Chain depth | 3 | At equal time (900 s): 1 / 2 / 4 / 5 = 1.062 / 1.039 / 0.983 / 0.979; depth 5 has no plan by 900 s on 12/110 and overshoots to 25,727 s; depth 3 is the deepest that always plans in under 1 s (E-113, E-115) | (b), kept (user, 2026-11-21) |
| Simulations per move | 256 | Lowest mean on the 32-2048 curve; fewer is 1-2% dearer on s4-s5, more gains nothing (E-110) | (a) |
| Verdict threshold | 10% losses | At 5/10/20% no cell becomes a loss; 13 of 46 move between level and beats (E-112) | (c), shown robust |

## Not justified yet

| Number | Ours | Evidence | Open |
|---|---|---|---|
| (none left) | | | |

## Next, in order

| # | Number | Ours | Run | Cost |
|---|---|---|---|---|
| 3 | C_PUCT | 1.5 | 0.25 … 8 at 256 sims (J2's second curve) | ~10 h, one night |
| 5 | Blend partners; split-tree steps; MRCM budget | 4; 4; 2,000,000 | J6 curves not run yet (MRCM base); MRCM budget binds on 114/120 mixtures (E-113) | ~3-6 h |

## Argued or cited (no run planned)

| Number or choice | Ours | Argument |
|---|---|---|
| Time budget | 900 s CPU | Chosen budget, not converged: s3-s5 still 0.8-1.2% cheaper from 600 to 900 s, the gain halving per doubling; curve and the 60 s column reported (E-112, Q-139). 32/1,815 runs overshoot 990 s |
| Fluid weights | 64 / 8 / 1 | From e-MRCM (cite); J8 sensitivity optional (~16 h) |
| Mix and waste weights | 1, 1 | Our choice; J8 sensitivity optional, or state it |
| Step limit floor | 2x witness | Set for the RL agent (SPEC 8.1); J8 optional |
| Value scale, floor −1, dead end −1, mean backup, first-play value, most-visited move | standard PUCT | Standard choices (cite AlphaZero / MCTS surveys); J3 (~14 h) only if a reviewer asks |
| Two-input k:l mixes, storage, partial takes | task definition | SPEC 1.3, SPEC 15 (platform) |
