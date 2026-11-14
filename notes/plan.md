# Plan

Where the project stands and what comes next, step by step. Rebuilt from scratch on 2026-10-26
from everything learned up to E-23; the plan from §4 on was rewritten on 2026-10-27 around
stages 3–5 (D-35). `SPEC.md` is the design; `notes/paper_log.md` holds the evidence (IDs like
E-22 point there). This file says **what we are doing and why, in order**.

**How to maintain it**
- Read it at the start of every session. Update the status of a step the day it changes, with
  the paper log ID that backs it.
- When a result changes the plan, edit the plan here and record the decision in the paper log
  (D-…). Keep finished steps, marked done, so the path stays visible.
- `notes/questions.md` lists every question with its status; each open question names the
  step here that answers it. When a step finishes, close its questions there.
- Each step names its pass bar and what happens on pass and on fail. No step starts before the
  step it depends on has passed.
- **Defaults** (marked *default*) are Claude's choices so work can go on; the user can change
  any of them. When a step reaches a real decision, bring the user that one question with a
  recommendation, not the whole list.
- Compute: the Mac runs only light jobs at background priority (`OMP_NUM_THREADS=1 taskpolicy
  -b nice -n 19`). Anything longer than ~10 minutes goes to the workstation (`~/code/Btp`,
  tmux; never reboot it during runs).

*Last updated 2026-11-14 (E-90: M11 grid harness built, acceptance met; D-88: M11 grid harness designed, SPEC 0.47; E-89: sharing and run time, final: no time cost on many targets, 6-12% on s1-s2; E-88: sharing and run time, first look; E-87: large-plan scheduling gap; E-84: scheduling gap; E-83: M10 built, 8.1 done; D-84: execution harness designed, SPEC 0.45, Phase 8; D-67: paper benchmark plan, `notes/benchmark_plan.md`; E-62: published methods beat our method on 5 of 167, Q-138 closed; D-66: published wins, set W; E-61: greedy MRCM finisher beats lookahead inside the search, 10.0% below the new planner, Q-137 closed; D-65: greedy MRCM finisher; E-60: search over the stronger planner 8.4% below it, Q-136 closed; F-17 fixed; D-64: search over the stronger planner; E-59: search still below the stronger planner on 54 of 99, 3.7%, Q-135 closed; D-63: stronger planner joins, search re-check; E-58: recipe choice matters, lookahead with MRCM's tree loses nowhere, Q-134 closed; D-62: recipe choice by lookahead; E-57: MRCM's tree in the planner, a separate variant, Q-133 closed; E-56: the search improves DFM, only weakly FacDA, Generic, DFMM, Q-132 closed; E-55: the search improves MRCM at its leaves, Q-128 closed; E-54: at 40 targets the network does not speed the search, Q-131 closed; E-53: planner finish too slow at 40 targets; E-51: on set D the network does not help the search; D-57: goal check under steps-first and waste-first; D-58: scale check designed; E-52: no room under the other goals).*

---

## 1. Goal

One RL policy that plans MEDA sample preparation (k:l mixing, any M, F, targets, demand and
storage) and produces near-best-known plans for **MTMD mixture preparation with a storage
limit**, a class no published method handles, while matching specialised methods on the
sub-classes they handle (claim fixed 2026-10-21, paper log §1.2). Generalisation across sizes is
a requirement of the agent, not the headline.

**What the agent must show (D-35):** on stages 3–5 it makes cheaper plans than the **best
planner** we have (the cheapest of the sharing pass, its lookahead and its extra target
orders; the proved optimum where one exists), above all where that planner is weak: tight
storage and many targets (E-28). The agent may use planner-derived hints; the paper says so
plainly and reports how much the agent leans on them.

**How we work (D-39, supervisor, 2026-10-28):** stages 3–5 (many targets, demand, storage
limits) are the problems that matter. Stages 1–2 are a testbed: quick, well understood, good for
trying a method first, but not to be polished for their own sake. Every experiment reports
stages 3–5 next to stages 1–2, and every method is judged by what it does there.

## 2. What we have learned (the reasons behind the plan)

| # | Lesson | Evidence |
|---|---|---|
| 1 | A failure penalty that is too small teaches the agent to give up. | F-11, D-26 |
| 2 | Pure PPO cannot start: random play solves 2/1,200 dilutions and 0/1,200 mixtures (200–800 legal moves per step, 6–12 correct moves in a row). The regime does not matter while nothing is learned. | E-9–E-11 |
| 3 | The network could not find the right intermediate droplet (first mix right 24% on s1, 6% on s2, even on training problems). It saw ratios as decimals; the arithmetic that makes 1/8 the way to 33/40 was invisible. | E-14–E-16 |
| 4 | Exact hint columns computed by the env (mixes left to a target along known recipes) fix that. They, not the choice of teacher, lift copying from ~30% / 0% to ~75% / 65%. | E-16, E-20, E-21 |
| 5 | The sharing pass (reuse leftovers across targets) is a far better teacher and reference than the old construction: optimal on 51/65 small proved problems vs 34, never dearer. Its quality on many-target **mixtures** is uncertified because the bound is loose there. | E-17 |
| 6 | The old failure penalty ignored droplets on the chip, so building good intermediates earned nothing. The penalty now finishes from the current state. | E-18, D-32 |
| 7 | Teachers are not experts everywhere: cheap moves plus a planner finish beat the teacher on 7/40 s1 problems, and RL found plans 2–3× cheaper on some. Imitation must stay a warm start. | E-19, E-22, D-31 |
| 8 | Warm start + PPO works: s1 100% and cheaper than the teacher on average; s2 82.5% from the first 100k steps. | E-22 |
| 9 | Without the imitation anchor, s2 decays (97.5% → ~80%). The failures are exactly the 6 longest mixture plans; there the agent hoards droplets and dithers until the step limit. | E-22–E-24 |
| 10 | Bound ratios on val sets mostly measure the bound's looseness. Only ratios near 1 certify a plan. | E-17, E-23 |
| 11 | Hoarding is a symptom: a clean chip and twice the steps rescue 0 of 6 long s2 failures. Either the agent will not pay for progress or it is blind off the hint map; not yet separated. | E-25, E-27, D-34 |
| 12 | Headroom above the sharing pass is sparse but large: optimal on 141 of 220 proved small problems, 2–90% off on 12%, mostly tight storage and many targets. Cheap planner variants take much of it; exact search and the agent find what they miss (agent 13–64% cheaper on some single-target problems). | E-28 |
| 13 | Every benchmark problem is one our planner can solve (the generator keeps only those; at our ranges it almost never discards one). At inference the planner enters only through the recipe hint, a fixed per-target map that ignores what is on the chip and reuse across targets. | E-28, D-34, D-35 |

## 3. Where we stand

| Piece | State |
|---|---|
| Env, core, validator, generator, exact search, lower bound | Done (SPEC ≤ 0.22) |
| Sharing pass: teacher, "finish from here" penalty | Done (SPEC 0.24–0.25) |
| Hint columns (10 per candidate) | Single-target recipes and pure-fluid chains; no chip awareness, no reuse across targets |
| Imitation + warm start in PPO | Done (SPEC 0.26) |
| Stages 1–2 agent | s1 100%; s2 85% (peak 97.5%); one seed (`outputs/warm/mixed_s0`) |
| Stages 3–5 | Not trained. Zero-shot: s3 32/40, s4 11/40, s5 16/42, held-out 20/40; many-target mixtures 0/32 (E-26) |
| Proofs | E2 small set 220/399 proved (E-28); val sets 3 proved (s1); medium none |
| Regime choice | Expanding as a stand-in (*default*); pilot 3 after step 3.5 |
| Answer keys | Still the old construction as best known; v3 from E-28 is step 2.1 |
| Paper benchmarks (tiers 2–6) | Not built |

## 4. The plan

### Phase 1. Stages 1–2 (done as far as learning goes; rest parked, D-34)

- **1.1 Diagnose the s2 decay.** *Done (E-23, E-24):* the 6 failures are the 6 longest plans;
  the agent hoards and dithers.
- **1.2 Fix the cause of the hoarding** (D-33). *1.2a done (E-25, E-27): hoarding is a
  symptom; 1.2b (waste charged up front) not adopted.* The open cause is handled in Phase 3
  (3.2) and, if it returns on stages 3–5, Phase 4.
- **1.3 Second seed, 1.4 read cost properly.** Parked; seeds are repeated on the full system
  in 5.1.

### Phase 2. Grading we can trust

**2.0 Headroom above the sharing pass.** *Done 2026-10-27 (E-28):* 155 new proofs (E2 220/399);
the sharing pass is optimal on 141 of them, 2–90% off on 12%; 8 cells meet the rule fixed in
advance, mostly tight storage and many targets; cheap planner variants take much of it, exact
search and the agent find the rest. Large many-target mixtures stay uncertified. Records:
`notes/headroom/records.jsonl`; script `scripts/headroom_share.py`.

**2.1 The bar and the answer keys (now, small, Mac).** (a) Define the **best planner**: per
problem, the cheapest of `share`, `share_look3` and `share_orders` (D-35), as one producer.
(b) Answer keys `_v3`: best known = the cheapest of every producer, the agent's plans and the
proved optima of E-28. Every later grading uses `_v3`, so a run is never compared with a weak
reference.

**2.2 More proofs** (workstation, background, whenever it is free): exact search seeded with
the best planner on the val sets and longer budgets on the E2/E3 timeouts.

**2.3 Certify mixtures.** The bound is loose on many-target mixtures; tighten it or prove more
small ones. Until then those results are intervals.

**2.4 Mini literature comparison on stages 1–2** (D-36; *started 2026-10-27 in its own
session*). Five or six published heuristics that claim single-target dilution or mixture
preparation (candidates on disk: FacDA, Dong 2024, dilution from arbitrary stocks 2023, MRCM,
Kundu 2022 mixture, the generic approach of Roy 2021), implemented from their PDFs and checked
against a worked example or table in each paper. They run on `val_s1_v2`, `val_s2_v2` and the
proved E2 s1/s2 problems, next to the sharing pass, the best planner and the agent (E-22). Each
plan is replayed through our engine. Report: cost under 64/8/1 with sample units, each paper's
own metric, and how many problems each method accepts. It also covers the papers' own example
problems and ~100 fresh s1 and s2 draws (Q-103). Run on the workstation with the E-22 agent,
and run again on the fully trained agent after 5.4 (user, 2026-10-27). It answers early where
we stand against the literature. It isn't the paper's benchmark (that's 6.3), but its implementations
carry over.

### Phase 3. Stages 3–5 (now)

Per stage: (a) is the teacher good there? (b) can the network copy it, with hints? (c) warm
start + PPO with the stages already learned. Never train a stage whose copy test fails.

**3.0 Fix F-16.** *Done 2026-10-27 (SPEC 0.27): `imitation.UidMap`; tests include
`val_s4-033`, which crashed before.* The env offers one move per set of identical droplets; the teacher's
plan names one particular droplet, so when it names the twin the env did not list, nothing
matches and the copy code crashes. Stages 3–5 make identical droplets often (demand). Fix:
match moves by what they do (canonical key), keep a map between the teacher's droplets and
the env's. Tests on identical-droplet cases.

**3.1 Zero-shot check.** *Done (E-26):* s3 32/40, s4 11/40, s5 16/42, held-out 20/40;
single-target dilutions transfer at the teacher's cost; many-target mixtures 0/32.

**3.2 Hint coverage** (*done 2026-10-27, E-30: not blind; the hint map covers the teacher's
route (87–100%) and the planner's next move in failures, but in failed episodes the agent
picks unhinted mixes (17–50% hinted vs 70–98% when solved). Held-out failures pick hinted
mixes and still fail: cross-target information is missing.*) Reading fixed in advance, on the planner's next move from the agent's state: blind
off the map if it is hinted on < 50% of the last-quarter steps of failed episodes (and less
than half as often as in solved ones); will not commit if it is hinted on ≥ 70% of failed
steps but gets median probability < 0.3. Along the s2 failures and along the
teacher's plans on s3–s5: how often the candidates carry any hint, whether the agent's move was
hinted, and the probability it gave the planner's next move. Answers two things at once:
whether the agent gets lost off the hint map (lesson 11), and where 3.3 must add hints.

**3.3 Better hints** (SPEC change; *design after 3.2*). Today's map is fixed per target, from
single-target recipes. Candidates, chosen by what 3.2 shows: recipes from the **best planner**
for many targets (reuse across targets), extending the map on the fly from compositions the
agent actually makes (cached), and the planner's next move from the current state (one
planner call per move: fine on s1–s3, too slow on s4 as it stands).

**3.4 Copy tests, stage by stage** (workstation, ~1 h each): s3, then s4, then s5.
- Teacher = the **best planner** (D-35); exact optima where proved.
- Pass bar (*default*, like stages 1–2 reached): move agreement ≥ 70% and greedy success
  ≥ 60% on the unseen val set. Fail: look at what is missed (hints, teacher, network) before
  any RL.
- Demonstrations are made once, before training, in one process (`train.imitation_demos`,
  1,000 problems). The best planner is up to ~15× slower than the plain pass, so for stages 4–5
  generate them in parallel or cache them before the run.

**3.5 Warm start + PPO** (workstation, ~75 min a run). Expanding regime (*default*, a stand-in
for the user's curriculum): s1–s3, then s1–s4, then s1–s5. One seed.
- Pass bar (*default*): success ≥ 90% on s3 and ≥ 80% on s4–s5; median cost within 5% of the
  best planner; cheaper than the best planner on a real share of the headroom cells (E-28).
- If long plans fail again as on s2 (lesson 11): the options, in order, are search at decision
  time (4.2), progress shaping, a small constant imitation weight; 3.2 says which fits.

**3.6 Regime pilot 3**, once 3.5 passes: sequential, expanding and mixed, all with the warm
start, stages 1–5, 2 seeds each, graded by `regime_report.py`. *The user brings the curriculum
design before it starts.*

### Phase 4. Beat the planner where it is weak

**4.1 Per-cell comparison** against the best planner and, where proved, the optimum; sample
units saved next to cost (D-25).
**4.2 Small MCTS test at decision time** (M9 early): the network's policy as prior, "finish from
here" as leaf value, on the headroom cells against the best planner. Lookahead targets the
planner's known miss (building two intermediates ahead and blending them, E-17). *Brought forward
2026-10-27 (D-37), before 3.3–3.5:* `scripts/search_check.py`, three searches (network +
planner finish, flat prior + planner finish, network + its own value with no planner) on 23
proved planner misses and 13 MRCM wins, reading fixed in advance. *Done 2026-10-28 (E-33):*
lookahead with planner finishes finds most misses even with a flat prior (20/23, 13 at the
optimum); the network's prior hurts it (16/23); without the planner the network finds 3/23.
*Expert iteration round 1 (D-38), done 2026-10-29 (E-34): bar not met.* The search-taught
network guides search as well as a flat prior, ~30× faster (best mixture result so far); alone,
or in search without the planner, it is still not below the planner (1 and 5 of 23); its value
head was never trained; s2 fell to 47.5%.
*Now (D-40, chosen 2026-10-29): expert iteration round 2*, the first real turn of the loop:
the round 1 network guides the search (planner finishes), the search records what it observed
(visit counts, cheapest remaining cost), two copy arms (policy + value vs policy only), re-test
as E-34. Pass bar in D-40: (a) the value head judges (no-planner search ≥ 8 of 23 and above the
control), (b) the loop climbs (greedy cheaper than round 1 on set A, s3-s5 success at least
round 1's). *Done 2026-10-30 (E-35): (a) and (b) met.* Trained on what search observed, the
value head judges: the no-planner search is below the planner on 17 of 23 (control 7, round 1
5); the network alone is cheaper than round 1 (1.390 vs 1.447 × optimum, 22/23 solved). Not on
mixtures: the no-planner search fails to finish 9 of 13 MRCM problems; s2 stays at 45%.
*Next (D-41, 2026-10-30): before round 3, the same search test on full-size val s3-s5* (set
C, 99 problems where the planner is not known optimal), round 2's network, no training; bar in
D-41. Then round 3 (value head as judge, the planner finishes only problems the search cannot
finish, gating on round 2), if D-41 says so. *Done 2026-10-30 (E-36): (1) yes, (3) no.* Search
with planner finishes beats the best planner on 85 of 99 full-size s3-s5 problems (16% cheaper
on average), with or without the network's prior; without the planner it fails to finish 47.
*Fair-time test (D-42) done 2026-10-30 (E-37): lookahead is real.* The planner given 900 s of
random restarts gains 1-2%; the search stays ~14% cheaper (79 of 99). *Race (D-43) done 2026-10-30 (E-39):* no speed gain from the network on set C (10 s and 60 s
tied); on the mixtures (B) it is faster and better (0.591 vs 0.675 of the planner at 60 s). *Now (D-45, 2026-10-31): round 3*, practice
mostly on full-size s3-s5, more value data, one copy arm on rounds 2 and 3; bar in D-45 (the
no-planner search on set C: ≥ 33 of 99 below the planner, ≤ 24 unfinished). Planner speed (E-38): ~2× on s1-s3 with identical plans, nothing on s4-s5;
depth 2 is 10-18× faster but 7-10% dearer, parked (D-44). *Round 3 done 2026-10-31 (E-40): bar not met, gate keeps round 2.* The no-planner search on C is below on 24 and fails to finish 53 (round 2: 29, 47); by D-45's rule the loop stops on s3-s5, and the method there is search with planner finishes (84 of 99, unchanged). *Dead-end check (D-46) done 2026-11-01 (E-42):* the value head is blind to dead ends (Q-115 confirmed); the practice search cannot supply them (0.5% dropped). E-41: on s3-s5 the network does not help as a guide on mixtures. *Round 4 (D-47) done 2026-11-01 (E-43): bar not met* ((a) 23 below / 50 unfinished, (b) 17 of 23, (c) cliff 1.03); the head learned that hopeless positions are dear on average but not the step into one. By D-47's rule the network stops on s3-s5 and the question goes to the supervisor; the method there stays search with planner finishes. *Fit check (D-48) done 2026-11-01 (E-44):* the network fits s4-s5 and carries it to new problems (guide 0.84 → 0.78, judge pair order 0.73 → 0.67); not the architecture. The guide is a good copy of a search that adds nothing over a flat prior; the judge ranks siblings poorly even on its training data, likely because its targets (cheapest plan seen, planner finishes) are noisy. *Now (D-49, user 2026-11-01): AlphaZero loop pilot* (`scripts/zero_loop.py`): the network trains on the outcomes of its own no-planner play (visit counts and own-outcome values on the positions played), small problems and s1-s3, 8 turns; bar in D-49 (no-planner search on A+B+C3 vs turn 0). *Pilot done 2026-11-01 (E-45): does not climb* (24/23 → 24/24 below/unfinished over 8 turns, noise only; small problems slightly worse). Own-outcome targets alone do not lift it at this scale. *Now (D-50, Q-121): resolution check* (`scripts/resolution_check.py`): does the search try every move anyway, and are sibling gaps smaller than the head's error? Measures only, no training; reading in D-50. *Done 2026-11-01 (E-46):* the prior has room (median 85 candidates on s4-s5, 10% tried); the value head cannot resolve sibling moves (gap 16 vs error 142), because it is trained to recompute the planner's finish cost (target = planner finish at the median position). *Now (D-51, user 2026-11-02):* check whether the exact lower bound carries most of the cost, so the head can predict only the excess above it (bound columns in the resolution check; reading in D-51). *Done 2026-11-02 (E-47):* the bound carries about half the cost on s4-s5, but the projected head error on the excess (62) is still above the sibling gap (16); the trained head ranks siblings no better than the bound alone; the search beats the planner's finish at 24% of positions, by a median 65. *Now (D-52, user 2026-11-02): gain head* — the value head learns how much the planner's finish can be beaten (planner stays at the leaves); steps: labels, copy, offline sibling check (bar 0.77 → 0.82, AUC 0.7), then the search on C. Steps 1-3 built (`scripts/gain_check.py`). *Done 2026-11-02 (E-48): fails both bars* (held-out s4-s5: gain picks the best sibling 0.74 vs planner 0.75, AUC 0.66); weak signal, shrunk predictions; step 4 not run. *Now (D-53, user 2026-11-02): data scaling check* (`scripts/scaling_check.py`): clean planner-finish labels on fresh s4-s5 positions, value head trained on 3k/10k/30k/100k (300k too slow, D-53); does its agreement with the planner's sibling choice rise? Reading in D-53. *Done 2026-11-03 (E-49): data helps, slowly* (head picks the planner's sibling 0.34 → 0.39 from 3k to 100k; error 194 → 139, about N^-0.10, against a gap of 7; bound alone 0.36). No feasible amount of labels closes the gap. *Next: the user's decision (Q-125)* on whether to change the claim to search over a planner, with learning where the planner is weak. *Now (D-54, user 2026-11-03): two new value heads, offline* (`scripts/head_check.py`, Q-126): a head that reads the candidate moves (and hints), and a head trained to rank siblings, alone and together, on E-49's data; bar: picks the cheapest sibling >= 0.55. *Done 2026-11-03 (E-50): below the bar* (picks the cheapest sibling: control 0.38, moves 0.38, rank 0.49, moves + rank 0.52 < 0.55; top2 0.68); the ranking loss is the change that matters, but by D-54's rule Q-125 (the claim) is next. Queued after it: save the search's plans on set A and part of C, and list intermediate droplets the planner never makes that a later step reuses (user, 2026-11-03). *Also now (D-55, Q-128, user 2026-11-03; an experiment, the claim is not changed): finisher test* — MRCM, not our planner, at the search's leaves (`MrcmFinish`), on set B (13 mixtures where MRCM beats our planner); bar: below MRCM on >= 5 of 13. Built, run on coe9 pending. *Now first (D-56, Q-129, user 2026-11-03): set D race* — full-size problems where e-MRCM or NFSP beats our planner (`lit_bench_full.py` run first), then `search_check.py sets=D`, round 2's V network vs flat prior; reading D-43's race rule. *Done 2026-11-03 (E-51): no gain* (13 problems; net 0.754 vs flat 0.755 of the planner at 60 s); the search is at or below e-MRCM / NFSP on 11 of 13. *Now (D-57, Q-130, user 2026-11-03): goal check* (`scripts/goal_check.py`): under the literature's steps-first and waste-first goals, on E2's small problems, how far are the planner and the flat-prior search over it from the optimum? Room left → step 2 trains a network on that goal. *Done 2026-11-04 (E-52): no room* (after the search at most 2.2% of problems > 2% above the optimum under either new goal; the planner alone misses 9.2% / 18.9%); step 2 not run. *Next (D-58, Q-131, user 2026-11-04): scale check* — time one planner finish on s5 and on bigger probes (20, 40 targets); median >= 10 s at some size → race the network there (user's go-ahead); otherwise the network has no role at these sizes. *Step 1 done 2026-11-04 (E-53): unaffordable at 40 targets* (median finish 25.6 s; s5 2.1 s, 20 targets 5.0 s); *Now (D-59, user 2026-11-04): step 2, the race at 40 targets* (`search_check.py sets=T`): round 2's network (`net`, `zero` at 1/16/256 sims) vs the flat prior; D-43's rule per arm. *Done 2026-11-04 (E-54): no speed gain* (at 60 s net 0.974 vs flat prior 0.971 of the planner; the network alone finishes 1 of 10; the search is 6.5% below the planner by 900 s with or without it). *Finisher test done 2026-11-06 (E-55): the search improves MRCM* (MRCM at the leaves: below MRCM on 7 of 13 of set B, 5.1% below on average). *Now (D-60, Q-132, user 2026-11-06): finisher test for the other published methods* (`search_check.py sets=F`): DFM, FacDA, Generic and DFMM at the search's leaves, 13 problems each; reading in D-60. *Done 2026-11-06 (E-56): 1 of 4 improved* (DFM 5/13; FacDA, Generic, DFMM weak); the search over our planner beats each method on 6-11 of its 13 best problems. *Now (D-61, Q-133, user 2026-11-06): MRCM's tree as a fifth recipe in the planner* (`scripts/planner_check.py`): gain on s3-s5, no loss elsewhere; reading in D-61. *Done 2026-11-06 (E-57): a separate variant* (11% cheaper on average, MRCM's and e-MRCM's wins gone; dearer than the plain pass on 17 of 242, all many-target). Left beating us: FacDA 7, DFMM 9, NFSP 4. *Now (D-62, Q-134, user 2026-11-06): recipe choice* — lookahead over each group's 3 cheapest recipes with MRCM's tree (`planner_check.py sets=val planners=...`); reading in D-62. *Done 2026-11-06 (E-58): the greedy choice was the problem* (dearer than the plain pass on 0 of 242, 12.1% cheaper on average, all 17 losses fixed; median 1.5 s vs 0.1 s). *Now (D-63, Q-135, user 2026-11-06):* it joins the best planner as a fourth variant (the search's leaves keep the plain pass), and the search on set C is re-checked against it from E-39's and E-58's records (`planner_check.py search=... look=...`); reading in D-63. *Done 2026-11-06 (E-59): the search still earns its place, by far less* (below the new planner on 54 of 99, 3.7% cheaper on average, was 17.2%; the new planner alone is 13% cheaper than the old). *Now (D-64, Q-136, user 2026-11-06): search over the stronger planner* (`search_check.py sets=C finishers=strong`, flat prior, 900 s cap; read with `planner_check.py search=...`); reading in D-64. *Done 2026-11-07 (E-60, after fixing F-17): a stronger finisher makes the search add more* (8.4% below the new planner, below on 72 of 99; 20.6% below the old planner). *Now (D-65, Q-137, user 2026-11-07): does the search need the planner's lookahead?* (`search_check.py sets=C finishers=share_mrcm`, MRCM's tree with greedy recipes); reading in D-65. *Done 2026-11-07 (E-61): the search makes the recipe choice itself* (greedy MRCM finishes: 10.0% below the new planner, below on 79 of 99, better than lookahead's 8.4% / 72; no faster chooser needed). The method on set C is search over the greedy MRCM pass. *Now (D-66, Q-138, user 2026-11-07): published wins* (`search_check.py sets=W`): every problem where a published method beats the old planner, first the new planner, then the search where a method still wins; reading in D-66. *Done 2026-11-07 (E-62): most closed* (published methods beat us on 5 of 167, was 167 against the old planner; two by ~1.8x: litbench_s1-052 FacDA, val_s3-015 NFSP). Next: the user's choice. Still queued: the plan check (Q-127), the claim (Q-125). Everything below 4.2 waits.
**4.3 Self-imitation** if cost stalls near the teacher.

### Phase 5. Make it solid

- **5.1** Second and third seeds of the full system.
- **5.2** Generalisation probes (unseen sizes and constraints; benchmark tier 4).
- **5.3** Recipe-hint ablation: copy test and agent without the planner-derived hint (chain
  hint only), to report how much the agent leans on the planner at inference. *Brought
  forward 2026-10-27 (Q-105, E-31):* `hint_source: problem` (SPEC 0.28) is built. A
  planner-free hint covers dilutions but not mixtures. The stage 1–2 copy test, planner hints
  vs problem hints, *done (E-32)*: problem hints hold s1 (100% greedy success) but s2 falls
  to 32.5% (planner hints 45–67.5%).
- **5.4** Full run, 5 stages × 3 seeds, with the regime chosen in 3.6.

### Phase 6. Paper benchmarks and baselines

*2026-11-07 (D-67): the full plan, rules and build order (B1–B5) are in `notes/benchmark_plan.md`. Now: B1, stage test sets. Sets built 2026-11-07 (D-68, 605 problems); runner `scripts/bench.py` and reading D-69; done 2026-11-08 (E-64): beats six published methods, level with NFSP. B2 built the same day (D-70): `test_emrcm_v1` (e-MRCM's protocol, 2,847 problems) and `litbench_paper_v1`; part (a) done 2026-11-08 (E-65), part (b) done 2026-11-09 (E-66: beats FacDA, MRCM, e-MRCM; weak at N = 4-5, Q-142). B3 done 2026-11-09 (E-67: checks pass). E-67 also finds why N = 4-5 is weak: the search's finisher is too slow there. *Now (D-72, user 2026-11-09): pilot A*, finisher picked by speed, v1 vs A on `val_emrcm_v1` and val s3-s5 on coe9; pass → A is method-v2 for the B1/B2 rerun with all 16 published methods; fail → N = 4-5 is a reported limit, then the architecture question. *Also (E-68, D-73, user 2026-11-09): trial v3*, planner jumps in the search so the 4 planner runs can go (Q-143); run on the same sets after A.* *Done 2026-11-10: A (E-69) passes at N = 4-5 (11% cheaper, DFMM losses 41 → 23) but fails on s5 and N3 (the switch fires too often); v3 (E-70) fails both bars, jumps dropped, Q-143 closed.* *Now (D-74, user 2026-11-10): pilot A2, both finishers at once (plain pass at every leaf, greedy MRCM too while it has used at most half the time), on the same sets; bar in D-74.* *Done 2026-11-10 (E-72): fails like A (N 4-5 0.884, but N3 1.244 and s5 1.034).* *Now (D-75, user 2026-11-10): pilot A3, v1 per problem until MRCM averages over 20 s a leaf, then A2 (E-73: offline it keeps the gain and is level elsewhere); bar in D-75.* *Done 2026-11-11 (E-74): passes* (N 4-5 0.927 of v1, DFMM losses 41 → 30; level or better elsewhere); A3 is method-v2. Next: the planner-time count (Q-144), then the B1 / B2 rerun with method-v2 and all 16 published methods. *Now first (D-76, user 2026-11-11): the "unexpected moves" check (Q-127)*, `novel_check.py` on val s1-s5 and `val_emrcm_v1`: does the search make intermediates no planner run makes and reuse them? It decides which story the paper tells before the rerun. *Done 2026-11-11 (E-78): the story holds* (91% of search wins use a novel reused intermediate). E-75 to E-77: the search alone carries every published verdict; it loses to the 4 runs only on 10 of 82 val s4-s5, starved and out-timed. *Now (D-78): CPU-time budgets* (coe9 shared). *D-79: method-v3 built* (search alone, first plan, novelty in records, answer-key reuse); freeze after D-77. *D-80: causal check of E-78 built* (single-target recipes, ablation); coe9 run after the fair test. *Done 2026-11-12 (E-79): the fair test fails* (5 of 10 still dearer at 1 h, by 0.1-4.1%; 3 of them stop improving after the switch to `tiered`); next lever: the finisher after the switch. *Now (D-81): the finisher check* (`finisher_check.py`, coe9): does a lookahead-3 finish of the search's own moves reach the best run? *Done 2026-11-12 (E-80): the causal check holds* (247 of 253 novel intermediates outside every single-target recipe; one move explains >= half the gain on 46 of 59). *Done 2026-11-12 (E-81): the finisher check fails its bar* (4 of 9); root cause is depth: on s4-s5 the tree picks only 0-2 moves of 22-174, the rest is one planner variant. *Now (D-82, user 2026-11-12): the prior check* (`prior_check.py`, coe9): can the network pick the search's moves on s4-s5, to let it go deeper? *Done 2026-11-12 (E-82): no* (search's moves in its top 10% on 48 of 147; the network stays out). *Decided 2026-11-12 (D-83): method-v3 frozen* (SPEC 0.44; search alone, 900 s CPU, no final lookahead; s4-s5 depth limit reported). Next: the final B1-B3 runs with all 16 published methods. *Started 2026-11-13 on coe9 at `6161ea9`. B2(a) done (E-85): beats 11 of 16 on their own goal, level with NFSP and Dong, 3 too few; B3 done (E-86): checks pass, same as E-67. B1, B2(b) running.* *Now (D-87, user 2026-11-13): leftover-time pilot* on val s1-s3 (`bench.py grow=4 max_sims=4096`, read by `grow_check.py`); joins → rerun the small sets; else method-v3 stays.

- **6.1 Paper checks:** robust-aliquot (Poddar 2020), Dong 2024, e-MRCM → claim wording
  (session `paper-checks`).
- **6.2 Tier 2:** DMFB sanity sweep (`dmfb_sweep_v1`). *Built 2026-11-08 (B3, D-71); done 2026-11-09 (E-67): checks pass.*
- **6.3 Tier 3:** published baselines, one set per paper (WSPM, FacDA, MRCM, e-MRCM, MTWM, and
  single-target MEDA dilution methods such as Dong 2024 for stage 1), plus the plan
  translator. Needs the PDFs; SPEC 10.4 forbids rebuilding methods from memory. *Started
  2026-10-31 (L-6, L-7):* all PDFs now on disk; e-MRCM built and matches its worked example;
  its Table III does not reproduce under our reading (Q-117); reported as "our reading" (user). NFSP built 2026-10-31 (L-8):
  matches its Fig. 2; 600 s per solve on the workstation (user, Q-118 closed, L-8); runner
  `scripts/lit_bench_full.py` run 2026-11-03 (E-51): NFSP applies to 17 of 122, e-MRCM to 13; together they beat our planner on 13. *Audit 2026-11-08 (L-9, `notes/lit_bench/coverage_audit.md`): 7 of about 25 methods in the folder built; 13 to build (Q-141).* *Done 2026-11-09 (L-10 to L-18): all built (STWM/MTWM, robust, MTCS/MTR/MTMD, MTC/DDMT, TVODA, WSPM, Dong, Fukui, SIMOP); next: one B1/B2 rerun on coe9 with them.*
- **6.4 Tier 4:** generalisation probes (with 5.2).
- **6.5 Tier 5:** frontier set `frontier_v1`.
- **6.6 Tier 6:** assay ratios from real protocols.

### Phase 7. Paper

Written from `notes/paper_log.md`.

### Phase 8. Execution: scheduling, routing, washing (D-84, 2026-11-12)

Design: SPEC 14 (harness) from the user's execution document. The question every step serves
(Q-145): **do the savings from shared intermediates survive execution, and if not, can planning
that sees execution cost keep them?** Rule for every stage (D-84): classical first, measure the
gap to an exact answer and where the time goes, then search over the classical stage; learning
only where search is too slow (the router is the likely place). No RL scheduler by default.

- **8.1 Build M10** (SPEC 14.2-14.4, 16): platform config, operation graph, list scheduler with
  storage and detector changeover, schedule validator, metrics, exact ILP check. *Done
  2026-11-12 (E-83, SPEC 0.46): all four acceptance criteria met; a plan-order safety check
  added after a storage deadlock on val_s5-006.* Next: 8.2.
- **8.2 Scheduling gap:** list schedule vs ILP on small plans (`tiny_exact_v1`, E2's proved
  problems). Small → scheduling is done. Large → search over priorities, still no RL. *Run
  2026-11-13 (E-84): optimal on all s1-s3 plans; 1.011 x the optimum overall on the 184 graded
  plans (bar met); the 28 largest s4-s5 plans ungraded (ILP no answer in 60 s; a simple bound
  allows up to ~3x). Open: grade them (longer ILP on coe9) before calling scheduling done. Graded on coe9
  2026-11-13 (E-87): 17 of 28 answered; list scheduling 1.077 x the ILP's best there (max 1.24),
  plan order better than critical path on 16 of 17. Overall bar met (1.017, 94.5% within 10%),
  not on the largest plans: a search over priorities is the fallback for them (user's call).*
- **8.3 First number (Q-145):** schedule the B1 plans of our method, the best planner and each
  published method; compare makespan, `storage_ticks`, `storage_peak` and `detector_wait`,
  across a small grid of the placeholder durations (Q-146). Reading fixed in a D-entry before
  it runs. *First look 2026-11-13 (E-88, planner as
  stand-in, 120 problems): no time cost (4.5% faster than no sharing on many targets), but 2.4x
  more storage x time. Full run with the search's plans: coe9 (D-86).* *Done 2026-11-14
  (E-89): many targets: 6% faster than no sharing, level with the plain pass and the published
  methods; s1-s2: 6-12% slower than the published single-target methods for 11-40% less fluid.
  Storage x time 1.3-3.9x everywhere: the open cost is routing (8.4). 8.5 not triggered on
  many targets; for s1-s2 the user's call.*
- **8.4 M11: the grid harness** (SPEC 14.6-14.10, D-88). *Designed 2026-11-14 (SPEC 0.47).*
  Build in order, each with tests: (1) layout and footprints, (2) space-time A* router,
  (3) placer, (4) executor (list scheduling on the grid), (5) run validator and metrics,
  (6) `scripts/grid_time.py`: E-89's producers on the grid (30 x 30, 60 x 60, 100 x 100; a 3 x 3
  mix = 3/15/60 cells; L-19), Mac pilot, then coe9; read with D-88. Answers Q-145's routing half.
  *Steps 1-5 built 2026-11-14 (E-90): every M11 acceptance criterion met on val; real run time
  2-3.5x the no-travel schedule.* *Step 6 built 2026-11-14: `scripts/grid_time.py` (9 variants,
  D-88's reading); Mac pilot on 6 test problems: every plan ran on every variant, ~46 CPU s per
  problem without the search. Full run with `method=1` on coe9 next.* Then M12 (washing, cost record, Protein/IVD
  benchmarks).
- **8.5 Execution-aware sample prep**, if 8.3 shows sharing costs time: the search scores plans
  by fluid cost plus a fast execution estimate (the M10 schedule). Same method, new objective.

## 5. Waiting on the user (only when the step is reached)

- **3.4 / 3.5 pass bars:** the defaults above stand until changed.
- **3.6:** the curriculum design, before regime pilot 3.
- **Pure RL vs RL + teacher on stages 3–5** (was 3.5, D-31 (6)): still wanted?
- **PDFs** (6.1, 6.3): e-MRCM (Liang et al., TCAD 39(10) 2020; free author copy at
  par.nsf.gov/servlets/purl/10155987) and NFSP (Dinh et al., ASP-DAC 2014) still missing.

## 6. Parked (small, not blocking)

- Stage 2 polish (85% → 95%) and the up-front waste charge (D-33, E-27).
- `regime_report --at-step common` breaks when a run is mid-round (workaround: wait or pass a
  step).
- Tests should cap PyTorch threads themselves; the `fork()` warning in async vector-env tests.
- Attention interpretability, for the paper's analysis section only.
- PopArt or PPG if the shared actor-critic loss ever shows value interference.
