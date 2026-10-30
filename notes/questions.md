# Questions

Every question the project has asked, big or small, and where it stands. An **open** question
points to the plan step (`notes/plan.md`) meant to answer it. A **closed** question gives the
answer, how it was settled, and the evidence (IDs like E-22 point to `notes/paper_log.md`).

**How to maintain it**
- Add a question the day it comes up, even a small one. Give it the next free ID; IDs never
  change or get reused.
- Status is one of:
  - **Open → plan x.y**: a plan step will answer it. If no step does yet, write **Open, no step**.
  - **Closed (experiment)**: a measurement answered it. Give the result and the E-/F- ID.
  - **Closed (decision)**: we chose and didn't measure. Give the D- ID. These can be reopened
    if new evidence comes in.
  - **Closed (by construction)**: the code or the definitions answer it. Say where.
  - **Parked**: real, but not worth answering now. Say why.
- When a question closes, keep it and change its status. If a later result overturns the
  answer, reopen it and say which entry did it.
- The first table lists only the open questions. Keep it in sync with the sections below.

*Started 2026-10-27 from the paper log (D-1–D-35, E-1–E-28, F-1–F-16), the plan and the
chats. Last updated 2026-10-30.*

---

## Open questions at a glance

| ID | Question | Where it gets answered |
|---|---|---|
| Q-8 | Is RL worth it here: does the agent beat the best planner on stages 3–5? | 3.5, 4.1 |
| Q-11 | How much does the agent depend on the planner-derived recipe hint? | 5.3 |
| Q-12 | Can search at decision time (AlphaGo Zero style) do what the hints do now? | 4.2 |
| Q-18 | Would progress shaping help long plans? | 3.5 (fallback) |
| Q-31 | Does a storage limit actually change what good plans look like at our sizes? | 4.1 |
| Q-34 | Paper benchmarks, tiers 2–6 | Phase 6 |
| Q-40 | Is the sharing pass near-optimal on many-target mixtures? | 2.3 |
| Q-41 | What's the reference ("best known") every grading uses? | 2.1 |
| Q-42 | Can we prove more optima on the val sets? | 2.2 |
| Q-49 | How do the agent and planner compare with published heuristics? Could one be a better teacher? | 2.4 (stages 1–2), 6.3 (all) |
| Q-101 | Has the agent been fully trained (all stages, regime chosen, 3 seeds)? | 3.5, 3.6, 5.4 |
| Q-103 | Is the mini benchmark's problem set big enough and unseen? | 2.4 |
| Q-104 | Should MRCM become part of our planner and teacher for mixtures? | Open, no step yet (E-29) |
| Q-105 | Too much dependence on the greedy planner's recipe hint? Can hints come from the problem alone? | Dilutions yes, mixtures no (E-32); open for mixtures |
| Q-50 | Does the best planner as teacher beat the plain sharing pass on stages 3–5? | 3.4 |
| Q-60 | What hints do stages 3–5 need? | 3.3 |
| Q-71 | Pure RL vs RL with a teacher on stages 3–5 | Waiting on the user (plan §5) |
| Q-107 | Can the network learn the plans search finds (search as teacher), and beat the planner without it? | Round 1 partly (E-34); 4.2, round 2 (D-40) |
| Q-108 | Can the value head learn to judge positions, so the planner leaves the search? | Small problems yes (E-35: 17/23); full size no (E-36: 48/99 unfinished); round 3 (D-45) |
| Q-109 | Does the AlphaZero loop climb (each round's network better than the last)? | One turn, slightly (E-35); 4.2, round 3 |
| Q-110 | Why did s2 drop to 47.5% when copying the search's plans? | Persists in round 2 (E-35: 45%); open, no step |
| Q-72 | Self-imitation if cost stalls near the teacher? | 4.3 |
| Q-73 | Are the results stable across seeds? | 5.1 |
| Q-78 | Root cause of the long-plan failures (s2, and s3–5 zero-shot) | 3.2, then 3.5 |
| Q-82 | Can the network copy the best planner on s3, s4, s5? | 3.4 |
| Q-83 | Does warm start + PPO pass on s1–s3, s1–s4, s1–s5? | 3.5 |
| Q-84 | Can stage 4–5 demonstrations from the best planner be made fast enough? | 3.4 |
| Q-87 | Which training regime? | 3.6 |
| Q-88 | What's the curriculum? | 3.6 (the user designs it) |
| Q-91 | Do robust-aliquot, Dong 2024 or e-MRCM narrow the claim? | 6.1 |
| Q-95 | The real platform: M, mixer footprint, reservoirs, cost weights, router as cost oracle | Supervisor |
| Q-96 | Do the generalisation claims hold (unseen sizes and constraints)? | 5.2, 6.4 |

---

## A. The problem and the claim

**Q-1 What exactly does the paper claim?** · Closed (decision, 2026-10-21, paper log §1.2)
One policy produces near-best-known plans for MTMD mixture preparation on MEDA with a storage
limit, a class no published method handles, and matches specialised methods on the
sub-classes they do handle. Fixed by the user.

**Q-2 Is "it generalises across sizes" the headline?** · Closed (decision, 2026-10-21, §1.2)
No. It's a requirement of the agent. Classical heuristics already work for any M and any
ratio without training, so "it generalises" alone is weak against them.

**Q-3 Which constraint cells does the literature already cover, and which are frontier?** ·
Closed (experiment, E-4)
Covered: no demand and no storage limit (WSPM, FacDA, MRCM, e-MRCM, MTWM). Frontier: demand > 1
together with many targets or a mixture, and every cell with a storage limit.

**Q-4 MEDA k:l mixing or DMFB 1:1?** · Closed (decision, D-10)
MEDA first. 1:1 (M = 2) is only a sanity check.

**Q-5 Must every frontier result be graded against an optimum before we train?** · Closed
(decision, D-18)
No. A cell is frontier because nobody has tried it. Where we can grade, we grade (optimum,
lower bound, constructions). Elsewhere the agent is first and the result is reported as an
interval.

## B. Why RL, and the planner's role

**Q-6 Why RL, when the transition function is known exactly?** · Closed (argument, §1.4;
user commit `d0da1f9`)
"The model is unknown" isn't the reason. The reasons: (1) amortisation, since exact search
is intractable past tiny problems while a policy answers in milliseconds; (2) a learned value
is the missing heuristic for k:l mixing with many targets, demand and storage; (3) later, the
cost becomes genuinely unknown (completion time from the router). This is an argument, not a
measurement. Q-8 is the measurement.

**Q-7 Where can the sharing pass be beaten, and by how much?** · Closed (experiment, E-28)
Rarely, but by a lot when it can. It's optimal on 141 of 220 proved small problems and more
than 2% off on 27 (12%), up to 90% (one outlier at 301%, checked as real). The 8 cells that
meet the rule are mostly tight storage (5) and many targets (6).

**Q-8 Is RL worth it: can the agent beat the best planner where it matters (stages 3–5)?** ·
**Open → plan 3.5, 4.1**
Partial evidence so far: on stages 1–3 the agent found plans 13–64% cheaper than every planner
variant on a few problems (E-28: s1 7/40, s2 8/40, s3 mixtures 3/20). But it's also dearer on
others (s1: 14/40, E-22). On stages 3–5 nothing has been trained yet.

**Q-9 Doesn't a planner-derived hint at inference undercut the RL claim?** · Closed (decision,
D-35)
The user's argument: if the agent beats the planner while using its hints, the hybrid is a
valid result. The paper says so plainly. Q-11 measures how much the agent depends on the hint.

**Q-10 Does the full planner run at inference?** · Closed (by construction; D-34, D-35)
No. At inference the planner enters only through the recipe hint. That's a fixed map per
target, built once from the problem statement when the episode starts, from the
single-target tree's and the sharing pass's recipes (`solvers/hints.py`). The teacher and the
failure penalty are used only in training. The chain hint is pure arithmetic on the problem,
with no planner involved.

**Q-11 How much does the agent depend on the recipe hint?** · **Open → plan 5.3**
(ablation: chain hint only). Partial evidence: on s1 the planner-free chain feature alone takes
copy success from 25% to 67.5% (E-16). But on s2 the teacher's mix isn't a pure-fluid chain
in 176 of 325 states (E-16), so mixtures probably need the recipe hint.

**Q-12 Can we do AlphaGo Zero style instead: search plus self-play, with no planner prior?** ·
**Open → plan 4.2** (small MCTS test at decision time)
MCTS needs a leaf value ("finish from here" exists, E-18) and a prior (the network's policy,
which only exists after the warm start). It was considered and deferred in D-30, D-32 and
D-34. It's favoured if headroom exists, because lookahead targets the sharing pass's known
miss (Q-55).

**Q-13 Are there benchmark problems the planner can't solve, where the agent is the only
option?** · Closed (by construction and a check, E-28, D-35)
No, not at our ranges. The generator keeps only problems a construction can solve, and the env
refuses problems it can't price. Only 1 val draw was discarded (`val_s5_v2`) and 3 of 48 in E3.

**Q-14 Do cheap planner variants already take the headroom?** · Closed (experiment, E-28)
Much of it, yes. Lookahead 3 and extra target orders found most of the improvements on the
medium and val sets. So the bar is the **best planner**, not the plain pass (D-35).

**Q-15 Can RL beat its teacher at all?** · Closed (experiment, E-19, E-22)
Yes, on s1: the mean gap went from −5% to −16% after imitation ended, and the largest win was
`val_s1-024` at 77 vs 213. It's not a uniform win (dearer on 14/40). On s2 it didn't improve
the cost (mean ratio 1.035).

## C. Environment, reward and penalty

**Q-100 Where is the agent better than the planner on stages 1–2?** · Closed (experiment, E-22,
E-28)

Against the best planner (`notes/headroom/records.jsonl`, E-22 weights, greedy):

| | cheaper | equal | dearer | failed |
|---|---|---|---|---|
| `val_s1_v2` | 7 | 19 | 14 | 0 |
| `val_s2_v2` | 8 | 12 | 14 | 6 |

- **Wins** are on problems where the planner draws many sample units (5–7 on s1). The saving
  is mostly a multiple of 64, so the agent draws 1–3 fewer sample units. Examples:
  `val_s1-003` 209 vs 402, `val_s1-024` 77 vs 213, `val_s2-007` 373 vs 458.
- **Losses** are mostly 1–6 units (a little extra waste). A few are large: `val_s1-016` 455 vs
  325, `val_s1-018` 205 vs 134, `val_s2-023` 459 vs 291, `val_s2-028` 845 vs 635.
- **On the proved small problems** (E2 s1/s2) it never beats the planner, except on
  `mix-one-d1-loose-14` (330 vs 385; the optimum is 96).

The agent's sample units weren't recorded, so "fewer sample units" is inferred from the cost
differences.

**Q-101 Has the agent been fully trained?** · **Open → plan 3.5, 3.6, 5.4**
No. The only trained agent is E-22, a test run:
- one seed, 3M steps;
- stages 1–2 only, half and half, with no promotions, so it wasn't a regime test;
- imitation faded to 0 at 1M steps, then pure PPO.

The regime was never chosen (pilots 1–2 chose none, E-9, E-10). Full training comes once stages
3–5 are learnable: warm start + PPO on s1–s3, s1–s4, s1–s5 (3.5), then regime pilot 3 with the
user's curriculum (3.6), then the full run with 3 seeds (5.4). Every result so far, including
the mini benchmark (2.4), is from the E-22 agent. It gets re-run on the trained agent.

**Q-102 Does the agent learn beyond copying the teacher?** · Closed (experiment, E-22, E-25,
E-28)

Yes. After the imitation term reached 0 it trained 2M steps on reward alone.
- **Cost:** on s1 its mean gap moved from about −5% to −16%, past the teacher's −13.3%.
- **New plans:** some are cheaper than anything the teacher or the planner variants make, and
  the teacher never showed them (`val_s1-024` 77 vs 213).
- **New habit:** it dropped the teacher's discards (probability 0.85–0.96 → 0.00–0.04).

But learning also made it worse in places: dearer on 14/40 problems per stage, and the s2
decay. It still reads the planner's recipe hint at inference (Q-9, Q-11).

**Q-103 Is the mini benchmark's problem set big enough and unseen?** · **Open → plan 2.4**
Adding the problems from each paper's own examples and ~100 fresh s1 and s2 draws (user,
2026-10-27). The val sets and paper problems are unseen by the agent. Fresh draws may overlap
its training draws, so they're reported separately.

**Q-16 Round mixes that can't be represented, or make them illegal?** · Closed (decision, D-1)
Illegal. Rounding makes the arithmetic inexact and breaks the oracle.

**Q-17 What's the reward?** · Closed (decision, D-2, D-3)
−cost per step, γ = 1, no terminal bonus, with exact fractions. V* is then minus the optimal
cost, which can be checked.

**Q-18 Would progress shaping help (reward += the planner's finish cost before − after the
move)?** · **Open → plan 3.5, second fallback**
Deferred in D-34. It costs one planner call per step: ~5 ms on s2, ~2 s on s4 (E-18).

**Q-19 How do we handle a huge, mostly illegal action space?** · Closed (decision, D-4)
Mask illegal actions. Each candidate carries exact afterstate features computed by the env.

**Q-20 How should an unfinished episode be charged?** · Closed (decisions D-5 → D-26 → D-32)
First, the construction's cost from scratch. Then ×2 after F-11. Now ×2 the sharing pass's
finish from the final state (`share` penalty).

**Q-21 Why did pilot 1 learn to give up?** · Closed (experiment, F-11)
Failing cost only ×1.09–1.33 of the best known plan. So any finished plan more than 9–33% worse
earned less than giving up. Fixed by D-26.

**Q-22 After the fix, does finishing always pay?** · Closed (experiment, E-11, E-19)
Under the construction penalty, failing cost a median 6.0× (s1) and 5.1× (s2) the best known.
Under the `share` penalty a failed episode can cost less than the best known, which is correct
when its moves were good. So the test bar became per state (D-32).

**Q-23 Did the old penalty ignore droplets already built?** · Closed (experiment, E-18)
Yes. Finishing from the state costs a median 4% (s1) to 54% (s4) of what the old penalty
charged. Fixed by D-32.

**Q-24 Does the `share` penalty fail to charge leftovers, and does it matter?** · Closed
(experiment, E-24, E-25 D4)
It's a real inconsistency: 17–62 units, 2–12% of the penalty. It's too small to explain the
hoarding.

**Q-25 Should waste be charged up front?** · Closed (decision, D-34, after E-27)
Not adopted. Hoarding turned out to be a symptom (Q-77).

**Q-26 How long should an episode be allowed to run?** · Closed (decisions D-15, D-27; F-12)
A depth rule computed from the problem, counting (fluids − 1) mixes per delivery and floored
at 2× the witness plan.

**Q-27 Should the cost weights be an input to the network?** · Closed (decision, D-12)
No. They're fixed at 64/8/1. Preference conditioning is later multi-objective work.

**Q-28 Does sample cost (64) dominate small plans?** · Closed (decision, D-25)
Yes, so report sample units saved next to cost.

## D. Problem distribution and benchmarks

**Q-29 What ranges of M, F, T, demand, storage and accuracy?** · Closed (decision, D-11)
The widest the literature supports. SPEC 1.3.

**Q-30 Droplet volume up to M/2 or up to M?** · Closed (decision, D-16; F-7)
Up to M. M/2 was only where our own constructions stopped.

**Q-31 Does a storage limit actually change what good plans look like at our sizes?** ·
**Open → plan 4.1**
For constructions it rarely binds: 1 of 180 problems (E-7). But 5 of the 8 headroom cells are
tight storage (E-28), so it matters for optimal plans. It still needs measuring on trained agents.

**Q-32 Can training draw benchmark problems?** · Closed (fix, F-5, D-24)
Not any more. `exclude` is wired, the hash is chip-agnostic, and a test fails if it's unwired.

**Q-33 How are stages built, and which cell is held out?** · Closed (decisions D-19, D-20)
One constraint is added per stage (dilution → mixture → demand → many targets → storage). The
held-out cell is dilution + many targets + demand + storage, so the headline mixture cell is
still trained.

**Q-34 Paper benchmarks, tiers 2–6.** · **Open → plan Phase 6**
Not built yet.

**Q-35 Total output volume as an input (TVODA style)?** · Parked (D-13)
It needs a core change, and no step needs it yet.

**Q-36 Is a 40-problem val grading reliable?** · Closed (experiment, E-21)
The noise is about ±6 points per grading. E-20's 95% on s1 was noise; repeated runs give
70–77%. Trust a trend over several gradings, never one.

## E. Exact search, bounds and grading

**Q-37 What does "proved optimal" mean, and how does exact search differ from our planner?** ·
Closed (by construction, SPEC 10; E-1, E-28)
Exact search is A* with an admissible lower bound plus branch and bound from a starting plan.
"Proved" means it exhausted the search space, so no cheaper plan exists. On a timeout it
returns only a lower bound. The sharing pass is a fast greedy constructor that proves nothing.

**Q-38 Can exact search grade medium problems?** · Closed (experiment, E-3, E-6, E-28)
No: 0 of 45 medium problems even with 600 s. The search space is the limit, not the bound.

**Q-39 Is the lower bound admissible, and is it tight?** · Closed (experiment, F-14, E-8, E-6,
E-17)
It's admissible after the 0.21 fix (0 failures over 300 problems). It's tight (within 5%) when
the optimum wastes almost nothing, and loose on single targets and mixtures. A val-set ratio
certifies a plan only when it's near 1 (lesson 10).

**Q-40 Is the sharing pass near-optimal on many-target mixtures?** · **Open → plan 2.3**
Uncertified. On val_s4/s5 it sits at 1.65–3.58× the bound, with 22–72 units of waste (E-28).

**Q-41 What's "best known" in every grading?** · **Open → plan 2.1** (decision made, D-35;
answer keys `_v3` not built)
The cheapest of every producer, the agent's plans and the proved optima. The bar to beat is the
best planner (cheapest of `share`, `share_look3`, `share_orders`).

**Q-42 Can we prove more optima on the val sets?** · **Open → plan 2.2**
Exact search seeded with the best planner, with longer budgets.

**Q-43 Is a mean gap over solved problems a fair number?** · Closed (lesson, F-11, §8)
No, on its own it's biased toward easy problems. Always report success next to gap.

## F. Teachers and planners

**Q-44 Is the old construction a good teacher?** · Closed (experiment, E-5, D-31)
It's an expert only on single-target, loose-storage problems (stages 1–2). On stages 3–5 it's
50–135% above the optimum, because it never shares droplets.

**Q-45 Is the sharing pass better?** · Closed (experiment, E-17)
Yes. It's optimal on 51/65 proved small problems against the construction's 34, never dearer,
and cheaper on 237 of 352.

**Q-46 How good is the teacher on stages 3–5?** · Closed (experiment, E-17, E-28)
On small proved problems it's at the optimum on all s3/s4 but one, and s5 median 1.00. Its
weak spots are tight storage and many targets. On many-target mixtures it's uncertified (Q-40).

**Q-47 Is the teacher optimal everywhere on stage 1?** · Closed (experiment, E-19, E-22)
No. On large denominators a cheap buffer-heavy start beats it (`val_s1-008` 168 vs 475;
`val_s1-024` 77 vs 213).

**Q-48 Do lookahead and extra target orders improve the sharing pass?** · Closed (experiment,
E-17, E-28)
Yes, often slightly. Lookahead 3 costs ~15× the time.

**Q-49 How do the agent and planner compare with published heuristics, and could one be a
better teacher?** · **Open → plan 2.4** (stages 1–2, D-36), **then 6.3** (everything)
No comparison exists yet. Methods can't be rebuilt from memory (SPEC 10.4), and E-17 task 3
was skipped for that reason. Step 2.4 implements 5–6 single-target methods from their PDFs.
Multi-target ones (Kundu 2023, Fukui 2024, MTCS, DDMT) wait for 6.3.
**First results (E-29):** on stage 1 the best published methods (FacDA, MRCM) are level with
our planner and the agent. On stage 2, MRCM is clearly cheaper than our planner: mean ratio
0.66 on the val problems it takes, and 1.8% above the optimum against our planner's 43% on 7
proved problems. The agent's costs on the new sets still need the workstation run.

**Q-104 Should MRCM (or its idea) become part of our planner, and the teacher, for
mixtures?** · **Open, no step yet** (raised by E-29)

**Q-105 Does the agent lean too much on the greedy planner through the recipe hint, and can
the hint come from the problem alone?** · **Open, no step yet** (user, 2026-10-27)
What we know:
- Without any hint, copying fails (E-15).
- A planner-free chain hint gets s1 to 67.5% (E-16).
- The agent uses the hints as a map, not a script: it doesn't follow the planner's next move
  (E-30), and it beats the planner on some s1/s2 problems (Q-100).
- But on mixtures it ends up no better than the planner, which MRCM beats (E-29). So the
  recipe hint may cap it there.

Candidate: replace "mixes left along the planner's recipe" with the exact fewest mixes from
a candidate's result to each target, computed from the problem alone.

**Feasibility (E-31).**
- *Dilutions:* yes. A depth-4 pure-fluid chain covers every teacher mix on s1, at under 0.1 s
  per target.
- *Mixtures:* not this way. Coverage is under half, and the search explodes on blends.

Built as SPEC 0.28 `hint_source: problem`.

**Copy test (E-32): s1 holds, s2 drops, as expected.** Problem-only hints: s1 greedy success
100% (planner hints 85–87.5%); s2 mix agreement 37–39% (87–91%) and success 32.5% (45–67.5%).
So the planner can leave inference on dilutions but not, so far, on mixtures. Still open:
whether search at decision time (plan 4.2) can stand in for the planner's recipe on mixtures.

**Q-50 Does the best planner as teacher do better on stages 3–5 than the plain pass?** ·
**Open → plan 3.4** (decision D-35 made, effect unmeasured)

**Q-51 Can the planner finish from a mid-episode state, and how fast?** · Closed (experiment,
E-18)
It always finishes (0 failures in 1,462 states): 4–9 ms on s1–s3 and s5, 1.8 s on s4.

**Q-52 Does the teacher's quality or the hints explain copy success?** · Closed (experiment,
E-21)
The hints. With hints, the old teacher copies as well as the sharing pass. The better teacher
only lowers the cost.

**Q-53 Should we imitate the teacher everywhere?** · Closed (decision, D-31)
No. Only where it's near-optimal, and the imitation fades so RL can leave it.

**Q-54 Exact optima as demonstrations?** · Closed (decision, D-31)
Only where proved. Small problems only.

**Q-55 What does the sharing pass miss?** · Closed (experiment, E-17)
It doesn't build two intermediates ahead and blend them (1/4 at volume 3 on M = 3: 72 vs 136).

## G. Network, observation and hints

**Q-56 One network for any F, T, M, storage?** · Closed (decisions D-8, D-9; E-13)
Yes. A set encoder with fluid tokens and relation-biased attention, 0.88M parameters,
invariant to reordering.

**Q-57 Why did train and eval mode give different outputs?** · Closed (bug, F-6)
PyTorch's encoder layer, once per-head biases were added. Replaced by a hand-written layer.

**Q-58 Can the network find the right first mix from compositions alone?** · Closed
(experiment, E-15, E-16)
No. First-mix agreement was 24% (s1) and 6% (s2), even on trained problems. The arithmetic
(why 1/8 leads to 33/40) is invisible to it.

**Q-59 Is the agent lost off the hint map, or does it see the right move and not take it?** ·
Closed (experiment, E-30)
It isn't lost. In failed episodes the planner's next move is hinted 67–100% of the time, but
the agent picks unhinted mixes (17–50% of its mixes hinted, against 70–98% when it solves)
with low confidence. The rule says "will not commit" on s2, s4, s5 and held-out. The held-out
failures are different: the agent picks hinted moves confidently and still fails, because the
map has no cross-target information.

The rule as first written (kept for the record):
The rule was fixed in advance. Lost: the planner's next move is hinted on < 50% of late steps
in failed episodes, and less than half as often as in solved ones. Doesn't take it: hinted on
≥ 70% of failed steps, but median probability < 0.3.

**Q-60 What hints do stages 3–5 need?** · **Open → plan 3.3** (designed after Q-59)
Candidates: recipes from the best planner across targets; a map extended on the fly from
compositions the agent makes; the planner's next move per state (too slow on s4 as it stands).

**Q-61 How are hints made, and do they look at the chip?** · Closed (by construction, D-32,
D-34, D-35)
10 of 27 candidate columns, on Mix candidates only:
- Recipe hint: fewest mixes from the result to an open target along the planner's
  single-target recipes, capped at "4 or more".
- Chain hint: depth-2 pure-fluid chains, pure arithmetic.

Both are fixed per target. They ignore what's on the chip and reuse across targets
(`val_s2-000`'s map has 15 compositions, 9 of them "4 or more").

**Q-62 Can the whole planner be run per candidate as a hint?** · Closed (experiment, E-18)
No. 4 ms × 200–800 candidates per step is too slow.

**Q-63 Would more data or a bigger network fix copying?** · Closed (experiment, E-15)
No. 3.3× the data moved train agreement only from 0.55 to 0.64.

**Q-64 Does an exact "mixes left" feature fix copying?** · Closed (experiment, E-16, E-20)
Yes, it was the missing information. s1 greedy went 25% → 67.5% with a crude feature, and
s2 went 0% → 65–72.5% with the hint columns.

## H. Training method

**Q-65 Why did pure PPO stall?** · Closed (experiment, E-11, E-14)
Exploration. Random play solves 2/1,200 s1 and 0/1,200 s2 problems, with 200–800 legal moves
per step and 6–12 correct moves needed in a row. A greedy "get closer" policy solves 20% of
s1 problems: the agent can see the distance but lacks the algorithm.

**Q-66 Imitation warm start, reward shaping, easier sub-stages, or MCTS?** · Closed (decision,
D-30)
Imitation warm start, then PPO (AlphaGo-style). The others were deferred.

**Q-67 Can the network copy the teacher?** · Closed (experiment, E-15, E-20, E-21)
Without hints, no (s2 0%). With hints, s1 ~70–77% and s2 ~60–70% greedy success.

**Q-68 Does warm start + PPO work?** · Closed (experiment, E-22)
Yes. s1 100%, s2 85% (peak 97.5%), from the first 100k steps. Pilot 2 had s1 at 42% and s2
at 0% after 10M steps.

**Q-69 Should the imitation term fade or stay?** · Closed (decision, D-31; D-33)
Fade to 0. A constant anchor was dropped as a crutch. It stays the third fallback in plan 3.5.

**Q-70 Is the critic any good?** · Closed (experiment, E-22)
Explained variance 0.88–0.89 at the end of the warm-start run.

**Q-71 Pure RL vs RL with a teacher on stages 3–5?** · **Open → waiting on the user** (plan §5;
was D-31 (6))

**Q-106 Can learning plus search find plans cheaper than the planner, without the planner at
play time?** · Closed (experiment, E-33)
Asked because the agent alone ends level with the planner (E-22, E-29) and every fix so far
improved the planner or its hints. `scripts/search_check.py` compares the greedy agent, search
with planner finishes (network or flat prior) and search with the network's own value (no
planner) on 23 proved planner misses; the reading is fixed in D-37.
**Answer (E-33, 2026-10-28): no, not with today's network.** Without the planner, search with
the network is below the planner on 3 of 23 (1.49 × optimum). Lookahead with planner finishes
finds most misses even with a flat prior (20 of 23, 13 at the optimum), and the network's prior
makes it worse (16 of 23). Learning has not yet added anything search with the planner does not
find.

**Q-107 Can the network learn the plans search finds, and then beat the planner without
it?** · **Open → plan 4.2, expert iteration round 1** (D-38; user, 2026-10-28)
E-33: search with planner finishes finds most of the planner's misses, slowly; the network,
taught by the planner, adds nothing. Round 1 teaches it the search's plans, with a
planner-taught control arm; pass bar in D-38.
**Round 1 (E-34): partly.** The search-taught network guides search as well as a flat prior,
~30× faster, and beats MRCM's cost on 8 of 13 mixtures with planner finishes; alone, or in search
without the planner, it is still not below the planner (greedy 1 of 23, no-planner search 5 of
23), and it got worse on mixtures (val_s2 47.5%). Its value head was not trained.
**Next: round 2 (D-40)**, with the value head trained on what the search observed.

**Q-108 Can the value head learn to judge positions, so the planner leaves the search?** ·
**Open → plan 4.2, round 2** (D-40; user, 2026-10-29)
The no-planner search runs on the value head, which PPO trained on stages 1-2 only and round 1
did not train at all (E-34: 5 of 23). Round 2 trains it on the cheapest remaining cost the
search saw, with a policy-only control; bar (a) in D-40.
**Round 2 (E-35): yes on the small proved problems, not on mixtures.** With the value head
trained, the no-planner search is below the planner on 17 of 23 (policy-only control 7; held-out
value error 269 vs 642 cost units), but fails to finish 9 of 13 MRCM problems. **Next: round 3**
with the value head as judge; handling unfinished problems is to be decided.

**Q-109 Does the AlphaZero loop climb?** · **Open → plan 4.2, round 2** (D-40; user,
2026-10-29)
Round 1 was half a turn (search taught, the network copied); the new network has not yet guided
a search that teaches it again. Bar (b) in D-40.
**Round 2 (E-35): one turn, slightly.** The round 2 network alone solves 22 of 23 at 1.390 ×
optimum (round 1: 21, 1.447); s3-s5 success 60.8% (59.2%). Bar met narrowly; more turns needed
to call it a climb.

**Q-110 Why did s2 drop to 47.5% when copying the search's plans?** · **Open, no step** (E-34)
A guess, unchecked: a searched opening spliced onto the planner's finish is harder to copy
than the planner's consistent style. Round 2 copies visit counts instead of one plan; look
again if the drop persists.
**Round 2 (E-35): it persists** (V 45%, policy-only P 60%), so copying visit counts instead of
one plan did not cure it.

**Q-112 Does search beat the planner on full-size stages 3-5?** · **Closed** (E-36)
(user, 2026-10-30)
Every search result so far is on small problems (set A) or stage 2 mixtures (set B); the claim
is about full-size stages 3-5. D-41 runs the search test on 99 val s3-s5 problems with round 2's
network before round 3 is built.
**Closed (E-36): with the planner yes, without it no.** Search with planner finishes is below
the best planner on 85 of 99 (16% cheaper on average), about the same with a flat prior (83);
with the value head alone 29 of 99, and 47 not finished.

**Q-113 Is search's gain on full-size s3-s5 lookahead, or only more time?** · **Closed** (E-37;
user, 2026-10-30)
E-36's search ran up to 900 s against the planner's ~1 s. D-42 gives the planner the same time
(random restarts) and checks whether the search still wins.
**Closed (E-37): lookahead.** With 900 s of random restarts the planner gains 1-2%; the search
stays ~14% cheaper, on 79 of 99.

**Q-114 Does the network make search faster on full-size s3-s5?** · **Closed** (E-39; user,
2026-10-30)
At 900 s the network adds no quality (E-36: 85 vs 83 of 99); the gain is lookahead (E-37). Its
remaining role as a guide is speed. D-43 races search with and without it at 10 s, 60 s, 900 s.
**Closed (E-39): no on set C** (tied at 10 s and 60 s); **yes on the mixtures** (B: 0.591 vs
0.675 of the planner at 60 s, still better at 900 s), where the planner's finish is weak.

**Q-111 Is the network itself what limits stages 4-5?** · **Parked** (user, 2026-10-29: for the proof of concept the network is unlikely to be the bottleneck; architecture and hyperparameters are tuning, after the training and search method is settled. Exception: if D-40's bar (a) fails, look at the value error on the training problems first)
Never questioned since E-16 (fixed by input features, not architecture). E-34: copy agreement on
the val sets falls to 47% / 46% on s4 / s5 (s1-s3 68-79%). Three explanations fit: too little
s4-s5 data (round 1 had no stage 3-5 draws), an inconsistent teacher, or a network that cannot
represent them (3 layers of width 128; the value head reads only the global token). Check after
round 2: per-stage agreement and value error on the training problems against validation. Low
on training too → architecture (deeper, or a value head attending over the tokens); high on
training only → data or generalisation.

**Q-72 Self-imitation if cost stalls near the teacher?** · **Open → plan 4.3**

**Q-73 Are the results stable across seeds?** · **Open → plan 5.1**
Every run so far uses one seed (the pilots used 2).

**Q-74 Can the agent learn a single mixture problem when it doesn't have to generalise?** ·
Parked (E-14 open item)
The warm start answered the bigger question first.

## I. Stage 2 failures

**Q-75 Why does s2 decay after imitation ends?** · Partly closed (experiment, E-23–E-27)
The 6 failures are exactly the 6 longest mixture plans (23–41 teacher steps). They hoard
droplets and dither until the step limit. The root cause is Q-78.

**Q-76 Do the failures loop, hit dead ends, or run out of steps?** · Closed (experiment, E-24)
They run out of steps while hoarding (10–25 droplets, mean p 0.31–0.64). The copy network
loops instead.

**Q-77 Did RL learn to avoid discarding, and is clutter the cause?** · Closed (experiment, E-25,
E-27)
RL unlearned discards (probability of the teacher's discard 0.85–0.96 → 0.00–0.04). But a
clean chip and twice the steps rescue 0 of 6. Hoarding is a symptom.

**Q-78 What is the root cause of the long-plan failures?** · **Open → plan 3.4, 3.5**
Narrowed by E-30: the right move is on the map, but in long episodes the agent drifts onto
unhinted mixes and doesn't come back. Next to test: whether training on s3–s5 (copy tests,
then PPO) keeps it on the map, and if not, search at decision time (4.2). The same failure appears on every untrained mixture cell of stages
3–5 (E-26).

**Q-79 Does s1 dominate the shared update once imitation stops?** · Parked (E-22 candidate
cause, never tested)
Stage 2 polish is parked (D-35).

## J. Stages 3–5

**Q-80 Does the stage 1–2 agent transfer zero-shot?** · Closed (experiment, E-26)
Single-target dilutions transfer at the teacher's cost: s3 32/40, s4 11/40, s5 16/42, held-out
20/40. Many-target mixtures: 0/32.

**Q-81 Can we copy the teacher on problems with identical droplets?** · Closed (fix, F-16,
SPEC 0.27)
Fixed with a uid map that matches moves by canonical key.

**Q-82 Can the network copy the best planner on s3, s4, s5?** · **Open → plan 3.4**
Pass bar: agreement ≥ 70% and greedy success ≥ 60%.

**Q-83 Does warm start + PPO pass on s1–s3, s1–s4, s1–s5?** · **Open → plan 3.5**
Pass bar: ≥ 90% on s3, ≥ 80% on s4–s5, median cost within 5% of the best planner.

**Q-84 Can stage 4–5 demonstrations from the best planner be made fast enough?** · **Open →
plan 3.4**
The best planner is up to ~15× slower than the plain pass, and demonstrations are made once in
one process. They need generating in parallel or caching.

**Q-85 Is the `share` failure penalty affordable on s4 (~2 s per failed episode)?** · Parked
(D-35)
The plain pass is kept for the penalty. Revisit if stage 4 training is slow.

## K. Training regime and curriculum

**Q-86 Does the regime matter while nothing is learned?** · Closed (experiment, E-9, E-10)
No. All three plateaued on the same walls (s2 0%).

**Q-87 Which regime: sequential, expanding or mixed?** · **Open → plan 3.6** (regime pilot 3)
Expanding was best in pilot 2 and is the stand-in until then.

**Q-88 What's the curriculum?** · **Open → plan 3.6** (the user designs it before pilot 3)

**Q-89 An automatic difficulty ladder?** · Closed (decision, 2026-10-16, D-21)
Rejected. Stages are a diagnostic, not the goal.

**Q-90 How are regimes judged and stages promoted?** · Closed (decision, D-21, D-28)
The rule was fixed before running. Promotion happens on validation success, or is forced and
logged.

## L. Literature and baselines

**Q-91 Do robust-aliquot (Poddar 2020), Dong 2024 or e-MRCM narrow the claim?** · **Open →
plan 6.1**
Partial: the unmerged `paper-checks` branch reads the first two as MEDA multi-target dilution
with no demand or storage. e-MRCM is still unread.

**Q-92 Is e-MRCM an RL method?** · Closed (literature, L-3)
No, it's a greedy heuristic.

**Q-93 Can we compare with baselines that use n-input mixes?** · Closed (by construction, D-29)
Yes: a chain of 2-input mixes, n − 2 extra mixes, no extra waste.

**Q-94 Can published methods be rebuilt from memory?** · Closed (rule, SPEC 10.4)
No. Only from the paper.

## M. Platform and supervisor

**Q-95 What are the real platform's M, mixer footprint, reservoir count and cost weights, and
can the router serve as a cost oracle?** · **Open → supervisor** (§8 of the paper log)

**Q-96 Do the generalisation claims hold on unseen sizes and constraints?** · **Open → plan
5.2, 6.4**

## N. Early rungs and checks

**Q-97 Can tabular Q solve tiny problems?** · Closed (experiment, E-2, D-7)
It solves 9 of 10 at the optimum and breaks at M = 8. It's kept as a canary, not as a method.

**Q-98 Did the prototype agent really beat the oracle?** · Closed (bug, F-1)
No. A dead end ended the episode cheaply. Fixed by pricing dead ends.

**Q-99 Is reported throughput real?** · Closed (bug, F-8, E-12)
No, `sps` left out the update and overstated speed ~2×. Every phase is now timed.
