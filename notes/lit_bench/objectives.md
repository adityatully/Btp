# What each published method optimises, and what it claims

*2026-10-31. Read from the PDFs in `~/Downloads/sample prepration/` (the five marked "new" were
added that day: e-MRCM, NFSP, MTMD, DDMT, robust aliquot). The other rows repeat `selection.md`
and `notes/track2/e1_coverage.md`. Nothing from memory (SPEC 10.4).*

"Steps" = mix(-split) operations, the literature's stand-in for assay time and split error. No
paper optimises solver run time; some report it.

| Method | Platform, class | Optimises (first) | Also reports | Claim, and against whom | Test protocol |
|---|---|---|---|---|---|
| DFM (Kundu, TCAD 2022) | MEDA, dilution, 1 target | steps, then sample | waste | tree phase of FacDA | worked examples |
| FacDA (same paper) | MEDA, dilution, 1 target | sample (SMT) | steps, waste | min sample on its skeleton | ε 0.001 / 0.005, worked examples |
| Generic (Poddar, TCAD 2022) | any platform, dilution, 1 target | steps (heuristic) | sample | one method for every platform | x/64 at 4X, Table III |
| WSPM (Li, TBioCAS 2017) | MEDA, dilution, 1 target (incremental: 2-4 targets) | sample, then steps, waste | | first droplet-size-aware MEDA method | δ = 1/256 |
| Dong (JETTA 2024) | MEDA, dilution, gradients of many targets | sample, waste | | splitting trees for dyadic gradients | dyadic only |
| MRCM (Liang, ASP-DAC 2019) | MEDA, mixture, 1 target | weighted reactant cost | steps, waste | beats BS, REMIA, CoDOS, WSPM on all three | as e-MRCM |
| **e-MRCM** (Liang, TCAD 39(10) 2020) **new** | MEDA, mixture, **2-5 targets**, N 2-5, demand 1, no storage | **weighted reactant cost**: geometric 1 to 64 (N = 2: sample 1, buffer 0) | steps, waste | beats BS, CoDOS (DMFB) and WSPM; over MRCM on many targets, e.g. N 2, 5 targets: cost -8.06%, steps -9.09%, waste -15.15% | P = 1/512; N = 2 all 511 targets; else 512 random; many targets: 50 random sets per (N, T), averages (Table III); target droplet = 4 aliquots |
| DFMM / STWM (Kundu, DSD 2022) | MEDA, mixture, 1 target | waste (STWM, SMT) | reagent, steps | min waste | worked examples, Fig. 7 averages |
| MTWM (Kundu, TODAES 2023) | MEDA, mixture, many targets, no demand | waste, reagent | | sharing within and across targets; storage only estimated afterwards | |
| **Robust aliquot** (Poddar, TODAES 26(1) 2020) **new** | MEDA, dilution, many targets, aliquot-and-mix, no splits, no sharing | **zero waste, robustness** to split and dispensing errors | sample cost, time | vs TWM, DMRW, REMIA, CoDOS, WSPM, MRCM | real targets (e.g. 358/512 ethanol), 1% / 2% dispensing error |
| **NFSP** (Dinh, ASP-DAC 2014) **new** | DMFB 1:1, dilution, many targets with **droplet counts** | **optimal** sample·cost_s + buffer·cost_b (variant S), or waste (variant W); min-cost flow ILP | steps | optimal: NS 7.8% below REMIA; NW 33% below DMRW | d = 10, all 1023 targets one by one; many targets at d = 9 (Table II) |
| **MTCS / MTR / MTMD** (Shalu, Kumar et al., TODAES 23(4) 2018) **new** | DMFB 1:1, mixture, 1 target / many / **many with demand** | **steps (Tms) and waste** | reagent | MTMD vs MTR: steps -64%, waste -85%; MTR vs repeated Min-Mix -23% / -44% | large set of target ratios, real protocols |
| **DDMT** (Poddar, TODAES 27(1) 2021) **new** | DMFB 1:1, dilution, many targets, **demand, k storage cells** | **steps**, given storage (one mixer) | sample (accepts more than XNFM) | far fewer steps than XNFM (ILP) and repeated DE; log2(Dmax) storage suffices | accuracy 7, synthetic sets of 10 and 20 CFs |

## What it means for our benchmark

1. **Three objective families:** steps (DFM, Generic, MTMD, DDMT), fluid cost (FacDA, WSPM,
   MRCM, e-MRCM, NFSP-S), waste (STWM, MTWM, NFSP-W, robust aliquot). Almost every paper reports
   all three columns, whichever it optimises.
2. **Same objective as ours (priced fluid units):** MRCM, e-MRCM (geometric 1-64, the same family
   as our 64/8/1) and NFSP-S (priced sample + buffer). These are the fair head-to-heads on our
   cost; the others are compared on their own metric.
3. **NFSP is optimal within 1:1 mixing.** Beating it on fluid cost shows what k:l mixing and
   our planning add; losing to it on a 1:1 chip (M = 2) would be a bug (SPEC tier 2 rule).
4. **DDMT trades sample for steps** (its conclusion says so): compare with it on steps and
   storage use, not on our cost alone.
5. **Protocols are random targets at a power-of-two precision (1/512, d = 9-10), averaged.**
   To reuse their printed numbers (tier 3) we must run the same sweeps, not our val sets.
