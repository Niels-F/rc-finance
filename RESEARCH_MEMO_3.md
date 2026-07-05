# Research Memo #3 — The Reservoir Computing Frontier: NG-RC, Random Projections, and Self-Aware Forecasts

**Desk:** Systematic Macro / Vol · **Status:** New champion model — recommend COMBO for the productionization study

## 1. Objective

Projects 1–2 established a replicable ESN edge in volatility space. Project 3 asked whether the *frontier* of reservoir computing beats our incumbent: (a) Next-Generation Reservoir Computing (Gauthier-style nonlinear vector autoregression) and its 2025 random-projection variant, and (b) meta-predictability — can the model detect when it is trustworthy? All tests on the established 28-year walk-forward VIX suite (OOS 1998–2026, horizons 1/5/10/21 days, benchmarks RW/HAR, HAC-adjusted Diebold–Mariano tests).

## 2. Iteration log

| Step | Design | Result | Diagnosis |
|---|---|---|---|
| 3a-i | Polynomial NG-RC: delay embedding (0/1/5/21d) + all quadratic monomials + ridge | **Underperforms HAR at every horizon** (h=21: 0.090 vs 0.109) | Quadratic monomials of noisy inputs amplify noise; NG-RC's chaos-benchmark success assumes clean deterministic dynamics — consistent with the complexity-calibrated-benchmark caution in the literature |
| 3a-ii | RP-NGRC: random tanh projection (dim 400) of the same delay embedding | **Ties the recurrent ESN at every horizon** (h=10: 0.0874 vs 0.0871; h=21: 0.1228 vs 0.1234) | Key structural insight: recurrence contributes ≈ nothing for vol forecasting; the value is nonlinear mixing of ~1 month of explicit history. RP-NGRC is ~10× cheaper: no washout, no spectral radius, no state continuity to manage |
| 3b | "Knows when it knows": 8-projection ensemble disagreement as confidence signal; hard gates (disagreement, regret) | Monotone conditional pattern at h=21 (RC beats HAR when seeds agree: 0.132 vs 0.112; loses when they argue: 0.085 vs 0.098). But Spearman(disagreement, error) only ≈ +0.03; **hard gates destroy value**; regret gate helps only at h=5 | Confidence signal is real but weak → binary switching on a weak signal is a noise trade |
| 3b-flag | Extended ridge grid selected λ=10⁵ on the inner split → OOS *loss* vs λ=10⁴ | Hyperparameter selection variance is itself a risk | Never trust a single stale validation split; never let a grid pin at its boundary silently |
| 3c | **COMBO**: λ-averaged RP-NGRC (grid 10³/10⁴/10⁵), soft-blended with HAR via trailing-252d inverse-MSE weights (walk-forward legal); disagreement as optional modulator | λ-averaging fully recovers best-λ performance with zero selection. COMBO vs HAR: **DM t = +1.94 / +2.26 / +2.08 at h = 5/10/21**, beats every parent at h=1/5. Disagreement modulation adds nothing beyond trailing-MSE weighting | Averaging beats selecting; soft blending beats hard switching; ensemble spread is informative but redundant given realized-performance conditioning |

## 3. Conclusions

1. **New champion:** λ-averaged RP-NGRC + inverse-regret HAR blend. First model in the program to clear conventional significance against HAR across the 5–21d horizons where forecasts monetize (term structure, risk limits, margin).
2. **Structural finding worth publishing internally:** for volatility, reservoir *recurrence* is dispensable — a feedforward random projection of an explicit ~21-day delay embedding captures the nonlinearity. This collapses the production surface: deterministic-given-seed, no dynamical stability concerns, trivially parallelizable, interpretable memory.
3. **Meta-predictability verdict on the original question ("detect when RC is good"):** yes, weakly — ensemble disagreement conditions relative skill in the right direction at long horizons — but the deployable version of "self-awareness" is trailing realized regret, used as a *continuous* blend weight, not a switch.
4. Negative results retained on the record: polynomial NG-RC (noise amplification), hard gating (weak-signal switching), single-split hyperparameter selection (λ boundary trap).

## 4. Limitations

Single underlying (VIX); log-space MSE as the sole loss (QLIKE robustness check pending); blend weights use one trailing window length (252d, not tuned — deliberately); DM t≈2 across correlated horizons is strong but not overwhelming; economic-value test still awaits a long clean daily equity/RV sample.

## 5. Next steps

- QLIKE + Model Confidence Set robustness pass on COMBO.
- Cross-asset replication: rates vol (MOVE-style), FX vol, single-name — 20+ series turns t≈2 into a portfolio-level verdict.
- Term-structure monetization study: h=10–21 edge → VIX calendar-spread signals, with realistic futures costs.
- Reservoir states as features into the desk's production vol stack (the lowest-friction deployment path, now with RP-NGRC's cheap features).

*Artifacts: `p3_ngrc.py`, `esn_frontier_project3.png`; Memos #1–2 unchanged.*
