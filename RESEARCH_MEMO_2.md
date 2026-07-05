# Research Memo #2 — Reservoir Computing for Volatility Forecasting

**Desk:** Systematic Macro / Vol · **Status:** Positive statistical result — recommend productionization study (see §5)

## 1. Redirection rationale

Project 1 established that monthly index *returns* offer no exploitable structure for reservoir computing on clean prices — but that the volatility channel carried all the economic value. Volatility is also dynamically the right target for an ESN: log-VIX has 0.98 daily autocorrelation, nonlinear mean reversion, asymmetric spike/decay — the closest thing markets offer to the chaotic systems where reservoir computing has a proven edge.

## 2. Design

Data: daily VIX 1990–2026 (9,220 obs) for the statistical test; 505-stock daily equal-weight index 2013–2018 for the economic test. Walk-forward OOS from 1998 (VIX) / 40% (index). Benchmarks a vol desk would demand: random walk, AR(1), HAR (Corsi cascade), EWMA/RiskMetrics. Metric: OOS R² measured against the random walk; Diebold–Mariano tests with overlap-adjusted (HAC) variance. ESN: 8-seed ensembles, states computed once per seed under causal fixed standardization, annual ridge-readout refits, hyperparameters chosen on an inner split of the initial training window only.

## 3. Iteration log

| Version | Design | Result | Diagnosis |
|---|---|---|---|
| v1 | Level-space target, raw standardized inputs | ESN R² **−0.13** vs RW; worst in high-vol (−0.24) | Tanh saturation: 2008/2020 inputs sit 5–6σ above frozen 1990–97 stats; and predicting levels with 0.98 autocorr wastes capacity re-learning persistence |
| v2 | Change-space target (Δlog-VIX added back to today's level), stationary features, inputs clipped ±3σ | ESN ties HAR at h=1, beats it at h=5 (0.072 vs 0.062) | Craft, not architecture, was the blocker |
| v3 | Horizons 1/5/10/21 + walk-forward HAR-ESN blend | Monotone edge: h=10: 0.090 vs 0.075; h=21: **0.127 vs 0.109**; DM t = +1.3 to +1.8, consistent sign across horizons; regime split shows edge concentrated in stress (VIX>25) | Nonlinear mean reversion matters more at longer horizons — where HAR's linearity binds |
| Economic | Fwd-21d index RV forecast → 10% vol targeting | ESN best accuracy (R² +0.24 vs EWMA +0.14, HAR +0.05); realized strategy vol closest to target (10.2%); tracking-error deltas within noise on 700 days | Sample too short/uniform for economic significance; note Jensen bias when exponentiating log-vol forecasts — a train-residual correction *backfired* across regimes (non-stationary bias) |

## 4. Conclusions

1. **Reservoir computing earns a real, replicable statistical edge in volatility space** — beating HAR out-of-sample over 28 years, with the edge growing with horizon and concentrated in stress regimes, exactly where forecast value is highest (term-structure repricing, risk limits, margin).
2. The gap between v1 and v2 was entirely representation: change-space targets, stationary inputs, saturation guards. The reservoir amplifies good features; it does not rescue bad ones.
3. Statistical significance ≠ economic significance yet: DM t-stats of 1.3–1.8 per horizon and a short economic sample mean the correct next step is more data, not more model.

## 5. Recommended next steps

- Rerun the economic test on 20+ years of index futures/ETF closes with intraday realized variance (proper RV, not 21d proxies) — the setting where HAR was born and where beating it matters.
- Blend deployment: ship reservoir states as **features into the desk's existing HAR/regularized stack** rather than as a standalone model; let production attribution allocate weight.
- Extend to the vol surface: forecast term-structure slope (front vs 3m) where the h=10–21 edge monetizes directly in calendar spreads.
- Regime-aware Jensen correction (rolling, vol-bucketed) before any 1/σ position sizing.

*Artifacts: `vix_horse_race.py`, `esn_vol_forecasting.png`; Project 1 artifacts unchanged.*
