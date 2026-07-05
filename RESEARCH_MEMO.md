# Research Memo — Reservoir Computing for Index Timing

**Desk:** Systematic Macro · **Status:** Concluded — do not allocate; redirect research (see §5)

## 1. Thesis

Echo State Networks excel at chaotic systems (Lorenz, Mackey–Glass) because those are deterministic with high signal-to-noise. The hypothesis was that a reservoir's nonlinear fading memory could extract regime structure from equity index returns that linear models miss. Design was defensive from the start: fixed random reservoir + ridge readout only, strict walk-forward retraining, train-only standardization, transaction costs, and honest benchmarks.

Data: S&P 500 monthly (1871–2026, Shiller-style) for the main study; 505-stock daily closes (2013–2018) for the clean-data control.

## 2. Iteration log

| Version | Design | Result | Diagnosis |
|---|---|---|---|
| v1 | Single ESN, sign long/short, 10 bps | Sharpe 0.10 vs 0.38 B&H; IC 0.054 | Symmetric shorting fights the equity premium; single-seed results untrustworthy |
| v2 | 10-seed ensemble, long/flat dead-zone gate, 12% vol targeting | Sharpe 0.41, MaxDD halved — but gate adds **−0.95%/yr (t −2.1)** vs vol-target B&H | All improvement came from vol targeting; ensemble IC fell to 0.019 → suspect seed luck in v1 |
| v3 | Seed-variance study + ridge/leak sweep | Per-seed IC at ridge 0.01: mean −0.004, range ±0.046 → **v1's IC was luck**. Ridge 100 → ensemble IC 0.27 | Strong regularization is essential; but IC 0.27 monthly is implausibly high → audit before celebrating |
| v4 | Best config, IC decomposition + attribution | Sharpe 0.67, gate +1.94%/yr (t +3.4). IC vs raw returns 0.22, survives vol-orthogonalization | Too good → audit the *data* |
| Audit | Autocorrelation + execution-lag stress | Lag-1 autocorr **0.274**, lag-2 ≈ 0 (MA(1) signature of monthly-averaged prices); pred correlates 0.72 with current-month return; with +1 month execution lag the edge flips to **−1.02%/yr (t −1.9)** | **The alpha is an untradeable data artifact.** The ESN faithfully learned "next month ≈ this month," which only exists because prices are time-averaged |
| Control | Same pipeline on clean daily close-to-close data | Autocorr −0.001; OOS IC 0.034 ≈ 0 (SE 0.035); gate t −0.54 | No exploitable signal at the index level on clean prices |

## 3. Conclusions

1. **The reservoir worked; the market didn't cooperate.** RC's strength — reconstructing deterministic dynamics — does not transfer to a series that is, at monthly index level, close to a martingale plus a risk premium.
2. Every intermediate "edge" traced back to something else: seed luck (v1), vol targeting (v2), and a time-averaged-price artifact (v4). A model can only be as honest as its benchmark and its data.
3. The defensive machinery (ensembling, walk-forward, attribution vs naive benchmarks, execution-lag stress, data audits) is what separated a publishable-looking Sharpe 0.67 from a production loss.

## 4. Known limitations

Price returns exclude dividends; costs are stylized (10 bps monthly / 5 bps daily); no borrow/financing modeling; single asset; hyperparameter sweep was small and the clean-data sample short (5 years, one regime).

## 5. Where reservoir computing may still earn its keep (next projects)

- **Volatility and realized-covariance forecasting** — vol dynamics are far more persistent/nonlinear than returns; an ESN vol forecast plugs directly into the vol-targeting layer that demonstrably drives risk-adjusted gains.
- **Cross-sectional equity signals** — one shared reservoir driven per-stock, readout trained cross-sectionally (thousands of effective samples per date instead of one).
- **Higher-frequency microstructure** — order-flow/imbalance series have stronger short-horizon structure, closer to RC's home turf.
- **Feature generator, not oracle** — feed reservoir states as nonlinear features into the desk's existing regularized stack, and let attribution decide if they earn weight.

*Research artifacts: `esn_lib.py`, `backtest_v1–v4.py`, `esn_research_postmortem.png`.*
