# Research Memo #4 — Stress Report: λ-averaged RP-NGRC + HAR Blend (COMBO)

**Desk:** Systematic Macro / Vol · **Status:** Stress battery passed — COMBO confirmed as the program's focus; proceed to monetization study

## 0. Why this memo exists

Memo #3 crowned COMBO on walk-forward validation. Validation is not stress testing. Before concentrating research capital, the model faced the four batteries a model-risk committee would demand. Verdict up front: **pass, with two documented soft spots.**

## A. Design-perturbation stress — PASS

Every arbitrary design choice was perturbed one at a time (h=21): projection dimension 200/400/800, delay set shortened and extended, tanh scale 1.0–2.0, two fresh seed families, blend window 126–504d, λ grid shifted ×10 both ways. Result: COMBO R² spans **0.114–0.127 across all 13 variants — every one above HAR's 0.109**. The performance surface is a plateau, not a needle. Seed sensitivity is ≈ nil (0.1229–0.1232), confirming the initialization-robustness rationale for retiring the recurrent ESN. The one sensitive knob is λ-grid *location*: over-regularizing (×10) costs the most (0.114), and shifting down slightly helps (0.127) — so the baseline is robust, not optimal, and λ placement goes on the monitoring list.

## B. Regime stress — PASS

Epoch-by-epoch DM tests (7 epochs × 2 horizons): COMBO ≥ HAR in **12 of 14 cells**; the two exceptions are small (t = −0.16 in 2022+, h=5; t = −0.74 in COVID, h=21). The edge is broad and modest rather than episodic — the healthiest possible profile. COMBO adds most exactly where HAR breaks (dot-com h=21: R² −0.008 → +0.040; GFC losses vs RW roughly halved). Humility note: in the GFC both models lose to the random walk at these horizons; nothing in this program forecasts structural breaks.

## C. Loss-function stress — PASS WITH CAVEAT

Under QLIKE (the vol-desk standard, punishing variance under-forecasts): COMBO still beats HAR, t = **+1.96 at h=5** but only **+0.96 at h=21**. The sign survives the loss change everywhere; long-horizon significance softens. Any production sizing that is asymmetrically hurt by under-forecast vol should weight the h=5–10 conclusions more heavily. QLIKE-targeted readouts are a known possible upgrade.

## D. Fresh-data stress — PASS (strongest evidence in the program)

The frozen pipeline (features, delays, dim, scale, seeds, λ grid, blend rule — zero retuning) was applied to a pooled panel of **505 single stocks never used in any development iteration**: one shared random projection (the "shared reservoir" of Memo #1) with a pooled cross-sectional readout, forecasting each stock's forward 21-day realized vol. On 363,619 OOS forecasts: R² vs RW = 0.343 (RC) vs 0.312 (HAR); **COMBO vs HAR DM t = +3.19** under conservative inference (daily cross-sectional clustering + HAC). This addresses the deepest worry — that three projects of iteration on one VIX series had leaked OOS information into design choices.

## Answer to "should we focus on RP-NGRC + λ-averaging?"

Yes — as **COMBO**, not standalone. The blend with HAR is not decoration: it stabilized inference on fresh data (t +1.68 alone → +3.19 blended), it rescued the worst λ perturbation, and it gives production a familiar fallback. Focus is justified because the edge (a) sits on a design plateau, (b) is spread across regimes, (c) survives the desk-standard loss at the horizons that matter most, and (d) replicated at t≈3 on untouched data.

## Residual risks (monitoring list)

λ-grid placement (only sensitive knob — consider regret-weighted λ averaging); h=21 QLIKE softness; structural-break blindness (both models); equity-only evidence so far (rates/FX vol replication pending); no transaction-cost monetization study yet; stock panel covers 2013–18 only.

## Next step

Monetization: map the h=5–21 edge into VIX calendar-spread and single-name variance signals with realistic futures/option costs — the study that converts t-stats into capacity estimates.

*Artifacts: `stress_a.py`, `stress_d.py`, `stress_report.png`.*
