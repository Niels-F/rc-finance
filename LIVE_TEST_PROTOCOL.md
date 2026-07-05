# Live Forward Test — Protocol (started 2026-07-05)

**Object under test:** COMBO (λ-averaged RP-NGRC + HAR blend), frozen per Memo #4.
**Config hash:** `029848b7fda9da99` — if this hash ever changes, the track record restarts from zero.

## Why forward-only

Every historical data point in this program has been touched by development: walk-forward
discipline protected each *prediction*, but the researcher saw results and iterated three
times. The only contamination-free evidence is forecasts logged before their outcomes exist.
Signal #1 (2026-07-05, data through 2026-07-03) is the first such record.

## The kit

- `live_model.py` — frozen pipeline + config hash. Do not edit.
- `live_signal.py` — fetches latest VIX data, logs timestamped forecasts (RW/HAR/RC/COMBO,
  h = 5/10/21) to `signal_log.csv`. Append-only; each row chain-hashes the previous row,
  so any tampering breaks the chain.
- `evaluate_log.py` — matures forecasts against realized VIX; reports R² vs RW, directional
  hit rate, and DM t-stat of COMBO vs HAR.
- `signal_log.csv` — the track record. This file IS the asset. Back it up.

## Operating rules

1. Run `live_signal.py` **once per week**, same weekday (weekly cadence keeps h=5 forecasts
   non-overlapping; h=21 remains overlapping and is scored with HAC adjustment).
2. Never edit or delete log rows. Missed weeks are fine; backfilling is not.
3. Never retune. Any improvement idea goes into a *new* config with a *new* hash and its own
   log, run in parallel — the old track record continues untouched.
4. Run `evaluate_log.py` whenever curious; judge nothing before the milestones below.

## What "holding its line" means (pre-registered, so we can't move the goalposts)

| Milestone | Expectation if the edge is real | Red flag |
|---|---|---|
| 13 weeks (~13 matured h=5) | Noise; directional hit rate anywhere 30–80%. No verdict. | COMBO RMSE > RW by wide margin persistently |
| 26 weeks | COMBO R² vs RW > 0 at h=5; hit rate > 50% | R² vs RW clearly < 0 at all horizons |
| 52 weeks | h=5: R² vs RW ≈ +0.03…+0.10 and COMBO ≥ HAR; DM t vs HAR ≈ +0.5…+1.5 (a single year cannot deliver t≈2 — that took 28 years in-sample) | COMBO < HAR *and* < RW at every horizon |

Power reality-check: the in-sample edge is R² ≈ 0.06–0.13. At weekly cadence that implies
roughly 1–2 years to distinguish COMBO from HAR with confidence, and ~6–12 months just to
distinguish it from a random walk. Patience is part of the test design.

## Known gaps this test does NOT close

- **Market-implied benchmark:** beating RW/HAR is not beating the VIX futures curve. When
  term-structure/futures data becomes available, log its implied forecast alongside (the
  schema has room; add columns, don't change existing ones).
- No tradability claim: no futures roll, basis, costs, or margin modeled. This test scores
  *forecasts*, not P&L.
- Single series: one VIX signal per week accrues evidence slowly; the 505-stock panel
  (Memo #4, t=+3.19) remains the strongest breadth evidence on record.

## Standing reminder

This is a research validation exercise. Nothing in it is investment advice, and no live
capital should reference these signals — a stance the PI has already, wisely, taken.
