# Reservoir Computing for Financial Time Series

A systematic research program testing whether reservoir computing (Echo State Networks and their modern
next-generation variants) can extract exploitable structure from financial time series — run like a
research desk would: walk-forward validation, honest benchmarks, negative results kept on the record,
and a stress-testing pass before anything gets called a result.

**Status:** four completed research memos, a "champion" model (COMBO) that survived design-perturbation,
regime, loss-function, and fresh-data stress tests, and an ongoing tamper-evident live forward-test
started 2026-07-05.

## TL;DR

- **Index returns:** no exploitable signal. An apparent monthly edge (Sharpe 0.67) turned out to be the
  model learning a data artifact from time-averaged prices — caught via an execution-lag stress test that
  flipped the sign. See [`RESEARCH_MEMO.md`](RESEARCH_MEMO.md).
- **Volatility forecasting:** a real, replicable edge. An ESN beats HAR out-of-sample over 28 years of VIX
  data, with the edge concentrated in stress regimes. See [`RESEARCH_MEMO_2.md`](RESEARCH_MEMO_2.md).
- **Architecture search:** recurrence turned out to be unnecessary — a feedforward random projection of a
  ~21-day delay embedding (RP-NGRC) matches the recurrent ESN at 1/10th the complexity. Averaging across
  ridge penalties instead of selecting one, and soft-blending with HAR, produced **COMBO**, the new
  champion (DM t ≈ +2 vs. HAR across horizons). See [`RESEARCH_MEMO_3.md`](RESEARCH_MEMO_3.md).
- **Stress testing:** COMBO passed design-perturbation, regime, and loss-function stress tests, and
  replicated on 505 stocks never touched during development (t = +3.19). See
  [`RESEARCH_MEMO_4_STRESS.md`](RESEARCH_MEMO_4_STRESS.md).
- **Live test:** COMBO is now frozen and forward-tested weekly with an append-only, chain-hashed signal
  log so the track record can't be quietly edited. See [`LIVE_TEST_PROTOCOL.md`](LIVE_TEST_PROTOCOL.md).

## Why this is structured as a research log, not a single script

Each memo documents a full iteration cycle, including the dead ends — because the dead ends are the
point. The project's actual finding isn't "COMBO works," it's that every earlier "edge" (seed luck, a
vol-targeting artifact, a time-averaging artifact) looked statistically real until it was tested the right
way. The defensive machinery — ensembling, walk-forward retraining, execution-lag stress, fresh-data
validation — is what separates a publishable-looking backtest from a real one.

## Repository contents

| File | Role |
|---|---|
| `esn_lib.py` | Core reservoir computing library (ESN implementation) |
| `backtest_v1.py` – `backtest_v4.py` | Iteration history for the index-timing study (Memo #1) |
| `vix_horse_race.py` | Volatility forecasting benchmark suite: RW / AR(1) / HAR / ESN (Memo #2) |
| `p3_ngrc.py` | Next-generation RC variants: polynomial NG-RC, RP-NGRC, COMBO (Memo #3) |
| `stress_a.py`, `stress_d.py` | Design-perturbation and fresh-data (505-stock) stress batteries (Memo #4) |
| `live_model.py` | Frozen COMBO pipeline used for the live test — not to be edited |
| `live_signal.py` | Fetches latest data, logs timestamped forecasts to `signal_log.csv` |
| `evaluate_log.py` | Scores the live log against realized outcomes |
| `signal_log.csv` | The live forward-test track record (append-only, chain-hashed) |

## Method, briefly

Fixed random reservoir + ridge readout only (no recurrent weights are ever trained). Every study uses
walk-forward retraining, train-only standardization, and benchmarks a real desk would demand (HAR for
volatility, buy-and-hold / vol-targeted buy-and-hold for returns), with Diebold–Mariano significance tests
using HAC-adjusted variance for overlapping horizons.

## Disclaimer

This is a research validation exercise, not investment advice. No live capital is or should be allocated
based on these signals — a conclusion the research explicitly reaches and states in
[`LIVE_TEST_PROTOCOL.md`](LIVE_TEST_PROTOCOL.md).
