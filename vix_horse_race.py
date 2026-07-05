"""Project 2a: forecast log-VIX at h=1 and h=5 days, walk-forward 1998-2026.
Benchmarks: random walk, AR(1), HAR-style cascade. ESN: 8-seed ensemble,
states computed once (fixed standardization), ridge readout refit annually."""
import numpy as np, pandas as pd
from esn_lib import ESN

vix = pd.read_csv("vix.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
lv = np.log(vix)

f = pd.DataFrame(index=lv.index)
f["lv"] = lv
f["d1"] = lv.diff()
f["m5"] = lv.rolling(5).mean()
f["m21"] = lv.rolling(21).mean()
f["m63"] = lv.rolling(63).mean()
f["dev21"] = lv - f["m21"]
f = f.dropna()
FEATS = list(f.columns)

START = f.index.searchsorted(pd.Timestamp("1998-01-01"))
RETRAIN = 252
N_ENS, N_RES, WASHOUT = 8, 400, 100

# fixed standardization from initial train window only (causal, no lookahead)
mu, sd = f.iloc[:START].mean(), f.iloc[:START].std()
U = ((f - mu) / sd).values

# reservoir states computed once per seed (states at t depend only on inputs <= t)
print("running reservoirs...")
states = []
for k in range(N_ENS):
    esn = ESN(len(FEATS), n_res=N_RES, spectral_radius=0.9, leak=0.3, seed=700 + k)
    X = esn._run(U)
    states.append(np.hstack([np.ones((len(U), 1)), X, U]))

def ridge_fit(Z, y, lam):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y)

def ols_fit(Z, y):
    return np.linalg.lstsq(Z, y, rcond=None)[0]

results = {}
for h in (1, 5):
    y = pd.Series(np.log(vix)).reindex(f.index)  # ensure alignment
    y = lv.reindex(f.index).shift(-h).values      # target: log VIX at t+h
    n = len(f)
    preds = {m: np.full(n, np.nan) for m in ("RW", "AR1", "HAR", "ESN")}

    # choose ESN ridge once on inner split of the initial train window
    tr_end = START
    inner = int(tr_end * 0.75)
    best_lam, best_mse = None, np.inf
    for lam in (1.0, 10.0, 100.0, 1000.0):
        mses = []
        for k in range(3):
            Z = states[k]
            w = ridge_fit(Z[WASHOUT:inner - h], y[WASHOUT:inner - h], lam)
            p = Z[inner:tr_end - h] @ w
            mses.append(np.mean((p - y[inner:tr_end - h]) ** 2))
        if np.mean(mses) < best_mse:
            best_mse, best_lam = np.mean(mses), lam
    print(f"h={h}: selected ridge lambda = {best_lam}")

    har_cols = ["lv", "m5", "m21"]
    Uh = f[har_cols].values
    i = START
    while i < n:
        j = min(i + RETRAIN, n)
        fit_end = i - h                             # only rows with known targets
        # random walk
        preds["RW"][i:j] = f["lv"].values[i:j]
        # AR(1)
        Zar = np.column_stack([np.ones(fit_end - WASHOUT), f["lv"].values[WASHOUT:fit_end]])
        w = ols_fit(Zar, y[WASHOUT:fit_end])
        preds["AR1"][i:j] = np.column_stack([np.ones(j - i), f["lv"].values[i:j]]) @ w
        # HAR-style
        Zh = np.column_stack([np.ones(fit_end - WASHOUT), Uh[WASHOUT:fit_end]])
        w = ols_fit(Zh, y[WASHOUT:fit_end])
        preds["HAR"][i:j] = np.column_stack([np.ones(j - i), Uh[i:j]]) @ w
        # ESN ensemble
        acc = np.zeros(j - i)
        for k in range(N_ENS):
            Z = states[k]
            w = ridge_fit(Z[WASHOUT:fit_end], y[WASHOUT:fit_end], best_lam)
            acc += Z[i:j] @ w
        preds["ESN"][i:j] = acc / N_ENS
        i = j

    oos = slice(START, n - h)
    yt = y[oos]
    mse_rw = np.mean((preds["RW"][oos] - yt) ** 2)
    rows = []
    hi = f["lv"].values[oos] > np.log(25)           # regime split: VIX > 25
    for m in ("RW", "AR1", "HAR", "ESN"):
        e = preds[m][oos] - yt
        r2 = 1 - np.mean(e ** 2) / mse_rw
        r2_hi = 1 - np.mean(e[hi] ** 2) / np.mean((preds["RW"][oos][hi] - yt[hi]) ** 2)
        r2_lo = 1 - np.mean(e[~hi] ** 2) / np.mean((preds["RW"][oos][~hi] - yt[~hi]) ** 2)
        rows.append({"model": m, "RMSE": np.sqrt(np.mean(e ** 2)),
                     "R2_vs_RW": r2, "R2_hiVol": r2_hi, "R2_loVol": r2_lo})
    res = pd.DataFrame(rows)
    print(f"\n=== horizon h={h} days, OOS 1998-2026 (n={len(yt)}) ===")
    print(res.round(4).to_string(index=False))
    results[h] = (res, preds, y)

# save ESN vs HAR predictions for the figure
np.savez("vix_preds.npz",
         idx=f.index.values, lv=f["lv"].values,
         **{f"h{h}_{m}": results[h][1][m] for h in (1, 5) for m in ("RW", "AR1", "HAR", "ESN")},
         **{f"h{h}_y": results[h][2] for h in (1, 5)})
print("\nsaved vix_preds.npz")
