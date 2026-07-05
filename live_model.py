"""live_model.py — FROZEN pipeline for the forward paper-trading test.
Any change to CONFIG changes the hash and invalidates the track record.
Frozen on 2026-07-05 per Research Memo #4 (stress-tested configuration)."""
import numpy as np, pandas as pd, hashlib, json

CONFIG = {
    "model": "RP-NGRC lambda-averaged + HAR (COMBO)",
    "features": ["d1", "d5", "dev5", "dev21", "dev63", "slope", "lvl"],
    "delays": [0, 1, 5, 21],
    "proj_dim": 400,
    "tanh_scale": 1.5,
    "n_seeds": 8,
    "seed_base": 50,
    "lambda_grid": [1e3, 1e4, 1e5],
    "horizons": [5, 10, 21],
    "target": "change in log VIX, added back to current level",
    "standardization": "mean/sd of pre-1998 window, clipped at +-3 (frozen)",
    "blend": "inverse-regret from matured live log entries; w=0.5 until >=20 matured",
    "washout": 100,
    "frozen_date": "2026-07-05",
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def build_features(vix_close: pd.Series) -> pd.DataFrame:
    lv = np.log(vix_close)
    f = pd.DataFrame(index=lv.index)
    f["d1"] = lv.diff()
    f["d5"] = lv.diff(5) / 5
    f["dev5"] = lv - lv.rolling(5).mean()
    f["dev21"] = lv - lv.rolling(21).mean()
    f["dev63"] = lv - lv.rolling(63).mean()
    f["slope"] = lv.rolling(5).mean() - lv.rolling(63).mean()
    f["lvl"] = lv
    return f.dropna()


def forecasts(vix_close: pd.Series) -> dict:
    """Fit on all matured targets, forecast from the latest observation.
    Returns log-VIX level forecasts for each horizon, for RC / HAR / RW."""
    f = build_features(vix_close)
    lv = np.log(vix_close).reindex(f.index)
    n = len(f)
    i98 = f.index.searchsorted(pd.Timestamp("1998-01-01"))
    mu, sd = f.iloc[:i98].mean(), f.iloc[:i98].std()          # frozen normalization era
    U = np.clip(((f - mu) / sd).values, -3, 3)
    D = np.hstack([np.vstack([np.tile(U[0], (d, 1)), U[:n - d]]) for d in CONFIG["delays"]])
    lvv = lv.values
    har = np.column_stack([lvv, pd.Series(lvv).rolling(5).mean().bfill().values,
                           pd.Series(lvv).rolling(21).mean().bfill().values])
    WASH = CONFIG["washout"]
    out = {"asof": str(f.index[-1].date()), "vix": float(np.exp(lvv[-1]))}
    for h in CONFIG["horizons"]:
        ylev = lv.shift(-h).values
        ychg = ylev - lvv
        fit = slice(WASH, n - h)                              # only matured targets
        # HAR
        Zh = np.column_stack([np.ones(n - h - WASH), har[fit]])
        wh = np.linalg.lstsq(Zh, ylev[fit], rcond=None)[0]
        har_fc = float(np.concatenate(([1.0], har[-1])) @ wh)
        # RC (8 seeds x 3 lambdas averaged)
        acc = 0.0
        for s in range(CONFIG["n_seeds"]):
            rng = np.random.default_rng(CONFIG["seed_base"] + s)
            R = rng.normal(0, 1 / np.sqrt(D.shape[1]), (D.shape[1], CONFIG["proj_dim"]))
            b = rng.uniform(-0.5, 0.5, CONFIG["proj_dim"])
            Z = np.hstack([np.ones((n, 1)), np.tanh(D @ R * CONFIG["tanh_scale"] + b), D])
            G = Z[fit].T @ Z[fit]; c = Z[fit].T @ ychg[fit]
            for lam in CONFIG["lambda_grid"]:
                acc += float(Z[-1] @ np.linalg.solve(G + lam * np.eye(Z.shape[1]), c))
        rc_fc = float(lvv[-1] + acc / (CONFIG["n_seeds"] * len(CONFIG["lambda_grid"])))
        out[f"h{h}"] = {"RW": float(lvv[-1]), "HAR": har_fc, "RC": rc_fc}
    return out
