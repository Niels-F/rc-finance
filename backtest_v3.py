"""V3 diagnostics: (a) per-seed OOS IC distribution, (b) hyperparameter sweep.
Everything evaluated strictly walk-forward out-of-sample (1900+)."""
import numpy as np, pandas as pd
from esn_lib import ESN, load_sp500, build_features

px, r = load_sp500()
data = build_features(r)
FEATS = ["ret_1m", "mom_3m", "mom_12m", "vol_12m", "ret_norm"]
dates = data.index
start_oos = dates.searchsorted(pd.Timestamp("1900-01-01"))
RETRAIN = 24  # coarser retrain to make the sweep tractable


def walkforward_pred(leak, ridge, rho, seed):
    preds = np.full(len(data), np.nan)
    i = start_oos
    esn = ESN(n_inputs=len(FEATS), n_res=300, spectral_radius=rho,
              leak=leak, ridge=ridge, seed=seed)  # same random W reused, refit readout
    while i < len(data):
        tr = data.iloc[:i]
        mu, sd = tr[FEATS].mean(), tr[FEATS].std()
        esn.fit(((tr[FEATS] - mu) / sd).values, tr["target"].values)
        j = min(i + RETRAIN, len(data))
        U_all = ((data[FEATS].iloc[:j] - mu) / sd).values
        preds[i:j] = esn.predict(U_all)[i:j]
        i = j
    return preds


tgt = data["target"].values
mask = ~np.isnan(walkforward_pred(0.3, 1e-2, 0.9, 0))  # oos mask

# (a) seed variance at v1's config
print("=== (a) Per-seed OOS IC, config = v1 (leak .3, ridge .01, rho .9) ===")
ics = []
for s in range(12):
    p = walkforward_pred(0.3, 1e-2, 0.9, s)
    ic = np.corrcoef(p[mask], tgt[mask])[0, 1]
    ics.append(ic)
    print(f"seed {s:2d}: IC = {ic:+.4f}")
ics = np.array(ics)
print(f"mean {ics.mean():+.4f}  sd {ics.std():.4f}  min {ics.min():+.4f}  max {ics.max():+.4f}")

# (b) hyperparameter sweep, 3-seed mini-ensembles
print("\n=== (b) Sweep (3-seed ensemble IC) ===")
rows = []
for leak in (0.1, 0.3):
    for ridge in (1e-2, 1.0, 100.0):
        P = np.nanmean([walkforward_pred(leak, ridge, 0.9, 200 + s) for s in range(3)], axis=0)
        ic = np.corrcoef(P[mask], tgt[mask])[0, 1]
        rows.append({"leak": leak, "ridge": ridge, "ens_IC": ic})
        print(f"leak {leak}  ridge {ridge:>6}  IC {ic:+.4f}")
res = pd.DataFrame(rows)
res.to_csv("v3_sweep.csv", index=False)
np.save("v3_seed_ics.npy", ics)
