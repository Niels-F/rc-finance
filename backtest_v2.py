"""V2: ensemble of 10 ESNs, long/flat with dead-zone, volatility targeting, costs.
Fixes chosen strictly from v1 diagnostics."""
import numpy as np, pandas as pd
from esn_lib import ESN, load_sp500, build_features, perf_stats

px, r = load_sp500()
data = build_features(r)
FEATS = ["ret_1m", "mom_3m", "mom_12m", "vol_12m", "ret_norm"]
N_ENS, RETRAIN, COST = 10, 12, 0.0010
TARGET_VOL = 0.12 / np.sqrt(12)          # 12% annualized, monthly units
MAX_LEV = 1.5

dates = data.index
start_oos = dates.searchsorted(pd.Timestamp("1900-01-01"))

preds = np.full(len(data), np.nan)
i = start_oos
while i < len(data):
    tr = data.iloc[:i]
    mu, sd = tr[FEATS].mean(), tr[FEATS].std()
    U_tr = ((tr[FEATS] - mu) / sd).values
    y_tr = tr["target"].values
    j = min(i + RETRAIN, len(data))
    U_all = ((data[FEATS].iloc[:j] - mu) / sd).values
    ens = np.zeros(j - i)
    for k in range(N_ENS):                                # ensemble over seeds
        esn = ESN(n_inputs=len(FEATS), n_res=300, spectral_radius=0.9,
                  leak=0.3, ridge=1e-2, seed=100 + k).fit(U_tr, y_tr)
        ens += esn.predict(U_all)[i:j]
    preds[i:j] = ens / N_ENS
    i = j

data = data.assign(pred=preds).dropna(subset=["pred"])
next_ret = r.shift(-1).reindex(data.index)

# --- position construction -------------------------------------------------
# 1) dead-zone long/flat: stay long unless prediction is confidently negative,
#    threshold scaled by the expanding std of PAST predictions (no lookahead).
pred = data["pred"]
run_sd = pred.expanding(24).std().shift(1)
gate = (pred > -0.5 * run_sd).astype(float)              # 1 = long, 0 = flat
gate[run_sd.isna()] = 1.0                                 # warmup: default long

# 2) volatility targeting using trailing 12m realized vol (known at t)
w_vol = (TARGET_VOL / data["vol_12m"]).clip(upper=MAX_LEV)

pos = gate * w_vol
turnover = pos.diff().abs().fillna(0)
strat = pos * next_ret - COST * turnover

# fair benchmarks
bh = next_ret
bh_vt = w_vol * next_ret - COST * w_vol.diff().abs().fillna(0)   # vol-targeted B&H

res = pd.DataFrame([
    perf_stats(strat, label="ESN v2 long/flat+VT (net)"),
    perf_stats(bh_vt, label="Vol-target B&H (net)"),
    perf_stats(bh, label="Buy & Hold"),
])
print(res.round(2).to_string(index=False))

ic = np.corrcoef(data["pred"], data["target"].values)[0, 1]
print(f"\nOOS IC (ensemble): {ic:.4f} | flat months: {100*(gate==0).mean():.1f}%")
print(f"Avg turnover/mo: {turnover.mean():.2f}")

# value added by the ESN gate over pure vol targeting (the honest comparison)
d = strat - bh_vt
print(f"ESN gate vs vol-target B&H: ann. excess {12*d.mean()*100:.2f}%  "
      f"t-stat {d.mean()/d.std()*np.sqrt(len(d)):.2f}")

by_dec = strat.groupby((strat.index.year // 10) * 10).apply(
    lambda s: (s.mean()*12)/(s.std()*np.sqrt(12)) if s.std() > 0 else np.nan)
bh_dec = bh_vt.groupby((bh_vt.index.year // 10) * 10).apply(
    lambda s: (s.mean()*12)/(s.std()*np.sqrt(12)) if s.std() > 0 else np.nan)
print("\nNet Sharpe by decade (v2 vs vol-target B&H):")
print(pd.DataFrame({"ESN v2": by_dec, "VT B&H": bh_dec}).round(2).to_string())

pd.DataFrame({"pred": pred, "pos": pos, "gate": gate, "next_ret": next_ret,
              "strat": strat, "bh_vt": bh_vt}).to_csv("v2_results.csv")
