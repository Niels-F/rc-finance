"""V4: leak .3 / ridge 100 / rho .9, 10-seed ensemble, retrain 12m.
Decompose the IC (normalized vs raw target) and re-run the trading attribution."""
import numpy as np, pandas as pd
from esn_lib import ESN, load_sp500, build_features, perf_stats

px, r = load_sp500()
data = build_features(r)
FEATS = ["ret_1m", "mom_3m", "mom_12m", "vol_12m", "ret_norm"]
N_ENS, RETRAIN, COST = 10, 12, 0.0010
TARGET_VOL, MAX_LEV = 0.12 / np.sqrt(12), 1.5
start_oos = data.index.searchsorted(pd.Timestamp("1900-01-01"))

preds = np.full(len(data), np.nan)
i = start_oos
while i < len(data):
    tr = data.iloc[:i]
    mu, sd = tr[FEATS].mean(), tr[FEATS].std()
    U_tr, y_tr = ((tr[FEATS] - mu) / sd).values, tr["target"].values
    j = min(i + RETRAIN, len(data))
    U_all = ((data[FEATS].iloc[:j] - mu) / sd).values
    ens = np.zeros(j - i)
    for k in range(N_ENS):
        esn = ESN(len(FEATS), n_res=300, spectral_radius=0.9, leak=0.3,
                  ridge=100.0, seed=300 + k).fit(U_tr, y_tr)
        ens += esn.predict(U_all)[i:j]
    preds[i:j] = ens / N_ENS
    i = j

data = data.assign(pred=preds).dropna(subset=["pred"])
next_ret = r.shift(-1).reindex(data.index)
pred = data["pred"]

# --- IC decomposition -------------------------------------------------------
ic_norm = np.corrcoef(pred, data["target"])[0, 1]
ic_raw = np.corrcoef(pred, next_ret)[0, 1]
ic_sign = np.corrcoef(pred, np.sign(next_ret))[0, 1]
ic_vol = np.corrcoef(pred, 1.0 / data["vol_12m"])[0, 1]
print(f"IC vs vol-normalized target : {ic_norm:+.4f}")
print(f"IC vs RAW next-month return : {ic_raw:+.4f}")
print(f"IC vs sign(next return)     : {ic_sign:+.4f}")
print(f"corr(pred, 1/current vol)   : {ic_vol:+.4f}   <- denominator effect")

# residual skill: orthogonalize pred against 1/vol, re-measure vs raw returns
x = (1.0 / data["vol_12m"]).values
beta = np.polyfit(x, pred.values, 1)
resid = pred.values - np.polyval(beta, x)
ic_resid = np.corrcoef(resid, next_ret)[0, 1]
print(f"IC of vol-orthogonalized pred vs raw return: {ic_resid:+.4f}")

# --- trading -----------------------------------------------------------------
run_sd = pred.expanding(24).std().shift(1)
gate = (pred > -0.5 * run_sd).astype(float); gate[run_sd.isna()] = 1.0
w_vol = (TARGET_VOL / data["vol_12m"]).clip(upper=MAX_LEV)
pos = gate * w_vol
turnover = pos.diff().abs().fillna(0)
strat = pos * next_ret - COST * turnover
bh_vt = w_vol * next_ret - COST * w_vol.diff().abs().fillna(0)

res = pd.DataFrame([
    perf_stats(strat, label="ESN v4 long/flat+VT (net)"),
    perf_stats(bh_vt, label="Vol-target B&H (net)"),
    perf_stats(next_ret, label="Buy & Hold"),
])
print("\n" + res.round(2).to_string(index=False))
d = strat - bh_vt
print(f"\nESN gate vs VT B&H: ann. excess {12*d.mean()*100:+.2f}%  "
      f"t-stat {d.mean()/d.std()*np.sqrt(len(d)):+.2f} | flat months {100*(gate==0).mean():.1f}%")

pd.DataFrame({"pred": pred, "pos": pos, "next_ret": next_ret,
              "strat": strat, "bh_vt": bh_vt}).to_csv("v4_results.csv")
