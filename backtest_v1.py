"""V1: single ESN, expanding walk-forward, sign(pred) long/short, 10 bps costs."""
import numpy as np, pandas as pd
from esn_lib import ESN, load_sp500, build_features, perf_stats

np.random.seed(0)
px, r = load_sp500()
data = build_features(r)
FEATS = ["ret_1m", "mom_3m", "mom_12m", "vol_12m", "ret_norm"]

# Walk-forward: first train window = everything before 1900; retrain every 12 months.
dates = data.index
start_oos = dates.searchsorted(pd.Timestamp("1900-01-01"))
RETRAIN = 12

preds = np.full(len(data), np.nan)
i = start_oos
while i < len(data):
    tr = data.iloc[:i]
    mu, sd = tr[FEATS].mean(), tr[FEATS].std()          # standardize on TRAIN only
    U_tr = ((tr[FEATS] - mu) / sd).values
    y_tr = tr["target"].values
    esn = ESN(n_inputs=len(FEATS), n_res=300, spectral_radius=0.9,
              leak=0.3, ridge=1e-2, seed=42).fit(U_tr, y_tr)
    j = min(i + RETRAIN, len(data))
    # Run the reservoir over full history up to each OOS point (state continuity)
    U_all = ((data[FEATS].iloc[:j] - mu) / sd).values
    p = esn.predict(U_all)
    preds[i:j] = p[i:j]
    i = j

data = data.assign(pred=preds).dropna(subset=["pred"])
next_ret = r.shift(-1).reindex(data.index)              # realized next-month return

pos = np.sign(data["pred"])                              # +1 / -1
turnover = pos.diff().abs().fillna(0)
COST = 0.0010                                            # 10 bps per unit turnover
strat = pos * next_ret - COST * turnover

res = pd.DataFrame([
    perf_stats(strat, label="ESN v1 long/short (net)"),
    perf_stats(pos * next_ret, label="ESN v1 (gross)"),
    perf_stats(next_ret, label="Buy & Hold"),
])
print(res.round(2).to_string(index=False))

ic = np.corrcoef(data["pred"], data["target"].values)[0, 1]
print(f"\nInformation coefficient (pred vs realized, OOS): {ic:.4f}")
print(f"Avg monthly turnover: {turnover.mean():.2f} | trades/yr ~ {6*turnover.mean():.1f}")
print(f"% months long: {100*(pos>0).mean():.1f}")

# decade breakdown of net Sharpe
by_dec = strat.groupby((strat.index.year // 10) * 10).apply(
    lambda s: (s.mean()*12)/(s.std()*np.sqrt(12)) if s.std() > 0 else np.nan)
print("\nNet Sharpe by decade:")
print(by_dec.round(2).to_string())

pd.DataFrame({"pred": data["pred"], "pos": pos, "next_ret": next_ret,
              "strat": strat}).to_csv("v1_results.csv")
