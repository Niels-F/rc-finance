"""Stress battery D: fresh-data replication. Pooled panel of 505 stocks,
forecast change in log fwd-21d realized vol. FROZEN pipeline from Project 3:
delays (0,1,5,21), dim 400, tanh scale 1.5, 8 seeds, lambda grid (1e3,1e4,1e5),
inverse-regret blend win 252. Shared projection = shared reservoir; pooled readout.
Benchmarks: RW (current log rv21), pooled HAR-RV."""
import numpy as np, pandas as pd

df = pd.read_csv("all_stocks_5yr.csv", parse_dates=["date"])
px = df.pivot_table(index="date", columns="Name", values="close")
r = np.log(px / px.shift(1))
T, S = r.shape
ANN = np.sqrt(252)

# wide per-stock features (T x S each)
lrv5  = np.log(r.rolling(5).std() * ANN)
lrv21 = np.log(r.rolling(21).std() * ANN)
lrv63 = np.log(r.rolling(63).std() * ANN)
ret5  = r.rolling(5).sum()
vix = pd.read_csv("vix.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
lvix = np.log(vix).reindex(r.index).ffill()
LVIX = pd.DataFrame(np.tile(lvix.values[:, None], (1, S)), index=r.index, columns=r.columns)
DVIX = pd.DataFrame(np.tile(lvix.diff(5).values[:, None], (1, S)), index=r.index, columns=r.columns)
FW = [lrv5, lrv21, lrv63, ret5, LVIX, DVIX]          # 6 features
Y  = np.log(r.rolling(21).std().shift(-21) * ANN)     # fwd realized vol (level, log)

H = 21
DELAYS = (0, 1, 5, 21)
# build delayed wide stacks -> long
def shift_wide(W, d):
    V = W.shift(d)
    return V

feat_wide = []
for d in DELAYS:
    for W in FW:
        feat_wide.append(shift_wide(W, d).values)      # each T x S
X = np.stack(feat_wide, axis=2)                        # T x S x 24
ylev = Y.values                                        # T x S
base = lrv21.values                                    # RW pred & change base

valid = ~np.isnan(X).any(axis=2) & ~np.isnan(ylev) & ~np.isnan(base)
tidx = np.repeat(np.arange(T)[:, None], S, axis=1)
rows = valid
t_long = tidx[rows]                                    # time index per row
Xl = X[rows]                                           # N x 24
yl = ylev[rows]; bl = base[rows]
ych = yl - bl                                          # change target
order = np.argsort(t_long, kind="stable")
t_long, Xl, yl, bl, ych = t_long[order], Xl[order], yl[order], bl[order], ych[order]
N = len(yl); print(f"panel rows: {N} | dates: {T} | stocks: {S}")

T0 = int(T * 0.4)                                      # OOS start (date-based)
train0 = t_long < T0
mu, sd = Xl[train0].mean(0), Xl[train0].std(0)
Xs = np.clip((Xl - mu) / sd, -3, 3).astype(np.float32)

# HAR pooled features (levels)
Hf = np.column_stack([np.ones(N), Xl[:, 0], Xl[:, 1], Xl[:, 2]])  # lrv5,21,63 at d=0

RETRAIN = 63
LAMS = (1e3, 1e4, 1e5)
DIM, SEEDS, SCALE = 400, 8, 1.5
cut_dates = list(range(T0, T, RETRAIN))

# row ranges per retrain (rows sorted by t)
starts = np.searchsorted(t_long, cut_dates)
starts.sort()
P_rc = np.full(N, np.nan); P_har = np.full(N, np.nan)

# HAR walk-forward
for ci, cd in enumerate(cut_dates):
    lo = starts[ci]; hi = starts[ci + 1] if ci + 1 < len(cut_dates) else N
    fit = t_long < (cd - H)
    w = np.linalg.lstsq(Hf[fit], yl[fit], rcond=None)[0]
    P_har[lo:hi] = Hf[lo:hi] @ w

# RC: seeds sequential, block-wise projection, incremental normal equations
m = 1 + DIM + Xs.shape[1]
acc = np.zeros(N)
for s in range(SEEDS):
    rng = np.random.default_rng(50 + s)
    R = rng.normal(0, 1 / np.sqrt(Xs.shape[1]), (Xs.shape[1], DIM)).astype(np.float32)
    b = rng.uniform(-.5, .5, DIM).astype(np.float32)
    def zblock(a, c):
        Xa = Xs[a:c]
        return np.hstack([np.ones((c - a, 1), np.float32),
                          np.tanh(Xa @ R * SCALE + b), Xa]).astype(np.float64)
    G = np.zeros((m, m)); cvec = np.zeros(m); done = 0
    for ci, cd in enumerate(cut_dates):
        lo = starts[ci]; hi = starts[ci + 1] if ci + 1 < len(cut_dates) else N
        fit_end = np.searchsorted(t_long, cd - H)
        for a in range(done, fit_end, 100000):
            c = min(a + 100000, fit_end)
            Z = zblock(a, c); G += Z.T @ Z; cvec += Z.T @ ych[a:c]
        done = fit_end
        Zo = zblock(lo, hi)
        for lam in LAMS:
            acc[lo:hi] += Zo @ np.linalg.solve(G + lam * np.eye(m), cvec)
    print(f"seed {s} done", flush=True)
P_rc = bl + acc / (SEEDS * len(LAMS))

# inverse-regret blend using daily cross-sectional mean losses (known with H-day lag)
oos_rows = ~np.isnan(P_rc)
d_err_rc = pd.Series((P_rc - yl) ** 2).groupby(t_long).mean()
d_err_ha = pd.Series((P_har - yl) ** 2).groupby(t_long).mean()
m_rc = d_err_rc.rolling(252, min_periods=63).mean().shift(H)
m_ha = d_err_ha.rolling(252, min_periods=63).mean().shift(H)
w_t = ((1 / m_rc) / ((1 / m_rc) + (1 / m_ha))).fillna(0.5)
w_row = w_t.reindex(t_long).values
P_c = w_row * P_rc + (1 - w_row) * P_har

msk = oos_rows & ~np.isnan(P_har)
e_rw = (bl[msk] - yl[msk]) ** 2
e_ha = (P_har[msk] - yl[msk]) ** 2
e_rc = (P_rc[msk] - yl[msk]) ** 2
e_c  = (P_c[msk]  - yl[msk]) ** 2
r2 = lambda e: 1 - e.mean() / e_rw.mean()
print(f"\n=== FRESH DATA: pooled 505-stock fwd-21d vol, OOS rows {msk.sum()} ===")
print(f"R2 vs RW:  HAR {r2(e_ha):.4f} | RC {r2(e_rc):.4f} | COMBO {r2(e_c):.4f}")

# DM on daily cross-sectional mean loss diff (handles cross-corr), HAC for overlap
for nm, e in (("RC", e_rc), ("COMBO", e_c)):
    dd = pd.Series(e_ha - e, index=t_long[msk]).groupby(level=0).mean()
    v = dd.var(); L = 2 * H
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * dd.autocorr(l) * dd.var()
    print(f"DM {nm} vs HAR (daily-clustered, HAC): t = {dd.mean()/np.sqrt(v/len(dd)):+.2f}  "
          f"({len(dd)} days)")
