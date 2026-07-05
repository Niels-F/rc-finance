"""Echo State Network toolkit for market prediction research."""
import numpy as np
import pandas as pd


class ESN:
    """Leaky Echo State Network with ridge readout. Only the readout is trained."""

    def __init__(self, n_inputs, n_res=300, spectral_radius=0.9, sparsity=0.9,
                 input_scale=0.5, leak=0.3, ridge=1e-2, seed=0, washout=24):
        rng = np.random.default_rng(seed)
        self.n_res, self.leak, self.ridge, self.washout = n_res, leak, ridge, washout
        # Fixed random input weights
        self.W_in = rng.uniform(-input_scale, input_scale, (n_res, n_inputs + 1))
        # Fixed sparse recurrent weights, rescaled to target spectral radius
        W = rng.uniform(-1, 1, (n_res, n_res))
        W[rng.random((n_res, n_res)) < sparsity] = 0.0
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / eig) if eig > 0 else W
        self.W_out = None

    def _run(self, U):
        """Drive reservoir with input matrix U (T x n_inputs); return states (T x n_res)."""
        T = U.shape[0]
        X = np.zeros((T, self.n_res))
        x = np.zeros(self.n_res)
        for t in range(T):
            u = np.concatenate(([1.0], U[t]))
            pre = self.W_in @ u + self.W @ x
            x = (1 - self.leak) * x + self.leak * np.tanh(pre)
            X[t] = x
        return X

    def fit(self, U, y):
        X = self._run(U)
        # Extended state: bias + reservoir + raw inputs (direct linear channel)
        Z = np.hstack([np.ones((len(U), 1)), X, U])
        Zw, yw = Z[self.washout:], y[self.washout:]
        A = Zw.T @ Zw + self.ridge * np.eye(Z.shape[1])
        self.W_out = np.linalg.solve(A, Zw.T @ yw)
        return self

    def predict(self, U):
        X = self._run(U)
        Z = np.hstack([np.ones((len(U), 1)), X, U])
        return Z @ self.W_out


def load_sp500(path="sp500_monthly.csv"):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    px = df["SP500"].astype(float)
    r = np.log(px / px.shift(1)).dropna()
    return px, r


def build_features(r):
    """Price-derived features only; every feature at index t uses info up to t."""
    f = pd.DataFrame(index=r.index)
    vol = r.rolling(12).std()
    f["ret_1m"] = r
    f["mom_3m"] = r.rolling(3).sum()
    f["mom_12m"] = r.rolling(12).sum()
    f["vol_12m"] = vol
    f["ret_norm"] = r / vol.shift(1)          # vol-scaled return
    # target: NEXT month return, vol-normalized (what the readout regresses on)
    tgt = (r / vol).shift(-1)
    data = pd.concat([f, tgt.rename("target")], axis=1).dropna()
    return data


def perf_stats(strat_r, freq=12, label=""):
    strat_r = pd.Series(strat_r).dropna()
    mu, sd = strat_r.mean() * freq, strat_r.std() * np.sqrt(freq)
    eq = strat_r.cumsum()                      # log-return equity curve
    dd = (eq - eq.cummax()).min()
    cagr = np.expm1(mu)  # from mean log return
    hit = (np.sign(strat_r[strat_r != 0]) > 0).mean() if (strat_r != 0).any() else np.nan
    return {"label": label, "CAGR%": 100 * cagr, "Vol%": 100 * sd,
            "Sharpe": mu / sd if sd > 0 else np.nan,
            "MaxDD(log)%": 100 * dd, "Hit%": 100 * hit, "N": len(strat_r)}
