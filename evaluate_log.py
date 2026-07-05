"""evaluate_log.py — score the live track record.
Matches each logged forecast with the realized log-VIX h trading days after
data_asof; reports per-horizon RMSE, R^2 vs RW, hit rate of the directional
view, and DM t-stat of COMBO vs HAR (HAC for overlapping horizons)."""
import numpy as np, pandas as pd, sys


def matured_errors(log_path, vix, h=5):
    log = pd.read_csv(log_path, parse_dates=["data_asof"])
    lv = np.log(vix)
    dates = lv.index
    rows = []
    for _, r in log.iterrows():
        pos = dates.searchsorted(r["data_asof"])
        if pos >= len(dates) or dates[pos] != r["data_asof"]:
            continue
        tgt = pos + h
        if tgt >= len(dates):
            continue                                    # not matured yet
        y = lv.iloc[tgt]
        rows.append({"data_asof": r["data_asof"], "realized": y,
                     "se_RW": (r[f"h{h}_RW"] - y) ** 2,
                     "se_HAR": (r[f"h{h}_HAR"] - y) ** 2,
                     "se_RC": (r[f"h{h}_RC"] - y) ** 2,
                     "se_COMBO": (r[f"h{h}_COMBO"] - y) ** 2,
                     "dir_ok": int((r[f"h{h}_COMBO"] > r[f"h{h}_RW"]) == (y > r[f"h{h}_RW"]))})
    return pd.DataFrame(rows) if rows else None


def main():
    vix = pd.read_csv("vix_latest.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
    for h in (5, 10, 21):
        e = matured_errors("signal_log.csv", vix, h)
        if e is None or len(e) == 0:
            print(f"h={h:2d}: no matured forecasts yet"); continue
        r2 = lambda c: 1 - e[c].mean() / e["se_RW"].mean()
        d = pd.Series(e["se_HAR"].values - e["se_COMBO"].values)
        v = d.var()
        L = max(1, int(2 * h / 5))                      # weekly cadence assumed
        for l in range(1, min(L, len(d) - 1) + 1):
            ac = d.autocorr(l)
            if not np.isnan(ac): v += 2 * (1 - l / (L + 1)) * ac * d.var()
        t = d.mean() / np.sqrt(v / len(d)) if v > 0 and len(d) > 2 else np.nan
        print(f"h={h:2d} | n={len(e):3d} matured | R2 vs RW: HAR {r2('se_HAR'):+.3f} "
              f"RC {r2('se_RC'):+.3f} COMBO {r2('se_COMBO'):+.3f} | "
              f"dir hit {100 * e['dir_ok'].mean():.0f}% | DM(COMBO vs HAR) t={t:+.2f}")


if __name__ == "__main__":
    main()
