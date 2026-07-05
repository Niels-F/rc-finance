"""live_signal.py — run this (e.g. weekly) to generate and log a timestamped signal.
Protocol: APPEND-ONLY log. Never edit or delete past rows. Each row carries the
config hash and a chain hash of the previous row (tamper-evident)."""
import pandas as pd, numpy as np, hashlib, os, sys, datetime, urllib.request
from live_model import forecasts, CONFIG_HASH

LOG = "signal_log.csv"
URL = "https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv"

def main():
    with urllib.request.urlopen(URL, timeout=30) as r:
        raw = r.read().decode()
    open("vix_latest.csv", "w").write(raw)
    vix = pd.read_csv("vix_latest.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
    fc = forecasts(vix)

    prev_hash = "GENESIS"
    if os.path.exists(LOG):
        last = pd.read_csv(LOG).iloc[-1]
        if str(last["data_asof"]) == fc["asof"]:
            print(f"No new data since {fc['asof']}; nothing logged."); return
        prev_hash = str(last["row_hash"])

    # blend weight from matured log entries (inverse-regret); 0.5 until >=20 matured
    w = 0.5
    if os.path.exists(LOG):
        try:
            from evaluate_log import matured_errors
            e = matured_errors(LOG, vix)
            if e is not None and len(e) >= 20:
                m_rc, m_har = e["se_RC"].mean(), e["se_HAR"].mean()
                w = float((1 / m_rc) / (1 / m_rc + 1 / m_har))
        except Exception as ex:
            print("blend-weight update skipped:", ex)

    row = {"run_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
           "data_asof": fc["asof"], "vix": round(fc["vix"], 2),
           "config_hash": CONFIG_HASH, "w_RC": round(w, 4)}
    for h in (5, 10, 21):
        d = fc[f"h{h}"]
        combo = w * d["RC"] + (1 - w) * d["HAR"]
        row[f"h{h}_RW"] = round(d["RW"], 6)
        row[f"h{h}_HAR"] = round(d["HAR"], 6)
        row[f"h{h}_RC"] = round(d["RC"], 6)
        row[f"h{h}_COMBO"] = round(combo, 6)
        row[f"h{h}_view"] = "VOL UP" if combo > d["RW"] + 1e-9 else "VOL DOWN"
    payload = "|".join(f"{k}={row[k]}" for k in sorted(row)) + "|prev=" + prev_hash
    row["row_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:16]

    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    print(f"Signal logged | data through {fc['asof']} | VIX {fc['vix']:.2f} | "
          f"config {CONFIG_HASH} | w_RC {w:.2f}")
    for h in (5, 10, 21):
        print(f"  h={h:2d}d: COMBO {np.exp(row[f'h{h}_COMBO']):6.2f}  "
              f"(HAR {np.exp(row[f'h{h}_HAR']):6.2f} | RW {np.exp(row[f'h{h}_RW']):6.2f})  "
              f"-> {row[f'h{h}_view']}")

if __name__ == "__main__":
    main()
