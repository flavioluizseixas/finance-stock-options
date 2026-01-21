import numpy as np
import pandas as pd

def liq_raw(trades, volume):
    t = float(trades) if np.isfinite(trades) else 0.0
    v = float(volume) if np.isfinite(volume) else 0.0
    return t + np.log1p(max(v, 0.0))

def liq_leg_from_row(row):
    t = pd.to_numeric(row.get("trades", 0), errors="coerce")
    v = pd.to_numeric(row.get("volume", 0), errors="coerce")
    t = float(t) if pd.notna(t) else 0.0
    v = float(v) if pd.notna(v) else 0.0
    return liq_raw(t, v), t, v

def liq_class_single(liq, trades, volume, cfg):
    if trades < cfg["liq_min_trades"] or volume < cfg["liq_min_volume"]:
        return "RUIM"
    if liq >= cfg["liq_min_ok"]:
        return "OK"
    if liq >= cfg["liq_min_alert"]:
        return "ALERTA"
    return "RUIM"

def liq_pair_metrics(liq1, liq2):
    mn = min(liq1, liq2)
    mx = max(liq1, liq2)
    ratio = (mn / mx) if mx > 0 else 0.0
    return mn, mx, ratio

def liq_class_pair(min_liq, ratio, cfg):
    if (min_liq >= cfg["liq_pair_min_ok"]) and (ratio >= cfg["liq_pair_ratio_ok"]):
        return "OK"
    if (min_liq >= cfg["liq_pair_min_alert"]) and (ratio >= cfg["liq_pair_ratio_alert"]):
        return "ALERTA"
    return "RUIM"

def liq_penalty(liq_class, cfg):
    if liq_class == "OK":
        return 0.0
    if liq_class == "ALERTA":
        return float(cfg["liq_penalty_alert"])
    return float(cfg["liq_penalty_bad"])

def liquidity_score_df(d: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(d.get("trades", 0), errors="coerce").fillna(0)
    v = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)
    return t + np.log1p(v)
