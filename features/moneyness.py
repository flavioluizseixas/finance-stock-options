import numpy as np
import pandas as pd

def _classify_one(df: pd.DataFrame, spot: float, atm_mode: str, atm_pct: float):
    d = df.copy()
    d["strike"] = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
    if spot is None or not np.isfinite(spot) or spot <= 0:
        d["moneyness"] = "UNKNOWN"
        d["moneyness_dist"] = np.nan
        return d

    d["moneyness_dist"] = (d["strike"] - spot).abs() / spot

    def base_itm_otm(row):
        t = str(row.get("option_type", "")).upper()
        K = float(row["strike"])
        if t == "CALL":
            return "ITM" if K < spot else "OTM"
        return "ITM" if K > spot else "OTM"

    d["m_base"] = d.apply(base_itm_otm, axis=1)

    if atm_mode == "pct":
        d["moneyness"] = np.where(d["moneyness_dist"] <= atm_pct, "ATM", d["m_base"])
        return d.drop(columns=["m_base"])

    d["moneyness"] = d["m_base"]
    idx = (
        d.groupby(["expiry_date", "option_type"])["moneyness_dist"]
        .idxmin().dropna().astype(int).tolist()
    )
    d.loc[idx, "moneyness"] = "ATM"
    return d.drop(columns=["m_base"])

def classify_moneyness_multi(df: pd.DataFrame, atm_mode: str = "pct", atm_pct: float = 0.01):
    if df is None or df.empty:
        return df
    out = []
    key = "ticker" if "ticker" in df.columns else "asset_id"
    for _, g in df.groupby(key):
        spot = None
        if "spot_ref" in g.columns:
            s = pd.to_numeric(g["spot_ref"], errors="coerce").dropna()
            spot = float(s.iloc[0]) if len(s) else None
        if spot is None and "spot" in g.columns:
            s = pd.to_numeric(g["spot"], errors="coerce").dropna()
            spot = float(s.iloc[0]) if len(s) else None
        out.append(_classify_one(g, spot, atm_mode=atm_mode, atm_pct=float(atm_pct)))
    return pd.concat(out, ignore_index=True)

def apply_universe_filter(df: pd.DataFrame, cfg_univ: dict):
    if df is None or df.empty:
        return df

    d = df.copy()
    d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
    d["delta"] = pd.to_numeric(d.get("delta", np.nan), errors="coerce")
    d["volume"] = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)
    d["trades"] = pd.to_numeric(d.get("trades", 0), errors="coerce").fillna(0)
    d["vol_fin"] = d["last_price"].fillna(0) * d["volume"].fillna(0)

    if float(cfg_univ.get("last_price_min", 0.0)) > 0:
        d = d[d["last_price"] >= float(cfg_univ["last_price_min"])].copy()
    if float(cfg_univ.get("delta_abs_min", 0.0)) > 0:
        d = d[d["delta"].abs().fillna(0) >= float(cfg_univ["delta_abs_min"])].copy()
    if float(cfg_univ.get("vol_fin_min", 0.0)) > 0:
        d = d[d["vol_fin"] >= float(cfg_univ["vol_fin_min"])].copy()

    return d
