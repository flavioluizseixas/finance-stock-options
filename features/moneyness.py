import numpy as np
import pandas as pd

def classify_moneyness_multi(df: pd.DataFrame, atm_mode: str, atm_pct: float):
    d = df.copy()
    d["strike"] = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
    d["spot_ref"] = pd.to_numeric(d.get("spot_ref", np.nan), errors="coerce")

    ok = d["strike"].notna() & d["spot_ref"].notna() & (d["spot_ref"] > 0)
    d.loc[~ok, "moneyness"] = "UNKNOWN"
    d.loc[~ok, "moneyness_dist"] = np.nan

    d.loc[ok, "moneyness_dist"] = (d.loc[ok, "strike"] - d.loc[ok, "spot_ref"]).abs() / d.loc[ok, "spot_ref"]

    def _base(row):
        t = str(row["option_type"]).upper()
        K = float(row["strike"])
        S = float(row["spot_ref"])
        if t == "CALL":
            return "ITM" if K < S else "OTM"
        return "ITM" if K > S else "OTM"

    d["m_base"] = "UNKNOWN"
    d.loc[ok, "m_base"] = d.loc[ok].apply(_base, axis=1)

    if atm_mode == "pct":
        d.loc[ok, "moneyness"] = np.where(d.loc[ok, "moneyness_dist"] <= atm_pct, "ATM", d.loc[ok, "m_base"])
        return d.drop(columns=["m_base"])

    d.loc[ok, "moneyness"] = d.loc[ok, "m_base"]

    gcols = ["asset_id", "expiry_date", "option_type"]
    tmp = d.loc[ok].copy()
    idx = tmp.groupby(gcols)["moneyness_dist"].idxmin().dropna().astype(int).tolist()
    d.loc[idx, "moneyness"] = "ATM"
    return d.drop(columns=["m_base"])

def apply_universe_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df.copy()

    S = pd.to_numeric(d.get("spot_ref", np.nan), errors="coerce")
    K = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
    delta = pd.to_numeric(d.get("delta", np.nan), errors="coerce")
    lastp = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
    vol = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)

    ok = S.notna() & K.notna() & (S > 0) & (K > 0)
    log_mny = pd.Series(np.nan, index=d.index, dtype="float64")
    log_mny.loc[ok] = np.abs(np.log(K.loc[ok] / S.loc[ok]))

    vol_fin = lastp.fillna(0) * vol * float(cfg["opt_contract_mult"])

    d["mny_log_abs"] = log_mny
    d["volume_fin"] = vol_fin

    mask = (
        ok
        & (d["mny_log_abs"] <= float(cfg["mny_log_max"]))
        & (delta.abs() >= float(cfg["delta_abs_min"]))
        & (lastp >= float(cfg["last_price_min"]))
        & (d["volume_fin"] >= float(cfg["vol_fin_min"]))
    )

    return d[mask].copy()
