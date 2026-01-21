import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any
from .base import Strategy, StrategyResult
from features.liquidity import liquidity_score_df, liq_class_single, liq_penalty
from features.common import ensure_date

def _filter_expiry(df: pd.DataFrame, expiry: date | None):
    if expiry is None:
        return df.copy()
    return df[df["expiry_date"] == expiry].copy()

class BuyDeepITMCall(Strategy):
    key = "buy_deep_itm_call"
    name = "Compra bem ITM (CALL deep ITM)"
    kind = "single"

    def candidates(self, df, expiry_sel, cfg, top_n):
        d = _filter_expiry(df, expiry_sel)
        d = d[d["option_type"].str.upper() == "CALL"].copy()

        d["delta"] = pd.to_numeric(d.get("delta", np.nan), errors="coerce")
        d["iv"] = pd.to_numeric(d.get("iv", np.nan), errors="coerce")
        d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
        d["mispricing_pct"] = pd.to_numeric(d.get("mispricing_pct", np.nan), errors="coerce")

        def dmin_for_reg(r):
            if r == "Alta":
                return cfg["deep_itm_delta_up"]
            if r == "Baixa":
                return cfg["deep_itm_delta_down"]
            return cfg["deep_itm_delta_neutral"]

        d = d[d["moneyness"] == "ITM"].copy()
        if d.empty:
            return StrategyResult(d)

        d["dmin"] = d["regime"].apply(dmin_for_reg)
        d = d[(d["delta"] >= d["dmin"]) & (d["iv"].notna()) & (d["last_price"] > 0)].copy()
        if d.empty:
            return StrategyResult(d)

        d["liq"] = liquidity_score_df(d)
        d["liq_class"] = d.apply(lambda r: liq_class_single(
            liq=float(pd.to_numeric(r.get("liq", 0), errors="coerce") or 0),
            trades=float(pd.to_numeric(r.get("trades", 0), errors="coerce") or 0),
            volume=float(pd.to_numeric(r.get("volume", 0), errors="coerce") or 0),
            cfg=cfg
        ), axis=1)

        if cfg.get("liq_single_filter_hard", True):
            d = d[d["liq_class"].isin(["OK", "ALERTA"])].copy()
            if d.empty:
                return StrategyResult(d)

        d["cheapness"] = d["mispricing_pct"].fillna(0)
        d["score"] = (
            d["liq"]
            - (d["iv"].fillna(0) * 10.0)
            + np.where(d["regime"] == "Baixa", (-d["cheapness"] * 0.5), 0.0)
            - d["liq_class"].map(lambda c: liq_penalty(c, cfg))
        )

        out = d.sort_values(["score", "liq"], ascending=[False, False]).head(top_n)
        out["expiry_date"] = out["expiry_date"].apply(ensure_date)
        out["strategy"] = self.name
        return StrategyResult(out.drop(columns=["dmin"], errors="ignore"))

    def payoff_spec(self, row, cfg):
        return {
            "label": f"{row['ticker']} | Compra ITM – LONG CALL {row['option_symbol']} (exp={ensure_date(row['expiry_date'])}, K={row['strike']}, P={row['last_price']})",
            "payoff_mode": "expiry",
            "spot_ref": float(row["spot_ref"]),
            "legs": [{"type":"CALL","side":"LONG","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": False,
        }

class SellPut(Strategy):
    key = "sell_put"
    name = "Venda de PUTs (renda)"
    kind = "single"

    def candidates(self, df, expiry_sel, cfg, top_n):
        d = _filter_expiry(df, expiry_sel)
        d = d[d["option_type"].str.upper() == "PUT"].copy()

        d["delta"] = pd.to_numeric(d.get("delta", np.nan), errors="coerce")
        d["iv"] = pd.to_numeric(d.get("iv", np.nan), errors="coerce")
        d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
        d["moneyness_dist"] = pd.to_numeric(d.get("moneyness_dist", np.nan), errors="coerce")

        def bounds(r):
            if r == "Baixa":
                return cfg["put_delta_lo_bear"], cfg["put_delta_hi_bear"], cfg["put_min_otm_dist_bear"]
            return cfg["put_delta_lo"], cfg["put_delta_hi"], 0.0

        d = d[d["moneyness"].isin(["OTM", "ATM"])].copy()
        if d.empty:
            return StrategyResult(d)

        d[["lo", "hi", "min_dist"]] = d["regime"].apply(lambda r: pd.Series(bounds(r)))
        d = d[(d["delta"].between(d["lo"], d["hi"])) & (d["iv"].notna()) & (d["last_price"] > 0)].copy()
        d = d[~((d["min_dist"] > 0) & ((d["moneyness"] != "OTM") | (d["moneyness_dist"] < d["min_dist"])))].copy()
        if d.empty:
            return StrategyResult(d)

        d["liq"] = liquidity_score_df(d)
        d["liq_class"] = d.apply(lambda r: liq_class_single(
            liq=float(pd.to_numeric(r.get("liq", 0), errors="coerce") or 0),
            trades=float(pd.to_numeric(r.get("trades", 0), errors="coerce") or 0),
            volume=float(pd.to_numeric(r.get("volume", 0), errors="coerce") or 0),
            cfg=cfg
        ), axis=1)

        if cfg.get("liq_single_filter_hard", True):
            d = d[d["liq_class"].isin(["OK", "ALERTA"])].copy()
            if d.empty:
                return StrategyResult(d)

        d["score"] = (
            (d["iv"].fillna(0) * 100.0)
            + (d["last_price"].fillna(0) * 5.0)
            + d["liq"].fillna(0)
            - d["liq_class"].map(lambda c: liq_penalty(c, cfg))
        )

        out = d.sort_values(["score"], ascending=[False]).head(top_n)
        out["expiry_date"] = out["expiry_date"].apply(ensure_date)
        out["strategy"] = self.name
        return StrategyResult(out.drop(columns=["lo", "hi", "min_dist"], errors="ignore"))

    def payoff_spec(self, row, cfg):
        return {
            "label": f"{row['ticker']} | Venda de PUT – SHORT PUT {row['option_symbol']} (exp={ensure_date(row['expiry_date'])}, K={row['strike']}, P={row['last_price']})",
            "payoff_mode": "expiry",
            "spot_ref": float(row["spot_ref"]),
            "legs": [{"type":"PUT","side":"SHORT","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": False,
        }

class CoveredCall(Strategy):
    key = "covered_call"
    name = "Venda Coberta (1 ação + short CALL)"
    kind = "single"

    def candidates(self, df, expiry_sel, cfg, top_n):
        d = _filter_expiry(df, expiry_sel)
        d = d[d["option_type"].str.upper() == "CALL"].copy()

        d["delta"] = pd.to_numeric(d.get("delta", np.nan), errors="coerce")
        d["iv"] = pd.to_numeric(d.get("iv", np.nan), errors="coerce")
        d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")

        def params(r):
            if r == "Alta":
                return ["OTM"], cfg["cc_delta_up"]
            if r == "Baixa":
                return ["OTM", "ATM"], cfg["cc_delta_down"]
            return ["OTM", "ATM"], cfg["cc_delta_neutral"]

        d[["mset", "dlo", "dhi"]] = d["regime"].apply(lambda r: pd.Series([params(r)[0], params(r)[1][0], params(r)[1][1]]))
        ok_m = d.apply(lambda x: x["moneyness"] in x["mset"], axis=1)
        d = d[ok_m].copy()
        d = d[(d["delta"].between(d["dlo"], d["dhi"])) & (d["iv"].notna()) & (d["last_price"] > 0)].copy()
        if d.empty:
            return StrategyResult(d)

        d["liq"] = liquidity_score_df(d)
        d["liq_class"] = d.apply(lambda r: liq_class_single(
            liq=float(pd.to_numeric(r.get("liq", 0), errors="coerce") or 0),
            trades=float(pd.to_numeric(r.get("trades", 0), errors="coerce") or 0),
            volume=float(pd.to_numeric(r.get("volume", 0), errors="coerce") or 0),
            cfg=cfg
        ), axis=1)

        if cfg.get("liq_single_filter_hard", True):
            d = d[d["liq_class"].isin(["OK", "ALERTA"])].copy()
            if d.empty:
                return StrategyResult(d)

        d["score"] = (
            (d["last_price"].fillna(0) * 10.0)
            + d["liq"].fillna(0)
            - d["liq_class"].map(lambda c: liq_penalty(c, cfg))
        )

        out = d.sort_values(["score"], ascending=[False]).head(top_n)
        out["expiry_date"] = out["expiry_date"].apply(ensure_date)
        out["strategy"] = self.name
        return StrategyResult(out.drop(columns=["mset", "dlo", "dhi"], errors="ignore"))

    def payoff_spec(self, row, cfg):
        return {
            "label": f"{row['ticker']} | Venda Coberta – exp={ensure_date(row['expiry_date'])} | STOCK + SHORT CALL {row['option_symbol']} (K={row['strike']}, P={row['last_price']})",
            "payoff_mode": "expiry",
            "spot_ref": float(row["spot_ref"]),
            "legs": [{"type":"CALL","side":"SHORT","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": True,
            "stock_qty": 1.0
        }
