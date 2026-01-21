import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any
from .base import Strategy, StrategyResult
from features.liquidity import liq_leg_from_row, liq_pair_metrics, liq_class_pair
from features.common import ensure_date

def _filter_expiry(df: pd.DataFrame, expiry: date | None):
    if expiry is None:
        return df.copy()
    return df[df["expiry_date"] == expiry].copy()

class LongStraddle(Strategy):
    key = "long_straddle"
    name = "Long Straddle (ATM)"
    kind = "complex"

    def candidates(self, df, expiry_sel, cfg, top_n):
        d = _filter_expiry(df, expiry_sel)
        d["strike"] = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
        d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
        d = d.dropna(subset=["strike", "last_price"]).copy()
        d = d[d["last_price"] > 0].copy()

        calls = d[(d["option_type"].str.upper() == "CALL") & (d["moneyness"] == "ATM")].copy()
        puts  = d[(d["option_type"].str.upper() == "PUT")  & (d["moneyness"] == "ATM")].copy()
        if calls.empty or puts.empty:
            return StrategyResult(pd.DataFrame())

        rows = []
        calls = calls.sort_values(["ticker", "expiry_date", "trades", "volume"], ascending=[True, True, False, False]).head(200)
        puts  = puts.sort_values(["ticker", "expiry_date", "trades", "volume"], ascending=[True, True, False, False]).head(200)

        for _, c in calls.iterrows():
            p_cand = puts[(puts["ticker"] == c["ticker"]) & (puts["expiry_date"] == c["expiry_date"])].copy()
            if p_cand.empty:
                continue
            p_cand["dK"] = (p_cand["strike"] - c["strike"]).abs()
            p = p_cand.sort_values(["dK", "trades", "volume"], ascending=[True, False, False]).head(1)
            if p.empty:
                continue
            p = p.iloc[0]

            liq_c, _, _ = liq_leg_from_row(c)
            liq_p, _, _ = liq_leg_from_row(p)
            min_liq, _, ratio = liq_pair_metrics(liq_c, liq_p)
            liq_class = liq_class_pair(min_liq, ratio, cfg)

            if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                continue

            prem = float(c["last_price"]) + float(p["last_price"])
            rows.append({
                "strategy": self.name,
                "ticker": c["ticker"],
                "trade_date": c["trade_date"],
                "expiry_date": ensure_date(c["expiry_date"]),
                "spot_ref": float(c["spot_ref"]) if pd.notna(c["spot_ref"]) else np.nan,
                "call": c["option_symbol"],
                "put": p["option_symbol"],
                "K": float(c["strike"]),
                "P_call": float(c["last_price"]),
                "P_put": float(p["last_price"]),
                "premium_total": prem,
                "liq_min": min_liq,
                "liq_ratio": ratio,
                "liq_class": liq_class,
                "regime": c.get("regime", "N/A")
            })

        if not rows:
            return StrategyResult(pd.DataFrame())

        out = pd.DataFrame(rows)
        out["premium_total"] = pd.to_numeric(out["premium_total"], errors="coerce")
        out["liq_min"] = pd.to_numeric(out["liq_min"], errors="coerce")
        out = out.sort_values(["liq_min", "premium_total"], ascending=[False, True]).head(top_n)
        return StrategyResult(out)

    def payoff_spec(self, row, cfg):
        return {
            "label": f"{row['ticker']} | Long Straddle – exp={ensure_date(row['expiry_date'])} | BUY {row['call']} + BUY {row['put']}",
            "payoff_mode": "expiry",
            "spot_ref": float(row["spot_ref"]),
            "legs": [
                {"type":"CALL","side":"LONG","K":float(row["K"]),"premium":float(row["P_call"])},
                {"type":"PUT","side":"LONG","K":float(row["K"]),"premium":float(row["P_put"])},
            ],
            "include_stock": False,
        }
