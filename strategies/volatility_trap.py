from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any

from .base import Strategy, StrategyResult
from features.liquidity import liq_leg_from_row, liq_pair_metrics, liq_class_pair, liq_penalty

def _filter_expiry(df: pd.DataFrame, expiry: Optional[date]):
    if expiry is None:
        return df.copy()
    return df[df["expiry_date"] == expiry].copy()

def _ensure_date(x):
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return x

class VolatilityTrapRatio(Strategy):
    """
    Trava de volatilidade (ratio spread a crédito):
      - Buy 1 opção ATM (ou próxima)
      - Sell N opções OTM extremo
      - Net credit > 0

    Implementa para CALL e PUT no mesmo ranking (coluna option_type).
    """
    key = "vol_trap_ratio_credit"
    name = "Trava de Volatilidade (Ratio a Crédito – ATM long + OTM extremo short)"
    kind = "multi"

    def candidates(self, df: pd.DataFrame, expiry_sel: Optional[date], cfg: Dict[str, Any], top_n: int) -> StrategyResult:
        d = _filter_expiry(df, expiry_sel)
        if d is None or d.empty:
            return StrategyResult(pd.DataFrame())

        # numeric
        for c in ["strike","last_price","delta","iv","trades","volume","spot_ref","moneyness_dist"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.dropna(subset=["strike","last_price","spot_ref"]).copy()
        d = d[(d["last_price"] > 0) & (d["spot_ref"] > 0)].copy()
        if d.empty:
            return StrategyResult(pd.DataFrame())

        ratios = cfg.get("voltrap_ratios", [2, 3])
        # thresholds
        otm_delta_max_call = float(cfg.get("voltrap_call_otm_delta_max", 0.08))
        otm_delta_max_put  = float(cfg.get("voltrap_put_otm_delta_abs_max", 0.08))
        otm_dist_min = float(cfg.get("voltrap_otm_dist_min", 0.08))  # 8% away default
        # liquidity hard gates (pair)
        hard_min = float(cfg.get("liq_pair_hard_min", 0.0))
        hard_ratio = float(cfg.get("liq_pair_hard_ratio", 0.0))

        rows = []

        # We'll search per ticker & option_type & expiry
        grp_cols = ["ticker","expiry_date","option_type"]
        for (ticker, ex, opt_type), dd in d.groupby(grp_cols):
            if dd.empty:
                continue
            spot = float(dd["spot_ref"].dropna().iloc[0]) if dd["spot_ref"].notna().any() else np.nan
            if not np.isfinite(spot) or spot <= 0:
                continue

            opt_type_u = str(opt_type).upper()

            # Buy candidate: ATM preferred; fallback nearest to ATM.
            buy_pool = dd.copy()
            if "moneyness" in buy_pool.columns:
                buy_pool = buy_pool[buy_pool["moneyness"].isin(["ATM","OTM","ITM"])].copy()
            buy_pool["atm_dist"] = (buy_pool["strike"] - spot).abs() / spot
            buy_pool = buy_pool.sort_values(["atm_dist","trades","volume"], ascending=[True, False, False]).head(25)

            # Sell pool: extreme OTM in the direction that creates tail risk
            sell_pool = dd.copy()
            if opt_type_u == "CALL":
                sell_pool["otm_dist"] = (sell_pool["strike"] - spot) / spot
                sell_pool = sell_pool[sell_pool["otm_dist"] >= otm_dist_min].copy()
                # prefer small delta if available
                if "delta" in sell_pool.columns and sell_pool["delta"].notna().any():
                    sell_pool = sell_pool[(sell_pool["delta"] > 0) & (sell_pool["delta"] <= otm_delta_max_call)].copy()
            else:  # PUT
                sell_pool["otm_dist"] = (spot - sell_pool["strike"]) / spot
                sell_pool = sell_pool[sell_pool["otm_dist"] >= otm_dist_min].copy()
                if "delta" in sell_pool.columns and sell_pool["delta"].notna().any():
                    # delta negative for puts; use abs
                    sell_pool = sell_pool[(sell_pool["delta"] < 0) & (sell_pool["delta"].abs() <= otm_delta_max_put)].copy()

            sell_pool = sell_pool.sort_values(["trades","volume"], ascending=[False, False]).head(40)
            if buy_pool.empty or sell_pool.empty:
                continue

            # Build ratio candidates
            for _, b in buy_pool.iterrows():
                P_buy = float(b["last_price"])
                iv_buy = float(b["iv"]) if ("iv" in b and pd.notna(b["iv"])) else np.nan
                K_buy = float(b["strike"])

                liq_b, _, _ = liq_leg_from_row(b)

                for _, s in sell_pool.iterrows():
                    K_sell = float(s["strike"])
                    # Direction checks: for call, sell strike above buy-ish; for put, sell strike below buy-ish
                    if opt_type_u == "CALL" and K_sell <= K_buy:
                        continue
                    if opt_type_u == "PUT" and K_sell >= K_buy:
                        continue

                    P_sell = float(s["last_price"])
                    iv_sell = float(s["iv"]) if ("iv" in s and pd.notna(s["iv"])) else np.nan
                    liq_s, _, _ = liq_leg_from_row(s)

                    min_liq, _, ratio_liq = liq_pair_metrics(liq_b, liq_s)
                    liq_class = liq_class_pair(min_liq, ratio_liq, cfg)

                    if (min_liq < hard_min) or (ratio_liq < hard_ratio):
                        continue

                    for n in ratios:
                        credit = (n * P_sell) - P_buy
                        if credit <= 0:
                            continue

                        skew = (iv_sell - iv_buy) if (np.isfinite(iv_sell) and np.isfinite(iv_buy)) else 0.0

                        score = (
                            credit * float(cfg.get("voltrap_credit_weight", 10.0))
                            + skew * float(cfg.get("voltrap_skew_weight", 50.0))
                            + min_liq
                            - liq_penalty(liq_class, cfg) * 0.5
                        )

                        rows.append({
                            "strategy": "Trava de Vol (Ratio Credit)",
                            "ticker": ticker,
                            "trade_date": b.get("trade_date"),
                            "expiry_date": _ensure_date(ex),
                            "option_type": opt_type_u,
                            "spot_ref": spot,
                            "buy": b["option_symbol"],
                            "sell": s["option_symbol"],
                            "ratio_n": int(n),
                            "K_buy": K_buy,
                            "K_sell": K_sell,
                            "P_buy": P_buy,
                            "P_sell": P_sell,
                            "credit": credit,
                            "iv_buy": float(iv_buy) if np.isfinite(iv_buy) else np.nan,
                            "iv_sell": float(iv_sell) if np.isfinite(iv_sell) else np.nan,
                            "skew_iv": float(skew),
                            "liq_buy": float(liq_b),
                            "liq_sell": float(liq_s),
                            "liq_min": float(min_liq),
                            "liq_ratio": float(ratio_liq),
                            "liq_class": liq_class,
                            "score": float(score),
                            "regime": b.get("regime", "N/A")
                        })

        out = pd.DataFrame(rows) if rows else pd.DataFrame()
        if out.empty:
            return StrategyResult(out)

        out = out.sort_values(["score","credit","liq_min"], ascending=[False, False, False]).head(top_n)
        return StrategyResult(out)

    def payoff_spec(self, row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
        spot = float(row.get("spot_ref", 1.0))
        opt_type = str(row.get("option_type","CALL")).upper()
        n = int(row.get("ratio_n", 2))

        legs = []
        # long 1 ATM-ish
        legs.append({"type": opt_type, "side": "LONG", "K": float(row["K_buy"]), "premium": float(row["P_buy"])})
        # short N extreme OTM
        legs.append({"type": opt_type, "side": "SHORT", "K": float(row["K_sell"]), "premium": float(row["P_sell"]), "qty": float(n)})

        # For payoff engine: support qty at leg level (we'll handle in renderer/payoff_strategy via repeat or qty)
        # Here we pass qty, and payoff_strategy will multiply.
        label = f"{row['ticker']} | Trava de Vol (Ratio Credit) – exp={row['expiry_date']} | BUY 1 {row['buy']} | SELL {n}× {row['sell']} | credit={float(row['credit']):.2f}"

        # payoff range hint for tail risk
        if opt_type == "CALL":
            payoff_range_mult = (0.5, float(cfg.get("voltrap_call_hi_mult", 3.0)))
        else:
            payoff_range_mult = (float(cfg.get("voltrap_put_lo_mult", 0.1)), 1.5)

        return {
            "payoff_mode": "expiry",
            "label": label,
            "spot_ref": spot,
            "legs": legs,
            "include_stock": False,
            "payoff_range_mult": payoff_range_mult,
        }
