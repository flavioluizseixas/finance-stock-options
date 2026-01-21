import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any
from .base import Strategy, StrategyResult
from features.liquidity import liq_leg_from_row, liq_pair_metrics, liq_class_pair, liq_penalty
from features.common import ensure_date

def _pick_short_long_expiries(df: pd.DataFrame, expiry_sel: date | None, long_steps: int) -> tuple[date | None, date | None]:
    expiries = sorted(df["expiry_date"].dropna().unique().tolist())
    expiries = [pd.to_datetime(x).date() for x in expiries]
    if not expiries:
        return None, None

    if expiry_sel is None:
        short_ex = expiries[0]
    else:
        short_ex = expiry_sel if expiry_sel in expiries else expiries[0]

    i = expiries.index(short_ex)
    j = min(i + int(long_steps), len(expiries) - 1)
    long_ex = expiries[j] if j != i else None
    return short_ex, long_ex

class BoosterHorizontalPuts(Strategy):
    key = "booster_puts"
    name = "Booster Horizontal (PUTs)"
    kind = "calendar"

    def candidates(self, df, expiry_sel, cfg, top_n):
        # df pode conter vários ativos. Vamos processar por ticker.
        rows = []

        long_steps = int(cfg.get("booster_long_steps", 2))
        target = float(cfg.get("booster_put_delta_target", -0.25))
        band = float(cfg.get("booster_put_delta_band", 0.10))
        use_same = bool(cfg.get("booster_use_same_strike", True))
        max_gap = float(cfg.get("booster_max_strike_gap_pct", 0.02))

        for ticker, d0 in df.groupby("ticker"):
            d0 = d0.copy()
            short_ex, long_ex = _pick_short_long_expiries(d0, expiry_sel, long_steps)
            if short_ex is None or long_ex is None:
                continue

            puts_short = d0[(d0["expiry_date"] == short_ex) & (d0["option_type"].str.upper()=="PUT")].copy()
            puts_long  = d0[(d0["expiry_date"] == long_ex)  & (d0["option_type"].str.upper()=="PUT")].copy()
            if puts_short.empty or puts_long.empty:
                continue

            puts_short["delta"] = pd.to_numeric(puts_short.get("delta", np.nan), errors="coerce")
            puts_long["delta"]  = pd.to_numeric(puts_long.get("delta", np.nan), errors="coerce")
            puts_short["strike"] = pd.to_numeric(puts_short.get("strike", np.nan), errors="coerce")
            puts_long["strike"]  = pd.to_numeric(puts_long.get("strike", np.nan), errors="coerce")
            puts_short["last_price"] = pd.to_numeric(puts_short.get("last_price", np.nan), errors="coerce")
            puts_long["last_price"]  = pd.to_numeric(puts_long.get("last_price", np.nan), errors="coerce")
            puts_short["t_years"] = pd.to_numeric(puts_short.get("t_years", np.nan), errors="coerce")
            puts_long["t_years"]  = pd.to_numeric(puts_long.get("t_years", np.nan), errors="coerce")

            # banda de delta na curta (onde vamos vender theta)
            lo = target - band
            hi = target + band
            sells = puts_short[puts_short["delta"].between(lo, hi) & (puts_short["last_price"] > 0)].copy()
            if sells.empty:
                # fallback: pega OTM/ATM por liquidez
                sells = puts_short[puts_short["moneyness"].isin(["OTM","ATM"])].copy()

            sells = sells.sort_values(["trades","volume"], ascending=[False,False]).head(30)

            # pool longa: precisa existir e ser negociada
            buys = puts_long[(puts_long["last_price"] > 0)].copy()
            buys = buys.sort_values(["trades","volume"], ascending=[False,False]).head(80)

            if buys.empty or sells.empty:
                continue

            for _, s in sells.iterrows():
                K = float(s["strike"])
                if use_same:
                    cand = buys.copy()
                    cand["dK"] = (cand["strike"] - K).abs()
                    if max_gap > 0:
                        cand = cand[cand["dK"] <= max_gap * K]
                    b = cand.sort_values(["dK","trades","volume"], ascending=[True,False,False]).head(1)
                else:
                    # permite strikes próximos
                    cand = buys.copy()
                    cand["dK"] = (cand["strike"] - K).abs()
                    b = cand.sort_values(["dK","trades","volume"], ascending=[True,False,False]).head(1)

                if b.empty:
                    continue
                b = b.iloc[0]

                P_short = float(s["last_price"])
                P_long  = float(b["last_price"])
                net = P_short - P_long  # crédito se positivo

                liq_s, _, _ = liq_leg_from_row(s)
                liq_b, _, _ = liq_leg_from_row(b)
                min_liq, _, ratio = liq_pair_metrics(liq_s, liq_b)
                liq_class = liq_class_pair(min_liq, ratio, cfg)

                if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                    continue

                # score: preferir crédito, liquidez e delta bem no alvo
                dpen = abs(float(s["delta"]) - target)
                score = (net * 100.0) + (min_liq * 2.0) - (dpen * 10.0) - (liq_penalty(liq_class, cfg) * 2.0)

                rows.append({
                    "strategy": self.name,
                    "ticker": ticker,
                    "trade_date": s["trade_date"],
                    "regime": s.get("regime","N/A"),
                    "spot_ref": float(s["spot_ref"]) if pd.notna(s.get("spot_ref", np.nan)) else np.nan,

                    "expiry_short": ensure_date(short_ex),
                    "expiry_long": ensure_date(long_ex),

                    "K": float(K),
                    "sell_short": s["option_symbol"],
                    "buy_long": b["option_symbol"],
                    "P_short": P_short,
                    "P_long":  P_long,
                    "net_credit": net,

                    "t_short": float(s.get("t_years", 0.0)) if pd.notna(s.get("t_years", np.nan)) else 0.0,
                    "t_long":  float(b.get("t_years", 0.0)) if pd.notna(b.get("t_years", np.nan)) else 0.0,

                    "liq_min": min_liq,
                    "liq_ratio": ratio,
                    "liq_class": liq_class,
                    "score": score,
                })

        if not rows:
            return StrategyResult(pd.DataFrame())

        out = pd.DataFrame(rows)
        out = out.sort_values(["score","liq_min"], ascending=[False,False]).head(top_n)
        return StrategyResult(out)

    def payoff_spec(self, row, cfg):
        return {
            "label": f"{row['ticker']} | Booster PUTs | short={ensure_date(row['expiry_short'])} sell={row['sell_short']} | long={ensure_date(row['expiry_long'])} buy={row['buy_long']}",
            "payoff_mode": "calendar_approx",
            "spot_ref": float(row["spot_ref"]),
            "K": float(row["K"]),
            "premium_long": float(row["P_long"]),
            "premium_short": float(row["P_short"]),
            "t_short": float(row.get("t_short", 0.0)),
            "t_long": float(row.get("t_long", 0.0)),
        }
