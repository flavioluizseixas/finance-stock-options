import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any
from .base import Strategy, StrategyResult
from features.liquidity import liq_leg_from_row, liq_pair_metrics, liq_class_pair, liq_penalty
from features.common import ensure_date

def _top_vertical_spreads_debit_one_expiry(df, expiry: date, kind="bull_call", top_n: int = 5, cfg=None) -> pd.DataFrame:
    cfg = cfg or {}
    d = df[df["expiry_date"] == expiry].copy()
    d["strike"] = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
    d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
    d = d.dropna(subset=["strike", "last_price"]).copy()
    d = d[d["last_price"] > 0].copy()

    rows = []
    if kind == "bull_call":
        calls = d[d["option_type"].str.upper() == "CALL"].copy()
        buy = calls[calls["moneyness"] == "ATM"].copy()
        sell = calls[calls["moneyness"] == "OTM"].copy()

        buy = buy.sort_values(["ticker", "trades", "volume"], ascending=[True, False, False]).head(120)
        sell = sell.sort_values(["ticker", "trades", "volume"], ascending=[True, False, False]).head(240)

        for _, b in buy.iterrows():
            cand = sell[(sell["ticker"] == b["ticker"]) & (sell["strike"] > b["strike"])].sort_values("strike").head(18)
            for _, s in cand.iterrows():
                debit = float(b["last_price"]) - float(s["last_price"])
                if debit <= 0:
                    continue
                max_profit = (float(s["strike"]) - float(b["strike"])) - debit
                if max_profit <= 0:
                    continue
                rr = max_profit / debit

                liq_b, _, _ = liq_leg_from_row(b)
                liq_s, _, _ = liq_leg_from_row(s)
                min_liq, _, ratio = liq_pair_metrics(liq_b, liq_s)
                liq_class = liq_class_pair(min_liq, ratio, cfg)

                if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                    continue

                rows.append({
                    "strategy": "Bull Call Spread (Debit)",
                    "ticker": b["ticker"],
                    "trade_date": b["trade_date"],
                    "expiry_date": ensure_date(expiry),
                    "spot_ref": float(b["spot_ref"]) if pd.notna(b["spot_ref"]) else np.nan,
                    "buy": b["option_symbol"],
                    "sell": s["option_symbol"],
                    "K_buy": float(b["strike"]),
                    "K_sell": float(s["strike"]),
                    "P_buy": float(b["last_price"]),
                    "P_sell": float(s["last_price"]),
                    "debit": debit,
                    "max_profit_est": max_profit,
                    "rr": rr,
                    "liq_min": min_liq,
                    "liq_ratio": ratio,
                    "liq_class": liq_class,
                    "regime": b.get("regime", "N/A")
                })
    else:
        puts = d[d["option_type"].str.upper() == "PUT"].copy()
        buy = puts[puts["moneyness"] == "ATM"].copy()
        sell = puts[puts["moneyness"] == "OTM"].copy()

        buy = buy.sort_values(["ticker", "trades", "volume"], ascending=[True, False, False]).head(120)
        sell = sell.sort_values(["ticker", "trades", "volume"], ascending=[True, False, False]).head(240)

        for _, b in buy.iterrows():
            cand = sell[(sell["ticker"] == b["ticker"]) & (sell["strike"] < b["strike"])].sort_values("strike", ascending=False).head(18)
            for _, s in cand.iterrows():
                debit = float(b["last_price"]) - float(s["last_price"])
                if debit <= 0:
                    continue
                max_profit = (float(b["strike"]) - float(s["strike"])) - debit
                if max_profit <= 0:
                    continue
                rr = max_profit / debit

                liq_b, _, _ = liq_leg_from_row(b)
                liq_s, _, _ = liq_leg_from_row(s)
                min_liq, _, ratio = liq_pair_metrics(liq_b, liq_s)
                liq_class = liq_class_pair(min_liq, ratio, cfg)

                if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                    continue

                rows.append({
                    "strategy": "Bear Put Spread (Debit)",
                    "ticker": b["ticker"],
                    "trade_date": b["trade_date"],
                    "expiry_date": ensure_date(expiry),
                    "spot_ref": float(b["spot_ref"]) if pd.notna(b["spot_ref"]) else np.nan,
                    "buy": b["option_symbol"],
                    "sell": s["option_symbol"],
                    "K_buy": float(b["strike"]),
                    "K_sell": float(s["strike"]),
                    "P_buy": float(b["last_price"]),
                    "P_sell": float(s["last_price"]),
                    "debit": debit,
                    "max_profit_est": max_profit,
                    "rr": rr,
                    "liq_min": min_liq,
                    "liq_ratio": ratio,
                    "liq_class": liq_class,
                    "regime": b.get("regime", "N/A")
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["rr_adj"] = out["rr"] - out["liq_class"].map(lambda c: liq_penalty(c, cfg) * 0.05)
    return out.sort_values(["rr_adj", "max_profit_est", "liq_min"], ascending=[False, False, False]).head(top_n)

def _top_credit_spreads_one_expiry(df, expiry: date, kind="bull_put", cfg=None, top_n: int = 5) -> pd.DataFrame:
    cfg = cfg or {}
    d = df[df["expiry_date"] == expiry].copy()
    d["strike"] = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
    d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
    d["delta"] = pd.to_numeric(d.get("delta", np.nan), errors="coerce")
    d["trades"] = pd.to_numeric(d.get("trades", 0), errors="coerce").fillna(0)
    d["volume"] = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)
    d = d.dropna(subset=["strike", "last_price"]).copy()
    d = d[d["last_price"] > 0].copy()

    rows = []
    if kind == "bull_put":
        puts = d[d["option_type"].str.upper() == "PUT"].copy().sort_values("strike")
        for ticker, dd in puts.groupby("ticker"):
            reg = str(dd["regime"].iloc[0]) if "regime" in dd.columns and len(dd) else "N/A"
            if reg == "Baixa":
                sell_band = cfg.get("bps_put_sell_band_down", (-0.25, -0.12))
                buy_band  = cfg.get("bps_put_buy_band_down",  (-0.12, -0.04))
            else:
                sell_band = cfg.get("bps_put_sell_band", (-0.30, -0.15))
                buy_band  = cfg.get("bps_put_buy_band",  (-0.15, -0.05))

            sells = dd[dd["delta"].between(*sell_band)].copy()
            buys  = dd[dd["delta"].between(*buy_band)].copy()

            if sells.empty:
                sells = dd[dd["moneyness"].isin(["OTM", "ATM"])].copy()
            if buys.empty:
                buys = dd[dd["moneyness"].isin(["OTM"])].copy()

            sells = sells.sort_values(["trades", "volume"], ascending=[False, False]).head(16)
            buys  = buys.sort_values(["trades", "volume"], ascending=[False, False]).head(32)

            for _, s in sells.iterrows():
                cand = buys[buys["strike"] < s["strike"]].sort_values("strike", ascending=False).head(18)
                for _, b in cand.iterrows():
                    credit = float(s["last_price"]) - float(b["last_price"])
                    if credit <= 0:
                        continue
                    width = float(s["strike"]) - float(b["strike"])
                    if width <= 0:
                        continue
                    max_loss = width - credit
                    if max_loss <= 0:
                        continue
                    cr = credit / max_loss

                    liq_s, _, _ = liq_leg_from_row(s)
                    liq_b, _, _ = liq_leg_from_row(b)
                    min_liq, _, ratio = liq_pair_metrics(liq_s, liq_b)
                    liq_class = liq_class_pair(min_liq, ratio, cfg)

                    if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                        continue

                    rows.append({
                        "strategy": "Bull Put Spread (Credit)",
                        "ticker": ticker,
                        "trade_date": s["trade_date"],
                        "expiry_date": ensure_date(expiry),
                        "spot_ref": float(s["spot_ref"]) if pd.notna(s["spot_ref"]) else np.nan,
                        "sell": s["option_symbol"],
                        "buy":  b["option_symbol"],
                        "K_sell": float(s["strike"]),
                        "K_buy":  float(b["strike"]),
                        "P_sell": float(s["last_price"]),
                        "P_buy":  float(b["last_price"]),
                        "credit": credit,
                        "width": width,
                        "max_loss_est": max_loss,
                        "cr": cr,
                        "liq_min": min_liq,
                        "liq_ratio": ratio,
                        "liq_class": liq_class,
                        "regime": reg
                    })
    else:
        calls = d[d["option_type"].str.upper() == "CALL"].copy().sort_values("strike")
        for ticker, dd in calls.groupby("ticker"):
            reg = str(dd["regime"].iloc[0]) if "regime" in dd.columns and len(dd) else "N/A"
            if reg == "Alta":
                sell_band = cfg.get("bcs_call_sell_band_up", (0.12, 0.22))
                buy_band  = cfg.get("bcs_call_buy_band_up",  (0.04, 0.10))
            else:
                sell_band = cfg.get("bcs_call_sell_band", (0.15, 0.30))
                buy_band  = cfg.get("bcs_call_buy_band",  (0.05, 0.15))

            sells = dd[dd["delta"].between(*sell_band)].copy()
            buys  = dd[dd["delta"].between(*buy_band)].copy()

            if sells.empty:
                sells = dd[dd["moneyness"].isin(["OTM", "ATM"])].copy()
            if buys.empty:
                buys = dd[dd["moneyness"].isin(["OTM"])].copy()

            sells = sells.sort_values(["trades", "volume"], ascending=[False, False]).head(16)
            buys  = buys.sort_values(["trades", "volume"], ascending=[False, False]).head(32)

            for _, s in sells.iterrows():
                cand = buys[buys["strike"] > s["strike"]].sort_values("strike").head(18)
                for _, b in cand.iterrows():
                    credit = float(s["last_price"]) - float(b["last_price"])
                    if credit <= 0:
                        continue
                    width = float(b["strike"]) - float(s["strike"])
                    if width <= 0:
                        continue
                    max_loss = width - credit
                    if max_loss <= 0:
                        continue
                    cr = credit / max_loss

                    liq_s, _, _ = liq_leg_from_row(s)
                    liq_b, _, _ = liq_leg_from_row(b)
                    min_liq, _, ratio = liq_pair_metrics(liq_s, liq_b)
                    liq_class = liq_class_pair(min_liq, ratio, cfg)

                    if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                        continue

                    rows.append({
                        "strategy": "Bear Call Spread (Credit)",
                        "ticker": ticker,
                        "trade_date": s["trade_date"],
                        "expiry_date": ensure_date(expiry),
                        "spot_ref": float(s["spot_ref"]) if pd.notna(s["spot_ref"]) else np.nan,
                        "sell": s["option_symbol"],
                        "buy":  b["option_symbol"],
                        "K_sell": float(s["strike"]),
                        "K_buy":  float(b["strike"]),
                        "P_sell": float(s["last_price"]),
                        "P_buy":  float(b["last_price"]),
                        "credit": credit,
                        "width": width,
                        "max_loss_est": max_loss,
                        "cr": cr,
                        "liq_min": min_liq,
                        "liq_ratio": ratio,
                        "liq_class": liq_class,
                        "regime": reg
                    })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["cr_adj"] = out["cr"] - out["liq_class"].map(lambda c: liq_penalty(c, cfg) * 0.05)
    out = out.sort_values(["cr_adj", "credit", "liq_min"], ascending=[False, False, False]).head(top_n)
    return out

class DebitSpreads(Strategy):
    key="debit_spreads"
    name="Travas no DÉBITO"
    kind="spread"

    def candidates(self, df, expiry_sel, cfg, top_n):
        expiries = sorted(df["expiry_date"].dropna().unique().tolist())
        expiries = [pd.to_datetime(x).date() for x in expiries]
        if expiry_sel is None and not expiries:
            return StrategyResult(pd.DataFrame())

        def collect(ex):
            a = _top_vertical_spreads_debit_one_expiry(df, ex, kind="bull_call", top_n=top_n, cfg=cfg)
            b = _top_vertical_spreads_debit_one_expiry(df, ex, kind="bear_put", top_n=top_n, cfg=cfg)
            return a, b

        if expiry_sel is not None:
            bull, bear = collect(expiry_sel)
        else:
            all_bull, all_bear = [], []
            for ex in expiries:
                b1, b2 = collect(ex)
                if not b1.empty: all_bull.append(b1)
                if not b2.empty: all_bear.append(b2)
            bull = pd.concat(all_bull, ignore_index=True) if all_bull else pd.DataFrame()
            bear = pd.concat(all_bear, ignore_index=True) if all_bear else pd.DataFrame()
            if not bull.empty:
                bull = bull.sort_values(["rr_adj","max_profit_est","liq_min"], ascending=[False,False,False]).head(top_n)
            if not bear.empty:
                bear = bear.sort_values(["rr_adj","max_profit_est","liq_min"], ascending=[False,False,False]).head(top_n)

        out = pd.concat([bull, bear], ignore_index=True) if (not bull.empty or not bear.empty) else pd.DataFrame()
        return StrategyResult(out)

    def payoff_spec(self, row, cfg):
        spot = float(row["spot_ref"])
        if "Bull Call" in row["strategy"]:
            legs = [
                {"type":"CALL","side":"LONG","K":float(row["K_buy"]),"premium":float(row["P_buy"])},
                {"type":"CALL","side":"SHORT","K":float(row["K_sell"]),"premium":float(row["P_sell"])},
            ]
            label = f"{row['ticker']} | Bull Call (Debit) – exp={ensure_date(row['expiry_date'])} | BUY {row['buy']} | SELL {row['sell']} (debit={row['debit']:.2f})"
        else:
            legs = [
                {"type":"PUT","side":"LONG","K":float(row["K_buy"]),"premium":float(row["P_buy"])},
                {"type":"PUT","side":"SHORT","K":float(row["K_sell"]),"premium":float(row["P_sell"])},
            ]
            label = f"{row['ticker']} | Bear Put (Debit) – exp={ensure_date(row['expiry_date'])} | BUY {row['buy']} | SELL {row['sell']} (debit={row['debit']:.2f})"
        return {"label": label, "payoff_mode":"expiry","spot_ref":spot,"legs":legs,"include_stock":False}

class CreditSpreads(Strategy):
    key="credit_spreads"
    name="Travas no CRÉDITO"
    kind="spread"

    def candidates(self, df, expiry_sel, cfg, top_n):
        expiries = sorted(df["expiry_date"].dropna().unique().tolist())
        expiries = [pd.to_datetime(x).date() for x in expiries]
        if expiry_sel is None and not expiries:
            return StrategyResult(pd.DataFrame())

        def collect(ex):
            bps = _top_credit_spreads_one_expiry(df, ex, kind="bull_put", cfg=cfg, top_n=top_n)
            bcs = _top_credit_spreads_one_expiry(df, ex, kind="bear_call", cfg=cfg, top_n=top_n)
            return bps, bcs

        if expiry_sel is not None:
            bps, bcs = collect(expiry_sel)
        else:
            all_bps, all_bcs = [], []
            for ex in expiries:
                a, b = collect(ex)
                if not a.empty: all_bps.append(a)
                if not b.empty: all_bcs.append(b)
            bps = pd.concat(all_bps, ignore_index=True) if all_bps else pd.DataFrame()
            bcs = pd.concat(all_bcs, ignore_index=True) if all_bcs else pd.DataFrame()
            if not bps.empty:
                bps = bps.sort_values(["cr_adj","credit","liq_min"], ascending=[False,False,False]).head(top_n)
            if not bcs.empty:
                bcs = bcs.sort_values(["cr_adj","credit","liq_min"], ascending=[False,False,False]).head(top_n)

        out = pd.concat([bps, bcs], ignore_index=True) if (not bps.empty or not bcs.empty) else pd.DataFrame()
        return StrategyResult(out)

    def payoff_spec(self, row, cfg):
        spot = float(row["spot_ref"])
        if "Bull Put" in row["strategy"]:
            legs = [
                {"type":"PUT","side":"SHORT","K":float(row["K_sell"]),"premium":float(row["P_sell"])},
                {"type":"PUT","side":"LONG", "K":float(row["K_buy"]), "premium":float(row["P_buy"])},
            ]
            label = f"{row['ticker']} | Bull Put (Credit) – exp={ensure_date(row['expiry_date'])} | SELL {row['sell']} | BUY {row['buy']} (credit={row['credit']:.2f})"
        else:
            legs = [
                {"type":"CALL","side":"SHORT","K":float(row["K_sell"]),"premium":float(row["P_sell"])},
                {"type":"CALL","side":"LONG", "K":float(row["K_buy"]), "premium":float(row["P_buy"])},
            ]
            label = f"{row['ticker']} | Bear Call (Credit) – exp={ensure_date(row['expiry_date'])} | SELL {row['sell']} | BUY {row['buy']} (credit={row['credit']:.2f})"
        return {"label": label, "payoff_mode":"expiry","spot_ref":spot,"legs":legs,"include_stock":False}
