import numpy as np
import pandas as pd

from .types import Strategy, StrategyResult
from features.liquidity import liquidity_score_df
from features.common import to_float


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _liq(df: pd.DataFrame) -> pd.Series:
    # trades + log(1+volume)
    return liquidity_score_df(df)


def _spot_from_row(row: pd.Series, fallback: float = 1.0) -> float:
    s = to_float(row.get("spot_ref"))
    if s is None or not np.isfinite(s) or s <= 0:
        s = to_float(row.get("spot"))
    if s is None or not np.isfinite(s) or s <= 0:
        s = float(fallback)
    return float(s)


# ============================================================
# 1) Compra bem ITM (CALL deep ITM)
# ============================================================



def _spot_from_df(df: pd.DataFrame) -> float:
    """Best-effort spot reference for a (sub)chain."""
    if df is None or df.empty:
        return 1.0
    for c in ["spot_ref", "spot", "close"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                v = float(s.median())
                if np.isfinite(v) and v > 0:
                    return v
    return 1.0

def _cand_buy_deep_itm_call(df: pd.DataFrame, expiry, cfg: dict, top_n: int) -> StrategyResult:
    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]
    d = d[d["option_type"].astype(str).str.upper() == "CALL"].copy()
    d = d[d.get("moneyness").isin(["ITM"])].copy()

    d["delta"] = _num(d.get("delta"))
    d["iv"] = _num(d.get("iv"))
    d["last_price"] = _num(d.get("last_price"))

    # regime-aware delta floor
    regime = str(d.get("regime", "Neutra").iloc[0]) if len(d) else "Neutra"
    dmin = float(cfg.get("deep_itm_delta_neutral", 0.80))
    if regime == "Alta":
        dmin = float(cfg.get("deep_itm_delta_up", 0.75))
    elif regime == "Baixa":
        dmin = float(cfg.get("deep_itm_delta_down", 0.85))

    d = d[(d["delta"].notna()) & (d["delta"] >= dmin) & (d["iv"].notna()) & (d["last_price"] > 0)].copy()
    if d.empty:
        return StrategyResult(pd.DataFrame(), ["Sem CALL ITM elegível"], [])

    d["liq"] = _liq(d)
    out = d.sort_values(["liq", "iv"], ascending=[False, True]).head(int(top_n)).copy()
    return StrategyResult(out, ["CALL ITM", f"delta ≥ {dmin:.2f}", "rank: liquidez alta e IV menor"], [])


def _payoff_buy_deep_itm_call(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Compra ITM – LONG CALL {row['option_symbol']} (K={row['strike']}, prémio={row['last_price']})",
        "spot_ref": _spot_from_row(row),
        "legs": [{"type": "CALL", "side": "LONG", "K": float(row["strike"]), "premium": float(row["last_price"])}],
        "include_stock": False,
    }


# ============================================================
# 2) Venda de PUTs (renda)
# ============================================================


def _cand_sell_put(df: pd.DataFrame, expiry, cfg: dict, top_n: int) -> StrategyResult:
    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]
    d = d[d["option_type"].astype(str).str.upper() == "PUT"].copy()

    d["delta"] = _num(d.get("delta"))
    d["iv"] = _num(d.get("iv"))
    d["last_price"] = _num(d.get("last_price"))
    d["moneyness_dist"] = _num(d.get("moneyness_dist"))

    regime = str(d.get("regime", "Neutra").iloc[0]) if len(d) else "Neutra"
    lo = float(cfg.get("put_delta_lo", -0.35))
    hi = float(cfg.get("put_delta_hi", -0.10))
    min_dist = 0.0
    if regime == "Baixa":
        lo = float(cfg.get("put_delta_lo_bear", -0.25))
        hi = float(cfg.get("put_delta_hi_bear", -0.08))
        min_dist = float(cfg.get("put_min_otm_dist_bear", 0.015))

    d = d[
        (d.get("moneyness").isin(["OTM", "ATM"]))
        & (d["delta"].between(lo, hi))
        & (d["iv"].notna())
        & (d["last_price"] > 0)
    ].copy()

    if min_dist > 0:
        d = d[(d.get("moneyness") == "OTM") & (d["moneyness_dist"] >= min_dist)].copy()

    if d.empty:
        return StrategyResult(pd.DataFrame(), ["Sem PUT elegível"], [])

    d["liq"] = _liq(d)
    d["score"] = (d["iv"].fillna(0) * 100.0) + (d["last_price"].fillna(0) * 5.0) + d["liq"].fillna(0)
    out = d.sort_values(["score"], ascending=[False]).head(int(top_n)).copy()
    crit = [
        "PUT OTM/ATM",
        f"delta em [{lo:.2f}, {hi:.2f}]",
        (f"moneyness_dist ≥ {min_dist:.3f}" if min_dist > 0 else ""),
        "rank: IV, prêmio e liquidez",
    ]
    crit = [c for c in crit if c]
    return StrategyResult(out, crit, [])


def _payoff_sell_put(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Venda de PUT – SHORT PUT {row['option_symbol']} (K={row['strike']}, prémio={row['last_price']})",
        "spot_ref": _spot_from_row(row),
        "legs": [{"type": "PUT", "side": "SHORT", "K": float(row["strike"]), "premium": float(row["last_price"])}],
        "include_stock": False,
    }


# ============================================================
# 3) Venda coberta (1 ação + short CALL)
# ============================================================


def _cand_covered_call(df: pd.DataFrame, expiry, cfg: dict, top_n: int) -> StrategyResult:
    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]
    d = d[d["option_type"].astype(str).str.upper() == "CALL"].copy()

    d["delta"] = _num(d.get("delta"))
    d["iv"] = _num(d.get("iv"))
    d["last_price"] = _num(d.get("last_price"))

    regime = str(d.get("regime", "Neutra").iloc[0]) if len(d) else "Neutra"
    if regime == "Alta":
        mset = ["OTM"]
        dlo, dhi = cfg.get("cc_delta_up", (0.12, 0.25))
    elif regime == "Baixa":
        mset = ["OTM", "ATM"]
        dlo, dhi = cfg.get("cc_delta_down", (0.20, 0.40))
    else:
        mset = ["OTM", "ATM"]
        dlo, dhi = cfg.get("cc_delta_neutral", (0.15, 0.30))
    dlo, dhi = float(dlo), float(dhi)

    d = d[(d.get("moneyness").isin(mset)) & (d["delta"].between(dlo, dhi)) & (d["last_price"] > 0)].copy()
    if d.empty:
        return StrategyResult(pd.DataFrame(), ["Sem CALL elegível"], [])

    d["liq"] = _liq(d)
    d["score"] = (d["last_price"].fillna(0) * 10.0) + d["liq"].fillna(0)
    out = d.sort_values(["score"], ascending=[False]).head(int(top_n)).copy()
    return StrategyResult(out, ["STOCK + SHORT CALL", f"delta em [{dlo:.2f},{dhi:.2f}]", f"moneyness ∈ {mset}", "rank: prêmio e liquidez"], [])


def _payoff_covered_call(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Venda Coberta – STOCK + SHORT CALL {row['option_symbol']} (K={row['strike']}, prémio={row['last_price']})",
        "spot_ref": _spot_from_row(row),
        "legs": [{"type": "CALL", "side": "SHORT", "K": float(row["strike"]), "premium": float(row["last_price"])}],
        "include_stock": True,
        "stock_qty": 1.0,
    }


# ============================================================
# 4) Travas verticais (Bull Call / Bear Put)
# ============================================================


def _cand_vertical_spreads(df: pd.DataFrame, expiry, kind: str, cfg: dict, top_n: int) -> StrategyResult:
    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]

    d["strike"] = _num(d.get("strike"))
    d["last_price"] = _num(d.get("last_price"))
    d = d.dropna(subset=["strike", "last_price"]).copy()
    d = d[d["last_price"] > 0].copy()
    if d.empty:
        return StrategyResult(pd.DataFrame(), ["Sem opções elegíveis"], [])

    rows = []
    if kind == "bull_call":
        calls = d[d["option_type"].astype(str).str.upper() == "CALL"].copy()
        buy = calls[calls.get("moneyness") == "ATM"].copy()
        sell = calls[calls.get("moneyness") == "OTM"].copy()
        for _, b in buy.iterrows():
            cand = sell[sell["strike"] > b["strike"]].sort_values("strike").head(12)
            for _, s in cand.iterrows():
                debit = float(b["last_price"]) - float(s["last_price"])
                if debit <= 0:
                    continue
                max_profit = (float(s["strike"]) - float(b["strike"])) - debit
                if max_profit <= 0:
                    continue
                rr = max_profit / debit
                rows.append(
                    {
                        "strategy": "Bull Call Spread",
                        "buy": b["option_symbol"],
                        "sell": s["option_symbol"],
                        "K_buy": float(b["strike"]),
                        "K_sell": float(s["strike"]),
                        "P_buy": float(b["last_price"]),
                        "P_sell": float(s["last_price"]),
                        "debit": debit,
                        "max_profit": max_profit,
                        "rr": rr,
                        "spot_ref": _spot_from_row(b),
                    }
                )
    else:
        puts = d[d["option_type"].astype(str).str.upper() == "PUT"].copy()
        buy = puts[puts.get("moneyness") == "ATM"].copy()
        sell = puts[puts.get("moneyness") == "OTM"].copy()
        for _, b in buy.iterrows():
            cand = sell[sell["strike"] < b["strike"]].sort_values("strike", ascending=False).head(12)
            for _, s in cand.iterrows():
                debit = float(b["last_price"]) - float(s["last_price"])
                if debit <= 0:
                    continue
                max_profit = (float(b["strike"]) - float(s["strike"])) - debit
                if max_profit <= 0:
                    continue
                rr = max_profit / debit
                rows.append(
                    {
                        "strategy": "Bear Put Spread",
                        "buy": b["option_symbol"],
                        "sell": s["option_symbol"],
                        "K_buy": float(b["strike"]),
                        "K_sell": float(s["strike"]),
                        "P_buy": float(b["last_price"]),
                        "P_sell": float(s["last_price"]),
                        "debit": debit,
                        "max_profit": max_profit,
                        "rr": rr,
                        "spot_ref": _spot_from_row(b),
                    }
                )

    if not rows:
        return StrategyResult(pd.DataFrame(), ["Sem combinações válidas"], [])
    out = pd.DataFrame(rows).sort_values(["rr", "max_profit"], ascending=[False, False]).head(int(top_n))
    return StrategyResult(out, ["BUY ATM, SELL OTM", "debit > 0", "rank: RR (max_profit/debit)"], [])


def _payoff_bull_call(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Bull Call Spread – BUY {row['buy']} | SELL {row['sell']} (debit={row['debit']:.2f})",
        "spot_ref": _spot_from_row(row),
        "legs": [
            {"type": "CALL", "side": "LONG", "K": float(row["K_buy"]), "premium": float(row["P_buy"])},
            {"type": "CALL", "side": "SHORT", "K": float(row["K_sell"]), "premium": float(row["P_sell"])},
        ],
        "include_stock": False,
    }


def _payoff_bear_put(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Bear Put Spread – BUY {row['buy']} | SELL {row['sell']} (debit={row['debit']:.2f})",
        "spot_ref": _spot_from_row(row),
        "legs": [
            {"type": "PUT", "side": "LONG", "K": float(row["K_buy"]), "premium": float(row["P_buy"])},
            {"type": "PUT", "side": "SHORT", "K": float(row["K_sell"]), "premium": float(row["P_sell"])},
        ],
        "include_stock": False,
    }



# ============================================================
# 4b) Credit Spreads (Bear Call / Bull Put) – Top CR
# ============================================================

def _cand_credit_spreads(df: pd.DataFrame, expiry, kind: str, cfg: dict, top_n: int) -> StrategyResult:
    """
    kind:
      - "bear_call": SELL CALL (near ATM/OTM) + BUY higher strike CALL (further OTM)  -> credit
      - "bull_put":  SELL PUT  (near ATM/OTM) + BUY lower strike PUT (further OTM)   -> credit
    Score: credit-to-risk (cr) with liquidity tiebreak.
    """
    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]

    d["strike"] = _num(d.get("strike"))
    d["last_price"] = _num(d.get("last_price"))
    d["delta"] = _num(d.get("delta"))
    d["trades"] = _num(d.get("trades")).fillna(0)
    d["volume"] = _num(d.get("volume")).fillna(0)
    d = d.dropna(subset=["strike", "last_price", "delta"]).copy()
    d = d[d["last_price"] > 0].copy()
    if d.empty:
        return StrategyResult(pd.DataFrame())

    # Liquidity score
    try:
        d["liq"] = liquidity_score_df(d)
    except Exception:
        d["liq"] = d["trades"] + np.log1p(d["volume"].clip(lower=0))

    # Default delta bands (can be overridden in cfg)
    call_sell_band = tuple(cfg.get("credit_call_sell_band", (0.15, 0.30)))
    call_buy_band  = tuple(cfg.get("credit_call_buy_band",  (0.05, 0.15)))
    put_sell_band  = tuple(cfg.get("credit_put_sell_band",  (-0.30, -0.15)))
    put_buy_band   = tuple(cfg.get("credit_put_buy_band",   (-0.15, -0.05)))

    spot_ref = _spot_from_df(d)
    rows = []

    if kind == "bear_call":
        calls = d[d["option_type"].str.upper() == "CALL"].copy()
        sells = calls[calls["delta"].between(*call_sell_band)].copy()
        buys  = calls[calls["delta"].between(*call_buy_band)].copy()
        if sells.empty or buys.empty:
            return StrategyResult(pd.DataFrame())

        sells = sells.sort_values(["liq", "trades", "volume"], ascending=[False, False, False]).head(20)

        for _, s in sells.iterrows():
            cand = buys[buys["strike"] > s["strike"]].sort_values("strike").head(30)
            for _, b in cand.iterrows():
                credit = float(s["last_price"]) - float(b["last_price"])
                if credit <= 0:
                    continue
                width = float(b["strike"]) - float(s["strike"])
                max_loss = width - credit
                if max_loss <= 0:
                    continue
                cr = credit / max_loss
                liq_pair = float(min(s["liq"], b["liq"]))
                row = {
                    "strategy": "Bear Call Credit Spread",
                    "spot_ref": spot_ref,
                "sell": s["option_symbol"], "buy": b["option_symbol"],
                    "K_sell": float(s["strike"]), "K_buy": float(b["strike"]),
                    "P_sell": float(s["last_price"]), "P_buy": float(b["last_price"]),
                    "credit": credit, "max_loss_est": max_loss, "cr": cr,
                    "liq_min": liq_pair,
                }
                if "ticker" in d.columns:
                    row["ticker"] = s.get("ticker")
                if "expiry_date" in d.columns:
                    row["expiry_date"] = s.get("expiry_date")
                rows.append(row)

    else:  # bull_put
        puts = d[d["option_type"].str.upper() == "PUT"].copy()
        sells = puts[puts["delta"].between(*put_sell_band)].copy()
        buys  = puts[puts["delta"].between(*put_buy_band)].copy()
        if sells.empty or buys.empty:
            return StrategyResult(pd.DataFrame())

        sells = sells.sort_values(["liq", "trades", "volume"], ascending=[False, False, False]).head(20)

        for _, s in sells.iterrows():
            cand = buys[buys["strike"] < s["strike"]].sort_values("strike", ascending=False).head(30)
            for _, b in cand.iterrows():
                credit = float(s["last_price"]) - float(b["last_price"])
                if credit <= 0:
                    continue
                width = float(s["strike"]) - float(b["strike"])
                max_loss = width - credit
                if max_loss <= 0:
                    continue
                cr = credit / max_loss
                liq_pair = float(min(s["liq"], b["liq"]))
                row = {
                    "strategy": "Bull Put Credit Spread",
                    "spot_ref": spot_ref,
                "sell": s["option_symbol"], "buy": b["option_symbol"],
                    "K_sell": float(s["strike"]), "K_buy": float(b["strike"]),
                    "P_sell": float(s["last_price"]), "P_buy": float(b["last_price"]),
                    "credit": credit, "max_loss_est": max_loss, "cr": cr,
                    "liq_min": liq_pair,
                }
                if "ticker" in d.columns:
                    row["ticker"] = s.get("ticker")
                if "expiry_date" in d.columns:
                    row["expiry_date"] = s.get("expiry_date")
                rows.append(row)

    if not rows:
        return StrategyResult(pd.DataFrame())

    out = pd.DataFrame(rows)
    out = out.sort_values(["cr", "credit", "liq_min"], ascending=[False, False, False]).head(int(top_n))
    bullets = [
        "Trava de crédito (SELL + BUY OTM, mesmo vencimento)",
        "ranking: maior credit-to-risk (cr) e liquidez",
    ]
    notes = [
        "⚠️ Tail risk pode ficar escondido se o gráfico tiver faixa curta. Use payoff AUTO para ver melhor."
    ]
    return StrategyResult(out, bullets, notes)


def _payoff_bear_call_credit(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Bear Call Credit Spread – SELL {row['sell']} | BUY {row['buy']} (credit={row['credit']:.2f})",
        "spot_ref": _spot_from_row(row),
        "legs": [
            {"type": "CALL", "side": "SHORT", "K": float(row["K_sell"]), "premium": float(row["P_sell"])},
            {"type": "CALL", "side": "LONG",  "K": float(row["K_buy"]),  "premium": float(row["P_buy"])},
        ],
        "include_stock": False,
    }


def _payoff_bull_put_credit(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Bull Put Credit Spread – SELL {row['sell']} | BUY {row['buy']} (credit={row['credit']:.2f})",
        "spot_ref": _spot_from_row(row),
        "legs": [
            {"type": "PUT", "side": "SHORT", "K": float(row["K_sell"]), "premium": float(row["P_sell"])},
            {"type": "PUT", "side": "LONG",  "K": float(row["K_buy"]),  "premium": float(row["P_buy"])},
        ],
        "include_stock": False,
    }

# ============================================================
# 5) Long Straddle ATM
# ============================================================


def _cand_long_straddle(df: pd.DataFrame, expiry, cfg: dict, top_n: int) -> StrategyResult:
    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]

    d["strike"] = _num(d.get("strike"))
    d["last_price"] = _num(d.get("last_price"))
    d = d.dropna(subset=["strike", "last_price"]).copy()
    d = d[d["last_price"] > 0].copy()

    calls = d[(d["option_type"].astype(str).str.upper() == "CALL") & (d.get("moneyness") == "ATM")].copy()
    puts = d[(d["option_type"].astype(str).str.upper() == "PUT") & (d.get("moneyness") == "ATM")].copy()
    if calls.empty or puts.empty:
        return StrategyResult(pd.DataFrame(), ["Sem CALL/PUT ATM"], [])

    rows = []
    for _, c in calls.iterrows():
        ptmp = puts.copy()
        ptmp["dK"] = (ptmp["strike"] - c["strike"]).abs()
        p = ptmp.sort_values(["dK", "trades", "volume"], ascending=[True, False, False]).head(1)
        if p.empty:
            continue
        p = p.iloc[0]
        prem = float(c["last_price"]) + float(p["last_price"])
        liq = float(_liq(pd.DataFrame([c])).iloc[0]) + float(_liq(pd.DataFrame([p])).iloc[0])
        rows.append(
            {
                "strategy": "Long Straddle (ATM)",
                "call": c["option_symbol"],
                "put": p["option_symbol"],
                "K_call": float(c["strike"]),
                "K_put": float(p["strike"]),
                "P_call": float(c["last_price"]),
                "P_put": float(p["last_price"]),
                "premium_total": prem,
                "liq": liq,
                "spot_ref": _spot_from_row(c),
            }
        )

    if not rows:
        return StrategyResult(pd.DataFrame(), ["Sem pares válidos"], [])

    out = pd.DataFrame(rows).sort_values(["liq", "premium_total"], ascending=[False, True]).head(int(top_n))
    return StrategyResult(out, ["BUY CALL ATM + BUY PUT ATM", "rank: maior liquidez e menor prêmio"], [])


def _payoff_long_straddle(row: pd.Series, cfg: dict) -> dict:
    return {
        "label": f"Long Straddle – BUY {row['call']} + BUY {row['put']} (premio_total={row['premium_total']:.2f})",
        "spot_ref": _spot_from_row(row),
        "legs": [
            {"type": "CALL", "side": "LONG", "K": float(row["K_call"]), "premium": float(row["P_call"])},
            {"type": "PUT", "side": "LONG", "K": float(row["K_put"]), "premium": float(row["P_put"])},
        ],
        "include_stock": False,
    }


# ============================================================
# 6) Short Condor puro (CALL/PUT) – crédito
# ============================================================


def _pick_nearest_strike_row(d: pd.DataFrame, target_k: float, avoid: set):
    tmp = d.copy()
    tmp["dist"] = (tmp["strike"] - target_k).abs()
    tmp = tmp[~tmp["option_symbol"].isin(avoid)]
    tmp = tmp.sort_values(["dist", "trades", "volume"], ascending=[True, False, False])
    if tmp.empty:
        return None
    return tmp.iloc[0]


def _condor_candidates(df: pd.DataFrame, expiry, opt_type: str, cfg: dict, top_n: int) -> pd.DataFrame:
    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]
    d = d[d["option_type"].astype(str).str.upper() == opt_type]

    d["strike"] = _num(d.get("strike"))
    d["last_price"] = _num(d.get("last_price"))
    d["delta"] = _num(d.get("delta"))
    d["trades"] = _num(d.get("trades")).fillna(0)
    d["volume"] = _num(d.get("volume")).fillna(0)
    d = d.dropna(subset=["strike", "last_price", "delta"]).copy()
    d = d[d["last_price"] > 0].sort_values("strike")
    if d.empty:
        return pd.DataFrame()

    regime = str(d.get("regime", "Neutra").iloc[0]) if len(d) else "Neutra"
    if opt_type == "CALL":
        sell_band = cfg.get("condor_call_sell_band", (0.18, 0.32))
        buy_band = cfg.get("condor_call_buy_band", (0.06, 0.16))
        if regime == "Alta":
            sell_band = cfg.get("condor_call_sell_band_up", (0.12, 0.22))
            buy_band = cfg.get("condor_call_buy_band_up", (0.04, 0.10))
    else:
        sell_band = cfg.get("condor_put_sell_band", (-0.32, -0.18))
        buy_band = cfg.get("condor_put_buy_band", (-0.16, -0.06))
        if regime == "Baixa":
            sell_band = cfg.get("condor_put_sell_band_down", (-0.22, -0.12))
            buy_band = cfg.get("condor_put_buy_band_down", (-0.10, -0.04))

    sell_band = (float(sell_band[0]), float(sell_band[1]))
    buy_band = (float(buy_band[0]), float(buy_band[1]))

    sells = d[d["delta"].between(*sell_band)].copy()
    buys = d[d["delta"].between(*buy_band)].copy()
    if sells.empty or buys.empty:
        return pd.DataFrame()

    sells = sells.sort_values("strike").head(12)
    spot_ref = _spot_from_df(d)
    rows = []
    for i in range(min(len(sells), 8)):
        for j in range(i + 1, min(len(sells), 10)):
            s1 = sells.iloc[i]
            s4 = sells.iloc[j]
            K1, K4 = float(s1["strike"]), float(s4["strike"])
            if K4 <= K1:
                continue

            b_in = buys[(buys["strike"] > K1) & (buys["strike"] < K4)].copy()
            if b_in.empty:
                continue

            gap = max((K4 - K1) / 3.0, 0.01)
            target_k2 = K1 + gap
            target_k3 = K4 - gap

            avoid = {s1["option_symbol"], s4["option_symbol"]}
            b2 = _pick_nearest_strike_row(b_in, target_k2, avoid)
            if b2 is None:
                continue
            avoid.add(b2["option_symbol"])
            b3 = _pick_nearest_strike_row(b_in, target_k3, avoid)
            if b3 is None:
                continue

            K2, K3 = float(b2["strike"]), float(b3["strike"])
            if not (K1 < K2 < K3 < K4):
                continue

            P_s1 = float(s1["last_price"])
            P_s4 = float(s4["last_price"])
            P_b2 = float(b2["last_price"])
            P_b3 = float(b3["last_price"])

            credit = (P_s1 + P_s4) - (P_b2 + P_b3)
            if credit <= 0:
                continue

            w_left = (K2 - K1)
            w_right = (K4 - K3)
            max_loss_est = max(w_left, w_right) - credit
            if max_loss_est <= 0:
                continue

            cr = credit / max_loss_est
            liq_sum = float(_liq(pd.DataFrame([s1])).iloc[0]) + float(_liq(pd.DataFrame([b2])).iloc[0]) + float(_liq(pd.DataFrame([b3])).iloc[0]) + float(_liq(pd.DataFrame([s4])).iloc[0])
            rows.append(
                {
                    "strategy": f"Short {opt_type} Condor (Credit)",
                    "sell_1": s1["option_symbol"],
                    "buy_2": b2["option_symbol"],
                    "buy_3": b3["option_symbol"],
                    "sell_4": s4["option_symbol"],
                    "K1": K1,
                    "K2": K2,
                    "K3": K3,
                    "K4": K4,
                    "P_sell_1": P_s1,
                    "P_buy_2": P_b2,
                    "P_buy_3": P_b3,
                    "P_sell_4": P_s4,
                    "credit": credit,
                    "max_loss_est": max_loss_est,
                    "cr": cr,
                    "liq": liq_sum,
                    "spot_ref": _spot_from_row(s1),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["cr", "credit", "liq"], ascending=[False, False, False]).head(int(top_n))
    return out


def _cand_short_call_condor(df: pd.DataFrame, expiry, cfg: dict, top_n: int) -> StrategyResult:
    out = _condor_candidates(df, expiry, "CALL", cfg, top_n)
    return StrategyResult(out, ["SELL/BUY/BUY/SELL (mesma família)", "crédito > 0", "rank: credit-to-risk"], [])


def _cand_short_put_condor(df: pd.DataFrame, expiry, cfg: dict, top_n: int) -> StrategyResult:
    out = _condor_candidates(df, expiry, "PUT", cfg, top_n)
    return StrategyResult(out, ["SELL/BUY/BUY/SELL (mesma família)", "crédito > 0", "rank: credit-to-risk"], [])


def _payoff_condor(row: pd.Series, cfg: dict, opt_type: str) -> dict:
    return {
        "label": f"Short {opt_type} Condor – credit={row['credit']:.2f} | {row['sell_1']} / {row['buy_2']} / {row['buy_3']} / {row['sell_4']}",
        "spot_ref": _spot_from_row(row),
        "legs": [
            {"type": opt_type, "side": "SHORT", "K": float(row["K1"]), "premium": float(row["P_sell_1"])},
            {"type": opt_type, "side": "LONG", "K": float(row["K2"]), "premium": float(row["P_buy_2"])},
            {"type": opt_type, "side": "LONG", "K": float(row["K3"]), "premium": float(row["P_buy_3"])},
            {"type": opt_type, "side": "SHORT", "K": float(row["K4"]), "premium": float(row["P_sell_4"])},
        ],
        "include_stock": False,
    }


# ============================================================
# 7) "Trava de volatilidade" (Call Ratio) – crédito
# ============================================================


def _cand_vol_trap_call_ratio(df: pd.DataFrame, expiry, cfg: dict, top_n: int) -> StrategyResult:
    """Buy 1 CALL ATM, Sell n CALLs far OTM, credit (ratio spread).

    Parâmetros (cfg):
      - ratio_n (default 2)
      - ratio_otm_dist_min (default 0.05)  # moneyness_dist mínimo para o short
      - ratio_short_delta_max (default 0.10)  # delta máximo para o short
    """

    d = df.copy()
    if expiry is not None:
        d = d[d["expiry_date"] == expiry]
    d = d[d["option_type"].astype(str).str.upper() == "CALL"].copy()
    if d.empty:
        return StrategyResult(pd.DataFrame(), ["Sem CALLs"], [])

    d["strike"] = _num(d.get("strike"))
    d["last_price"] = _num(d.get("last_price"))
    d["delta"] = _num(d.get("delta"))
    d["moneyness_dist"] = _num(d.get("moneyness_dist"))
    d = d.dropna(subset=["strike", "last_price", "delta"]).copy()
    d = d[d["last_price"] > 0].copy()

    n = int(cfg.get("ratio_n", 2))
    min_dist = float(cfg.get("ratio_otm_dist_min", 0.05))
    dmax = float(cfg.get("ratio_short_delta_max", 0.10))

    buy = d[d.get("moneyness") == "ATM"].copy()
    sell = d[(d.get("moneyness") == "OTM") & (d["moneyness_dist"] >= min_dist) & (d["delta"] <= dmax)].copy()
    if buy.empty or sell.empty:
        return StrategyResult(pd.DataFrame(), ["Sem ATM/OTM extremos elegíveis"], [])

    buy = buy.sort_values(["trades", "volume"], ascending=[False, False]).head(10)
    sell = sell.sort_values(["moneyness_dist", "trades", "volume"], ascending=[False, False, False]).head(20)

    rows = []
    for _, b in buy.iterrows():
        # escolha short strike maior (mais OTM) do que o strike do ATM
        s_cand = sell[sell["strike"] > b["strike"]].sort_values("strike")
        for _, s in s_cand.head(12).iterrows():
            credit = (n * float(s["last_price"])) - float(b["last_price"])
            if credit <= 0:
                continue
            tail_k = float(s["strike"])
            liq_sum = float(_liq(pd.DataFrame([b])).iloc[0]) + float(_liq(pd.DataFrame([s])).iloc[0])
            # score: mais crédito e mais liquidez; penaliza short menos OTM
            score = credit * 10.0 + liq_sum + float(s.get("moneyness_dist", 0.0)) * 100.0
            rows.append(
                {
                    "strategy": f"Call Ratio (1x{n}) – crédito",
                    "buy": b["option_symbol"],
                    "sell": s["option_symbol"],
                    "K_buy": float(b["strike"]),
                    "K_sell": float(s["strike"]),
                    "P_buy": float(b["last_price"]),
                    "P_sell": float(s["last_price"]),
                    "n_short": n,
                    "credit": credit,
                    "tail_strike": tail_k,
                    "liq": liq_sum,
                    "score": score,
                    "spot_ref": _spot_from_row(b),
                }
            )

    if not rows:
        return StrategyResult(pd.DataFrame(), ["Sem combinações a crédito"], ["Atenção: risco explode acima do strike vendido."])

    out = pd.DataFrame(rows).sort_values(["score"], ascending=[False]).head(int(top_n))
    return StrategyResult(out, ["BUY 1 CALL ATM", f"SELL {n} CALL OTM (dist≥{min_dist:.2f})", "montada a crédito", "⚠️ risco de cauda no upside"], ["No payoff, aumente hi_mult (ex.: 2.5–3.0×S0) para enxergar o tail risk."])


def _payoff_vol_trap_call_ratio(row: pd.Series, cfg: dict) -> dict:
    n = int(row.get("n_short", 2))
    legs = [{"type": "CALL", "side": "LONG", "K": float(row["K_buy"]), "premium": float(row["P_buy"])}]
    for _ in range(n):
        legs.append({"type": "CALL", "side": "SHORT", "K": float(row["K_sell"]), "premium": float(row["P_sell"])})
    return {
        "label": f"Call Ratio (1x{n}) – BUY {row['buy']} | SELL {n}x {row['sell']} (credit={row['credit']:.2f})",
        "spot_ref": _spot_from_row(row),
        "legs": legs,
        "include_stock": False,
    }


# ============================================================
# Public registry
# ============================================================


def get_strategies():
    return [
        Strategy(
            key="buy_deep_itm_call",
            name="Compra bem ITM (CALL deep ITM)",
            candidates=_cand_buy_deep_itm_call,
            payoff_spec=_payoff_buy_deep_itm_call,
        ),
        Strategy(
            key="sell_put",
            name="Venda de PUTs (renda)",
            candidates=_cand_sell_put,
            payoff_spec=_payoff_sell_put,
        ),
        Strategy(
            key="covered_call",
            name="Venda coberta (STOCK + Short CALL)",
            candidates=_cand_covered_call,
            payoff_spec=_payoff_covered_call,
        ),
        Strategy(
            key="bull_call_spread",
            name="Trava de alta (Bull Call Spread)",
            candidates=lambda df, expiry, cfg, top_n: _cand_vertical_spreads(df, expiry, "bull_call", cfg, top_n),
            payoff_spec=_payoff_bull_call,
        ),
        Strategy(
            key="bear_put_spread",
            name="Trava de baixa (Bear Put Spread)",
            candidates=lambda df, expiry, cfg, top_n: _cand_vertical_spreads(df, expiry, "bear_put", cfg, top_n),
            payoff_spec=_payoff_bear_put,
        ),
        Strategy(
            key="bear_call_credit_spread",
            name="Trava de crédito (Bear Call Credit Spread)",
            candidates=lambda df, expiry, cfg, top_n: _cand_credit_spreads(df, expiry, "bear_call", cfg, top_n),
            payoff_spec=_payoff_bear_call_credit,
        ),
        Strategy(
            key="bull_put_credit_spread",
            name="Trava de crédito (Bull Put Credit Spread)",
            candidates=lambda df, expiry, cfg, top_n: _cand_credit_spreads(df, expiry, "bull_put", cfg, top_n),
            payoff_spec=_payoff_bull_put_credit,
        ),
        Strategy(
            key="long_straddle_atm",
            name="Long Straddle (ATM)",
            candidates=_cand_long_straddle,
            payoff_spec=_payoff_long_straddle,
        ),
        Strategy(
            key="short_call_condor",
            name="Short Call Condor (Crédito)",
            candidates=_cand_short_call_condor,
            payoff_spec=lambda row, cfg: _payoff_condor(row, cfg, "CALL"),
        ),
        Strategy(
            key="short_put_condor",
            name="Short Put Condor (Crédito)",
            candidates=_cand_short_put_condor,
            payoff_spec=lambda row, cfg: _payoff_condor(row, cfg, "PUT"),
        ),
        Strategy(
            key="vol_trap_call_ratio",
            name="Trava de volatilidade (Call Ratio – crédito)",
            candidates=_cand_vol_trap_call_ratio,
            payoff_spec=_payoff_vol_trap_call_ratio,
        ),
    ]
