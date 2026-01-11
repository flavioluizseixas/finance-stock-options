# app_streamlit_options.py
# ------------------------------------------------------------
# Streamlit dashboard for options:
# - ITM/ATM/OTM tables, greeks, BSM/IV, last collection only
# - DI curve is NOT shown by vertices anymore (removed yield_curve table display)
# - Strategy picker (Top 3) with payoff chart shown ONLY when user selects an operation
#
# NEW in this version:
# - Uses technical indicators from daily_bars to infer "trend/regime"
#   (SMA20/50/200, MACD_hist, RSI14, ATR14)
# - Strategy filters adapt to the inferred regime and show criteria used on-screen
# - Clear "criteria box" explaining what was used for each strategy
#
# Includes:
# - Payoff with break-even(s)
# - Payoff in % of spot toggle
# - "per 100 shares" toggle
# - Strategies (Top 3):
#   1) Long deep ITM call
#   2) Short put (sell puts)
#   3) Vertical spreads (bull call / bear put)
#   4) Covered call
#   5) Long straddle
#   6) Short condor with CALLs and PUTs (pure condor, not iron)
#
# DB tables assumed (adjust SQL if your schema differs):
# - assets(id, ticker, is_active)
# - daily_bars(asset_id, trade_date, close, vol_annual,
#             sma_20, sma_50, sma_200,
#             macd, macd_signal, macd_hist,
#             rsi_14, atr_14)
# - option_quote(asset_id, trade_date, option_symbol, option_type, expiry_date,
#              strike, last_price, trades, volume, collected_at)
# - option_model(asset_id, trade_date, option_symbol, spot, rate_r, dividend_q, t_years,
#               iv, bsm_price, bsm_price_histvol, mispricing, mispricing_pct,
#               delta, gamma, vega, theta, rho, hist_vol_annual, collected_at)
#
# Streamlit note:
# - uses st.dataframe(..., on_select="rerun", selection_mode="single-row")
#   If your Streamlit version doesn't support it, it falls back to selectbox.
# ------------------------------------------------------------

import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finance Options Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", ""))
DB_NAME = os.getenv("DB_NAME", "finance_options")

if not DB_PASS:
    st.error("DB_PASSWORD/DB_PASS não definido no .env")
    st.stop()

# =========================
# DB helpers (cache)
# =========================

def get_conn():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        database=DB_NAME, charset="utf8mb4",
        cursorclass=DictCursor, autocommit=True,
    )

@st.cache_data(ttl=60)
def load_assets():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, ticker FROM assets WHERE is_active=1 ORDER BY ticker;")
            rows = cur.fetchall()
    return pd.DataFrame(rows)

@st.cache_data(ttl=60)
def load_latest_trade_date(asset_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(trade_date) AS trade_date FROM option_quote WHERE asset_id=%s", (asset_id,))
            row = cur.fetchone()
    return row["trade_date"] if row else None

@st.cache_data(ttl=60)
def load_daily_indicators(asset_id: int, trade_date: date):
    """
    Pull indicators from daily_bars for the selected trade_date.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  trade_date,
                  close,
                  vol_annual,
                  sma_20, sma_50, sma_200,
                  macd, macd_signal, macd_hist,
                  rsi_14,
                  atr_14
                FROM daily_bars
                WHERE asset_id=%s AND trade_date=%s
                """,
                (asset_id, trade_date),
            )
            row = cur.fetchone()
    return row

@st.cache_data(ttl=60)
def load_chain(asset_id: int, trade_date: date):
    """
    Loads options chain (filtered to trades>0 and last_price>0).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  oq.option_symbol,
                  oq.option_type,
                  oq.expiry_date,
                  oq.strike,
                  oq.last_price,
                  oq.trades,
                  oq.volume,
                  oq.collected_at AS quote_collected_at,

                  om.spot,
                  om.rate_r,
                  om.dividend_q,
                  om.t_years,
                  om.iv,
                  om.bsm_price,
                  om.bsm_price_histvol,
                  om.mispricing,
                  om.mispricing_pct,
                  om.delta,
                  om.gamma,
                  om.vega,
                  om.theta,
                  om.rho,
                  om.hist_vol_annual,
                  om.collected_at AS model_collected_at
                FROM option_quote oq
                LEFT JOIN option_model om
                  ON om.asset_id=oq.asset_id
                 AND om.trade_date=oq.trade_date
                 AND om.option_symbol=oq.option_symbol
                WHERE oq.asset_id=%s
                  AND oq.trade_date=%s
                  AND oq.trades > 0
                  AND oq.last_price > 0
                ORDER BY oq.expiry_date, oq.option_type, oq.strike;
                """,
                (asset_id, trade_date),
            )
            rows = cur.fetchall() or []
    return pd.DataFrame(rows)

# =========================
# Trend / regime inference
# =========================

def _to_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None

def infer_regime(ind: dict, rsi_hi: float = 55, rsi_lo: float = 45):
    """
    Returns:
      regime: "Alta" | "Baixa" | "Neutra" | "N/A"
      score_up, score_down, details(dict)
    """
    if not ind:
        return "N/A", 0, 0, {}

    close = _to_float(ind.get("close"))
    sma20 = _to_float(ind.get("sma_20"))
    sma50 = _to_float(ind.get("sma_50"))
    sma200 = _to_float(ind.get("sma_200"))
    macdh = _to_float(ind.get("macd_hist"))
    rsi = _to_float(ind.get("rsi_14"))
    atr = _to_float(ind.get("atr_14"))

    up = 0
    down = 0
    details = {}

    # SMA stack
    if (sma20 is not None) and (sma50 is not None) and (sma200 is not None):
        if sma20 > sma50 > sma200:
            up += 2
            details["SMA_stack"] = "Alta (SMA20>SMA50>SMA200)"
        elif sma20 < sma50 < sma200:
            down += 2
            details["SMA_stack"] = "Baixa (SMA20<SMA50<SMA200)"
        else:
            details["SMA_stack"] = "Misto"
    else:
        details["SMA_stack"] = "N/A"

    # close vs SMA50
    if (close is not None) and (sma50 is not None):
        if close > sma50:
            up += 1
            details["Close_vs_SMA50"] = "Acima"
        elif close < sma50:
            down += 1
            details["Close_vs_SMA50"] = "Abaixo"
        else:
            details["Close_vs_SMA50"] = "Em cima"
    else:
        details["Close_vs_SMA50"] = "N/A"

    # MACD hist
    if macdh is not None:
        if macdh > 0:
            up += 1
            details["MACD_hist"] = "Positivo"
        elif macdh < 0:
            down += 1
            details["MACD_hist"] = "Negativo"
        else:
            details["MACD_hist"] = "Zero"
    else:
        details["MACD_hist"] = "N/A"

    # RSI regime
    if rsi is not None:
        if rsi >= rsi_hi:
            up += 1
            details["RSI"] = f"Forte (≥{rsi_hi})"
        elif rsi <= rsi_lo:
            down += 1
            details["RSI"] = f"Fraco (≤{rsi_lo})"
        else:
            details["RSI"] = "Neutro"
    else:
        details["RSI"] = "N/A"

    # ATR (não vota para cima/baixo; só informa volatilidade/risco)
    details["ATR14"] = atr

    if up >= down + 2:
        return "Alta", up, down, details
    if down >= up + 2:
        return "Baixa", up, down, details
    return "Neutra", up, down, details

# =========================
# Moneyness + formatting
# =========================

def classify_moneyness(df: pd.DataFrame, spot: float, atm_mode: str, atm_pct: float):
    d = df.copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    if spot is None or not np.isfinite(spot) or spot <= 0:
        d["moneyness"] = "UNKNOWN"
        d["moneyness_dist"] = np.nan
        return d

    d["moneyness_dist"] = (d["strike"] - spot).abs() / spot

    def base_itm_otm(row):
        t = str(row["option_type"]).upper()
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

def format_table(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    num_cols = [
        "strike","last_price","trades","volume","moneyness_dist",
        "iv","bsm_price","bsm_price_histvol","mispricing","mispricing_pct",
        "delta","gamma","vega","theta","rho","rate_r"
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "theta" in out.columns:
        out["theta_day_365"] = out["theta"] / 365.0
        out["theta_day_252"] = out["theta"] / 252.0
    if "vega" in out.columns:
        out["vega_1pct"] = out["vega"] * 0.01
    if "iv" in out.columns:
        out["iv_pct"] = out["iv"] * 100.0

    cols = [
        "option_symbol","option_type","expiry_date",
        "strike","last_price","trades","volume",
        "moneyness","moneyness_dist",
        "rate_r","iv","iv_pct",
        "bsm_price","bsm_price_histvol",
        "mispricing","mispricing_pct",
        "delta","gamma","vega","vega_1pct",
        "theta","theta_day_365","theta_day_252","rho",
        "quote_collected_at","model_collected_at"
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()

    # arredondamento
    for c in ["strike","last_price"]:
        if c in out.columns:
            out[c] = out[c].round(2)
    for c in ["moneyness_dist","rate_r"]:
        if c in out.columns:
            out[c] = out[c].round(6 if c == "rate_r" else 4)
    for c in ["iv","iv_pct","bsm_price","bsm_price_histvol","mispricing","mispricing_pct","delta","gamma","vega","vega_1pct","theta","theta_day_365","theta_day_252","rho"]:
        if c in out.columns:
            out[c] = out[c].round(6 if c == "gamma" else 4)

    return out

# =========================
# Payoff logic (P&L at expiry)
# =========================

def payoff_long_call(ST, K, premium):
    return np.maximum(ST - K, 0.0) - premium

def payoff_short_call(ST, K, premium):
    return premium - np.maximum(ST - K, 0.0)

def payoff_long_put(ST, K, premium):
    return np.maximum(K - ST, 0.0) - premium

def payoff_short_put(ST, K, premium):
    return premium - np.maximum(K - ST, 0.0)

def payoff_stock(ST, S0):
    return ST - S0

def payoff_strategy(op: dict, ST: np.ndarray, spot: float) -> np.ndarray:
    """
    op: dict:
      - legs: list of {type:'CALL'|'PUT', side:'LONG'|'SHORT', K, premium}
      - include_stock: bool
      - stock_qty: float
    """
    legs = op.get("legs", [])
    include_stock = bool(op.get("include_stock", False))
    stock_qty = float(op.get("stock_qty", 1.0))

    total = np.zeros_like(ST, dtype=float)

    if include_stock:
        total += stock_qty * payoff_stock(ST, spot)

    for lg in legs:
        opt_type = lg["type"]
        side = lg["side"]
        K = float(lg["K"])
        prem = float(lg["premium"])

        if opt_type == "CALL" and side == "LONG":
            total += payoff_long_call(ST, K, prem)
        elif opt_type == "CALL" and side == "SHORT":
            total += payoff_short_call(ST, K, prem)
        elif opt_type == "PUT" and side == "LONG":
            total += payoff_long_put(ST, K, prem)
        elif opt_type == "PUT" and side == "SHORT":
            total += payoff_short_put(ST, K, prem)

    return total

def find_break_evens(ST: np.ndarray, pnl: np.ndarray, tol: float = 1e-8):
    """
    Find approximate break-even points (pnl==0) by sign changes + linear interpolation.
    """
    be = []
    y = pnl
    x = ST

    near = np.where(np.abs(y) < tol)[0]
    for i in near:
        be.append(float(x[i]))

    s = np.sign(y)
    for i in range(len(x) - 1):
        if s[i] == 0:
            continue
        if s[i] * s[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            xb = x0 - y0 * (x1 - x0) / (y1 - y0)
            be.append(float(xb))

    be = sorted(set([round(v, 6) for v in be]))
    return be

def payoff_summary(pnl: np.ndarray):
    return float(np.min(pnl)), float(np.max(pnl))

def plot_payoff(op: dict, spot: float, multiplier: float, show_pct: bool):
    s0 = float(spot) if (spot is not None and np.isfinite(spot) and spot > 0) else 1.0

    lo = max(0.01, 0.5 * s0)
    hi = 1.5 * s0
    ST = np.linspace(lo, hi, 700)

    pnl = payoff_strategy(op, ST, s0) * float(multiplier)

    denom = s0 * float(multiplier)
    pnl_plot = (pnl / denom) * 100.0 if show_pct else pnl

    be = find_break_evens(ST, pnl, tol=1e-6)
    mn, mx = payoff_summary(pnl)

    fig = plt.figure()
    plt.plot(ST, pnl_plot, linestyle="-")
    plt.axhline(0.0)
    plt.axvline(s0)
    for xbe in be:
        plt.axvline(xbe, linestyle="--")

    plt.xlabel("Preço do ativo no vencimento (ST)")
    plt.ylabel("Payoff (% do spot)" if show_pct else "Payoff (P&L no vencimento)")
    plt.title(op["label"])
    st.pyplot(fig, clear_figure=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Break-even(s)**")
        st.write(", ".join([f"{v:.2f}" for v in be]) if be else "—")
    with c2:
        st.write("**Perda máx (na faixa do gráfico)**")
        st.write(f"{mn:.2f}" + (f" ({(mn/denom*100):.2f}%)" if show_pct else ""))
    with c3:
        st.write("**Ganho máx (na faixa do gráfico)**")
        st.write(f"{mx:.2f}" + (f" ({(mx/denom*100):.2f}%)" if show_pct else ""))

    st.caption(
        f"Multiplicador: {multiplier:g} | "
        f"Payoff exibido em {'% do spot' if show_pct else 'valor monetário'} | "
        f"S0={s0:.2f}"
    )

# =========================
# Strategy selectors (Top3) + regime-aware criteria
# =========================

def _liquidity_score(df):
    t = pd.to_numeric(df["trades"], errors="coerce").fillna(0)
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return t + np.log1p(v)

def top_itm_buy_calls(df, expiry, regime: str, cfg: dict) -> pd.DataFrame:
    """
    Long deep ITM call:
    - In Alta: allow delta >= 0.75 (prefer 0.80+), prefer lower IV (cheaper), good liquidity
    - In Baixa: still allowed but stricter (delta >= 0.85) and prefer lower IV + higher mispricing_pct negative (cheap)
    - In Neutra: base delta >= 0.80
    """
    d = df[(df["option_type"]=="CALL") & (df["expiry_date"]==expiry)].copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d["mispricing_pct"] = pd.to_numeric(d.get("mispricing_pct", np.nan), errors="coerce")

    if regime == "Alta":
        dmin = cfg["deep_itm_delta_up"]
    elif regime == "Baixa":
        dmin = cfg["deep_itm_delta_down"]
    else:
        dmin = cfg["deep_itm_delta_neutral"]

    d = d[(d["moneyness"]=="ITM") & (d["delta"]>=dmin) & (d["iv"].notna()) & (d["last_price"]>0)].copy()
    if d.empty:
        return d

    d["liq"] = _liquidity_score(d)
    # prefer: liquidity high, IV low; if bearish, also prefer mispricing_pct more negative (cheaper vs BSM)
    if regime == "Baixa":
        d["cheapness"] = d["mispricing_pct"].fillna(0)  # negative is "cheap"
        return d.sort_values(["liq","cheapness","iv"], ascending=[False, True, True]).head(3)

    return d.sort_values(["liq","iv"], ascending=[False, True]).head(3)

def top_sell_puts(df, expiry, regime: str, cfg: dict) -> pd.DataFrame:
    """
    Short put:
    - Prefer in Alta/Neutra; in Baixa, either skip or require far OTM and smaller |delta|
    - Base: PUT OTM/ATM with delta in [put_delta_lo, put_delta_hi] (negative deltas)
    """
    d = df[(df["option_type"]=="PUT") & (df["expiry_date"]==expiry)].copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d["moneyness_dist"] = pd.to_numeric(d["moneyness_dist"], errors="coerce")

    lo, hi = cfg["put_delta_lo"], cfg["put_delta_hi"]  # negative values, e.g. -0.35..-0.10

    # If bearish, reduce risk: closer to -0.10 and farther OTM
    if regime == "Baixa":
        lo = cfg["put_delta_lo_bear"]     # e.g. -0.25
        hi = cfg["put_delta_hi_bear"]     # e.g. -0.08
        min_dist = cfg["put_min_otm_dist_bear"]  # e.g. 0.015
    else:
        min_dist = 0.0

    d = d[(d["moneyness"].isin(["OTM","ATM"])) & (d["delta"].between(lo, hi)) & (d["iv"].notna()) & (d["last_price"]>0)].copy()
    if min_dist > 0:
        d = d[(d["moneyness"]=="OTM") & (d["moneyness_dist"]>=min_dist)].copy()

    if d.empty:
        return d

    d["liq"] = _liquidity_score(d)
    # prefer: higher IV and higher premium (but still liquid)
    d["score"] = (d["iv"].fillna(0)*100.0) + (d["last_price"].fillna(0)*5.0) + d["liq"].fillna(0)
    return d.sort_values(["score"], ascending=[False]).head(3)

def top_covered_calls(df, expiry, regime: str, cfg: dict) -> pd.DataFrame:
    """
    Covered call (sell call):
    - In Alta: prefer OTM (avoid capping too much), delta 0.15..0.25
    - In Neutra: OTM/ATM, delta 0.20..0.30
    - In Baixa: can sell closer ATM (higher premium), delta 0.25..0.40 (but higher assignment risk)
    """
    d = df[(df["option_type"]=="CALL") & (df["expiry_date"]==expiry)].copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")

    if regime == "Alta":
        mset = ["OTM"]
        dlo, dhi = cfg["cc_delta_up"]
    elif regime == "Baixa":
        mset = ["OTM","ATM"]
        dlo, dhi = cfg["cc_delta_down"]
    else:
        mset = ["OTM","ATM"]
        dlo, dhi = cfg["cc_delta_neutral"]

    d = d[(d["moneyness"].isin(mset)) & (d["delta"].between(dlo, dhi)) & (d["iv"].notna()) & (d["last_price"]>0)].copy()
    if d.empty:
        return d

    d["liq"] = _liquidity_score(d)
    # prefer: higher premium and good liquidity (covered call is income)
    d["score"] = (d["last_price"].fillna(0)*10.0) + d["liq"].fillna(0)
    return d.sort_values(["score"], ascending=[False]).head(3)

def top_vertical_spreads(df, expiry, kind="bull_call", regime: str = "Neutra", cfg: dict = None) -> pd.DataFrame:
    """
    Vertical spreads (debit) – regime-aware preference:
    - If regime == Alta: show Bull Call first, filters are more permissive.
    - If regime == Baixa: show Bear Put first, filters are more permissive.
    - If regime == Neutra: both.
    """
    if cfg is None:
        cfg = {}

    d = df[df["expiry_date"]==expiry].copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d = d.dropna(subset=["strike","last_price"]).copy()
    d = d[d["last_price"] > 0].copy()

    rows = []
    if kind == "bull_call":
        calls = d[d["option_type"]=="CALL"].copy()
        buy = calls[calls["moneyness"]=="ATM"].copy()
        sell = calls[calls["moneyness"]=="OTM"].copy()

        # In Alta, allow ATM buy; in Baixa, keep stricter to avoid fighting trend
        if regime == "Baixa":
            # demand MACD/RSI etc handled outside; here just reduce candidates by delta if available
            buy["delta"] = pd.to_numeric(buy.get("delta", np.nan), errors="coerce")
            buy = buy[(buy["delta"].isna()) | (buy["delta"] >= 0.45)].copy()

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
                rows.append({
                    "strategy":"Bull Call Spread",
                    "buy": b["option_symbol"], "sell": s["option_symbol"],
                    "K_buy": float(b["strike"]), "K_sell": float(s["strike"]),
                    "P_buy": float(b["last_price"]), "P_sell": float(s["last_price"]),
                    "debit": debit, "max_profit": max_profit, "rr": rr
                })
    else:
        puts = d[d["option_type"]=="PUT"].copy()
        buy = puts[puts["moneyness"]=="ATM"].copy()
        sell = puts[puts["moneyness"]=="OTM"].copy()

        if regime == "Alta":
            buy["delta"] = pd.to_numeric(buy.get("delta", np.nan), errors="coerce")
            buy = buy[(buy["delta"].isna()) | (buy["delta"] <= -0.45)].copy()

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
                rows.append({
                    "strategy":"Bear Put Spread",
                    "buy": b["option_symbol"], "sell": s["option_symbol"],
                    "K_buy": float(b["strike"]), "K_sell": float(s["strike"]),
                    "P_buy": float(b["last_price"]), "P_sell": float(s["last_price"]),
                    "debit": debit, "max_profit": max_profit, "rr": rr
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["rr","max_profit"], ascending=[False, False]).head(3)

def top_straddles(df, expiry, regime: str, cfg: dict) -> pd.DataFrame:
    """
    Long Straddle:
    - Works best with expectation of high realized move; we rank by liquidity and lower total premium.
    - Regime-aware note: in Neutra, emphasize straddle (range breakout).
    """
    d = df[df["expiry_date"]==expiry].copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d = d.dropna(subset=["strike","last_price"]).copy()
    d = d[d["last_price"] > 0].copy()

    calls = d[(d["option_type"]=="CALL") & (d["moneyness"]=="ATM")].copy()
    puts  = d[(d["option_type"]=="PUT")  & (d["moneyness"]=="ATM")].copy()

    if calls.empty or puts.empty:
        return pd.DataFrame()

    rows = []
    for _, c in calls.iterrows():
        p_cand = puts.copy()
        p_cand["dK"] = (p_cand["strike"] - c["strike"]).abs()
        p = p_cand.sort_values(["dK","trades","volume"], ascending=[True, False, False]).head(1)
        if p.empty:
            continue
        p = p.iloc[0]
        prem = float(c["last_price"]) + float(p["last_price"])
        liq = float(_liquidity_score(pd.DataFrame([c])).iloc[0]) + float(_liquidity_score(pd.DataFrame([p])).iloc[0])
        rows.append({
            "strategy": "Long Straddle (ATM)",
            "call": c["option_symbol"],
            "put": p["option_symbol"],
            "K_call": float(c["strike"]),
            "K_put": float(p["strike"]),
            "P_call": float(c["last_price"]),
            "P_put": float(p["last_price"]),
            "premium_total": prem,
            "liq": liq
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["liq","premium_total"], ascending=[False, True]).head(3)
    return out

# ============================================================
# SHORT CONDOR (PURE) — Calls and Puts (NOT iron condor)
# ============================================================

def _pick_nearest_strike_row(d: pd.DataFrame, targetK: float, avoid_symbols: set):
    tmp = d.copy()
    tmp["dist"] = (tmp["strike"] - targetK).abs()
    tmp = tmp[~tmp["option_symbol"].isin(avoid_symbols)]
    tmp = tmp.sort_values(["dist","trades","volume"], ascending=[True, False, False])
    if tmp.empty:
        return None
    return tmp.iloc[0]

def top_short_call_condors(df: pd.DataFrame, expiry: date, regime: str, cfg: dict) -> pd.DataFrame:
    """
    Short Call Condor (credit), pure CALLs:
      SELL K1, BUY K2, BUY K3, SELL K4   (K1 < K2 < K3 < K4)

    Regime:
      - In Alta: avoid short call condor (can fight upside). If allow, push strikes farther OTM.
      - In Baixa/Neutra: ok.
    """
    d = df[(df["expiry_date"]==expiry) & (df["option_type"]=="CALL")].copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["trades"] = pd.to_numeric(d["trades"], errors="coerce").fillna(0)
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d = d.dropna(subset=["strike","last_price","delta"]).copy()
    d = d[d["last_price"] > 0].sort_values("strike")
    if d.empty:
        return pd.DataFrame()

    # adjust delta bands per regime
    if regime == "Alta":
        sell_band = cfg["condor_call_sell_band_up"]   # e.g. (0.12, 0.22)
        buy_band  = cfg["condor_call_buy_band_up"]    # e.g. (0.04, 0.10)
    else:
        sell_band = cfg["condor_call_sell_band"]
        buy_band  = cfg["condor_call_buy_band"]

    sells = d[d["delta"].between(*sell_band)].copy()
    buys  = d[d["delta"].between(*buy_band)].copy()
    if sells.empty or buys.empty:
        return pd.DataFrame()

    sells = sells.sort_values("strike").head(12)
    rows = []

    for i in range(min(len(sells), 8)):
        for j in range(i+1, min(len(sells), 10)):
            s1 = sells.iloc[i]
            s4 = sells.iloc[j]
            K1, K4 = float(s1["strike"]), float(s4["strike"])
            if K4 <= K1:
                continue

            b_in = buys[(buys["strike"] > K1) & (buys["strike"] < K4)].copy()
            if b_in.empty:
                continue

            gap = max((K4-K1)/3.0, 0.01)
            targetK2 = K1 + gap
            targetK3 = K4 - gap

            avoid = {s1["option_symbol"], s4["option_symbol"]}
            b2 = _pick_nearest_strike_row(b_in, targetK2, avoid)
            if b2 is None:
                continue
            avoid.add(b2["option_symbol"])
            b3 = _pick_nearest_strike_row(b_in, targetK3, avoid)
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

            rr = credit / max_loss_est
            liq = float(
                _liquidity_score(pd.DataFrame([s1])).iloc[0]
                + _liquidity_score(pd.DataFrame([b2])).iloc[0]
                + _liquidity_score(pd.DataFrame([b3])).iloc[0]
                + _liquidity_score(pd.DataFrame([s4])).iloc[0]
            )

            rows.append({
                "strategy":"Short Call Condor (Credit)",
                "sell_1": s1["option_symbol"],
                "buy_2": b2["option_symbol"],
                "buy_3": b3["option_symbol"],
                "sell_4": s4["option_symbol"],
                "K1": K1, "K2": K2, "K3": K3, "K4": K4,
                "P_sell_1": P_s1, "P_buy_2": P_b2, "P_buy_3": P_b3, "P_sell_4": P_s4,
                "credit": credit,
                "max_loss_est": max_loss_est,
                "cr": rr,
                "liq": liq
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["cr","credit","liq"], ascending=[False, False, False]).head(3)
    return out

def top_short_put_condors(df: pd.DataFrame, expiry: date, regime: str, cfg: dict) -> pd.DataFrame:
    """
    Short Put Condor (credit), pure PUTs:
      SELL K1, BUY K2, BUY K3, SELL K4   (K1 < K2 < K3 < K4)

    Regime:
      - In Baixa: avoid being too aggressive; push strikes farther OTM (smaller |delta|)
      - In Alta/Neutra: base bands.
    """
    d = df[(df["expiry_date"]==expiry) & (df["option_type"]=="PUT")].copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["trades"] = pd.to_numeric(d["trades"], errors="coerce").fillna(0)
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    d = d.dropna(subset=["strike","last_price","delta"]).copy()
    d = d[d["last_price"] > 0].sort_values("strike")
    if d.empty:
        return pd.DataFrame()

    if regime == "Baixa":
        sell_band = cfg["condor_put_sell_band_down"]  # e.g. (-0.22, -0.12)
        buy_band  = cfg["condor_put_buy_band_down"]   # e.g. (-0.10, -0.04)
    else:
        sell_band = cfg["condor_put_sell_band"]
        buy_band  = cfg["condor_put_buy_band"]

    sells = d[d["delta"].between(*sell_band)].copy()
    buys  = d[d["delta"].between(*buy_band)].copy()
    if sells.empty or buys.empty:
        return pd.DataFrame()

    sells = sells.sort_values("strike").head(12)
    rows = []

    for i in range(min(len(sells), 8)):
        for j in range(i+1, min(len(sells), 10)):
            s1 = sells.iloc[i]
            s4 = sells.iloc[j]
            K1, K4 = float(s1["strike"]), float(s4["strike"])
            if K4 <= K1:
                continue

            b_in = buys[(buys["strike"] > K1) & (buys["strike"] < K4)].copy()
            if b_in.empty:
                continue

            gap = max((K4-K1)/3.0, 0.01)
            targetK2 = K1 + gap
            targetK3 = K4 - gap

            avoid = {s1["option_symbol"], s4["option_symbol"]}
            b2 = _pick_nearest_strike_row(b_in, targetK2, avoid)
            if b2 is None:
                continue
            avoid.add(b2["option_symbol"])
            b3 = _pick_nearest_strike_row(b_in, targetK3, avoid)
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

            rr = credit / max_loss_est
            liq = float(
                _liquidity_score(pd.DataFrame([s1])).iloc[0]
                + _liquidity_score(pd.DataFrame([b2])).iloc[0]
                + _liquidity_score(pd.DataFrame([b3])).iloc[0]
                + _liquidity_score(pd.DataFrame([s4])).iloc[0]
            )

            rows.append({
                "strategy":"Short Put Condor (Credit)",
                "sell_1": s1["option_symbol"],
                "buy_2": b2["option_symbol"],
                "buy_3": b3["option_symbol"],
                "sell_4": s4["option_symbol"],
                "K1": K1, "K2": K2, "K3": K3, "K4": K4,
                "P_sell_1": P_s1, "P_buy_2": P_b2, "P_buy_3": P_b3, "P_sell_4": P_s4,
                "credit": credit,
                "max_loss_est": max_loss_est,
                "cr": rr,
                "liq": liq
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["cr","credit","liq"], ascending=[False, False, False]).head(3)
    return out

# =========================
# UI selection helper
# =========================

def selectable_table(df: pd.DataFrame, key: str, label: str):
    """
    Try clickable selection in st.dataframe; fallback to selectbox if unsupported.
    Returns selected row index (int) or None.
    """
    if df is None or df.empty:
        st.info("Sem operações candidatas para esta estratégia.")
        return None

    disp = df.copy().reset_index(drop=True)
    st.caption(label)

    try:
        evt = st.dataframe(
            disp,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"{key}_df",
        )
        if evt and getattr(evt, "selection", None):
            rows = evt.selection.get("rows", [])
            if rows:
                return int(rows[0])
        return None
    except TypeError:
        options = [f"{i}: {disp.iloc[i].to_dict()}" for i in range(len(disp))]
        pick = st.selectbox("Escolha a operação para ver o payoff:", ["(nenhuma)"] + options, key=f"{key}_sb")
        if pick == "(nenhuma)":
            return None
        return int(pick.split(":")[0])

def plot_smile(df: pd.DataFrame, expiry_date: date, option_type: str, spot: float):
    d = df[(df["expiry_date"] == expiry_date) & (df["option_type"].str.upper() == option_type.upper())].copy()
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d = d.dropna(subset=["iv","strike"]).sort_values("strike")
    if d.empty:
        st.info(f"Sem IV para {option_type} no vencimento {expiry_date}.")
        return
    fig = plt.figure()
    plt.plot(d["strike"], d["iv"], marker="o", linestyle="-")
    if spot is not None and np.isfinite(spot):
        plt.axvline(x=float(spot))
    plt.xlabel("Strike")
    plt.ylabel("IV")
    plt.title(f"Sorriso de Vol – {option_type} – {expiry_date}")
    st.pyplot(fig, clear_figure=True)

# =========================
# Criteria display helpers
# =========================

def criteria_box(title: str, bullets: list[str], notes: list[str] | None = None):
    st.markdown(f"**Critérios – {title}:**")
    st.markdown("\n".join([f"- {b}" for b in bullets]))
    if notes:
        st.markdown("\n".join([f"> {n}" for n in notes]))

# =========================
# App
# =========================

st.title("Opções – Última Coleta (Greeks + BSM/IV + Estratégias + Payoff)")

assets = load_assets()
if assets.empty:
    st.warning("Nenhum ativo em assets. Rode o pipeline primeiro.")
    st.stop()

# Toggles payoff
st.sidebar.markdown("## Payoff (config)")
show_pct = st.sidebar.toggle("Exibir payoff em % do spot", value=False)
mult_100 = st.sidebar.toggle("Payoff por 100 ações (multiplicador=100)", value=True)
multiplier = 100.0 if mult_100 else 1.0

# Trend thresholds
st.sidebar.markdown("## Tendência (daily_bars)")
rsi_hi = st.sidebar.slider("RSI alto (força)", 50, 70, 55, 1)
rsi_lo = st.sidebar.slider("RSI baixo (fraqueza)", 30, 50, 45, 1)

ticker = st.sidebar.selectbox("Ativo", assets["ticker"].tolist())
asset_id = int(assets.loc[assets["ticker"]==ticker, "id"].iloc[0])

trade_date = load_latest_trade_date(asset_id)
if not trade_date:
    st.warning("Sem dados em option_quote.")
    st.stop()

ind = load_daily_indicators(asset_id, trade_date)
regime, up_score, down_score, reg_details = infer_regime(ind, rsi_hi=float(rsi_hi), rsi_lo=float(rsi_lo))

spot = None
hist_vol_annual = None
if ind:
    spot = _to_float(ind.get("close"))
    hist_vol_annual = _to_float(ind.get("vol_annual"))

df = load_chain(asset_id, trade_date)
if df.empty:
    st.warning("Sem opções com trades>0 e last_price>0.")
    st.stop()

expiry_list = sorted(df["expiry_date"].dropna().unique().tolist())
expiry_choice = st.sidebar.selectbox("Vencimento", [str(x) for x in expiry_list])
expiry_sel = pd.to_datetime(expiry_choice).date()

atm_mode_ui = st.sidebar.radio("ATM", options=["Faixa percentual","Mais próximo"], index=0)
atm_mode = "pct" if atm_mode_ui=="Faixa percentual" else "nearest"
atm_pct = st.sidebar.slider("Faixa ATM (|K−S|/S)", 0.001, 0.05, 0.01, 0.001, disabled=(atm_mode!="pct"))

df2 = df[df["expiry_date"]==expiry_sel].copy()

# spot fallback if daily_bars not available
if spot is None or not np.isfinite(spot):
    s2 = pd.to_numeric(df2["spot"], errors="coerce").dropna()
    spot = float(s2.iloc[0]) if len(s2) else None

df2 = classify_moneyness(df2, spot=spot if spot else np.nan, atm_mode=atm_mode, atm_pct=atm_pct)

quote_ts = pd.to_datetime(df2["quote_collected_at"], errors="coerce").max()
model_ts = pd.to_datetime(df2["model_collected_at"], errors="coerce").max()

df2["rate_r"] = pd.to_numeric(df2["rate_r"], errors="coerce")
r_expiry = float(df2["rate_r"].dropna().median()) if df2["rate_r"].notna().any() else np.nan

# Header metrics
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Ticker", ticker)
c2.metric("Pregão", str(trade_date))
c3.metric("Spot (close)", f"{spot:.4f}" if spot is not None else "N/A")
c4.metric("Hist vol anual", f"{hist_vol_annual:.4f}" if hist_vol_annual is not None else "N/A")
c5.metric("Vencimento", str(expiry_sel))
c6.metric("r (curva via model)", f"{r_expiry:.4%}" if np.isfinite(r_expiry) else "N/A")
c7.metric("Regime", regime)

st.caption(f"Coleta quotes: {quote_ts} | Coleta model: {model_ts}")

# Regime + indicators panel
with st.expander("Tendência / Indicadores (daily_bars) usados nas estratégias", expanded=True):
    if not ind:
        st.info("Sem daily_bars para este pregão — o regime fica N/A e as estratégias usam critérios padrão.")
    else:
        st.markdown(
            f"""
**Regime inferido:** **{regime}** (up_score={up_score}, down_score={down_score})

**Regras do regime (heurística):**
- SMA: *stack* (SMA20/SMA50/SMA200) e close vs SMA50
- MACD_hist: sinal (positivo/negativo)
- RSI14: força (≥{rsi_hi}) / fraqueza (≤{rsi_lo})

**Detalhes:**
- SMA_stack: {reg_details.get("SMA_stack")}
- Close_vs_SMA50: {reg_details.get("Close_vs_SMA50")}
- MACD_hist: {reg_details.get("MACD_hist")}
- RSI: {reg_details.get("RSI")}
            """.strip()
        )
        cols = ["trade_date","close","vol_annual","sma_20","sma_50","sma_200","macd_hist","rsi_14","atr_14"]
        st.dataframe(pd.DataFrame([{k: ind.get(k) for k in cols}]), width="stretch", hide_index=True)

# =========================
# Strategy configuration (regime-aware)
# =========================
cfg = {
    # Long deep ITM call deltas
    "deep_itm_delta_up": 0.75,
    "deep_itm_delta_neutral": 0.80,
    "deep_itm_delta_down": 0.85,

    # Short put deltas
    "put_delta_lo": -0.35,
    "put_delta_hi": -0.10,
    # bear regime: less aggressive
    "put_delta_lo_bear": -0.25,
    "put_delta_hi_bear": -0.08,
    "put_min_otm_dist_bear": 0.015,

    # Covered call deltas
    "cc_delta_up": (0.12, 0.25),
    "cc_delta_neutral": (0.15, 0.30),
    "cc_delta_down": (0.20, 0.40),

    # Condor delta bands
    "condor_call_sell_band": (0.18, 0.32),
    "condor_call_buy_band": (0.06, 0.16),
    "condor_call_sell_band_up": (0.12, 0.22),   # farther OTM if uptrend
    "condor_call_buy_band_up": (0.04, 0.10),

    "condor_put_sell_band": (-0.32, -0.18),
    "condor_put_buy_band": (-0.16, -0.06),
    "condor_put_sell_band_down": (-0.22, -0.12),  # farther OTM if downtrend
    "condor_put_buy_band_down": (-0.10, -0.04),
}

# =========================
# Strategies + Payoff on select
# =========================

st.subheader("Top 3 por estratégia (seleção mostra payoff com break-even)")

# Strategy criteria text (regime-aware)
reg_note = {
    "Alta": "Regime de alta: prioriza estruturas *bullish* e venda de calls mais OTM (evita limitar upside).",
    "Baixa": "Regime de baixa: reduz agressividade em venda de puts e prioriza estruturas *bearish*.",
    "Neutra": "Regime neutro: prioriza renda/estruturas com melhor RR e liquidez.",
    "N/A": "Sem daily_bars: usa critérios padrão (sem ajuste por tendência)."
}.get(regime, "—")

st.info(reg_note)

colA, colB = st.columns(2)

with colA:
    st.markdown("### 1) Compra bem ITM (CALL deep ITM)")
    if regime == "Alta":
        criteria_box(
            "Compra ITM",
            [
                f"CALL ITM (moneyness=ITM)",
                f"delta ≥ {cfg['deep_itm_delta_up']:.2f}",
                "trades>0 e last_price>0 (já filtrado na query)",
                "ranking: maior liquidez (trades+log(1+volume)) e menor IV",
            ],
            ["Em alta, deep ITM dá exposição com menor vega relativa e menos ruído de IV."]
        )
    elif regime == "Baixa":
        criteria_box(
            "Compra ITM",
            [
                f"CALL ITM (moneyness=ITM)",
                f"delta ≥ {cfg['deep_itm_delta_down']:.2f}",
                "ranking: maior liquidez e preferência por mispricing_pct mais negativo (mais 'barata' vs BSM)",
            ],
            ["Em baixa, call deep ITM é mais defensiva; ainda assim, pode estar contra tendência."]
        )
    else:
        criteria_box(
            "Compra ITM",
            [
                f"CALL ITM (moneyness=ITM)",
                f"delta ≥ {cfg['deep_itm_delta_neutral']:.2f}",
                "ranking: maior liquidez e menor IV",
            ]
        )

    t1 = top_itm_buy_calls(df2, expiry_sel, regime=regime, cfg=cfg)
    idx = selectable_table(format_table(t1), key="t1", label="Selecione uma operação para ver o payoff.")
    if idx is not None and not t1.empty:
        row = t1.iloc[idx]
        op = {
            "label": f"Compra ITM – LONG CALL {row['option_symbol']} (K={row['strike']}, prémio={row['last_price']})",
            "legs": [{"type":"CALL","side":"LONG","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": False,
        }
        plot_payoff(op, spot, multiplier, show_pct)

    st.markdown("### 2) Venda de PUTs (cash-secured / renda)")
    if regime == "Baixa":
        criteria_box(
            "Venda de PUT",
            [
                f"PUT OTM (preferência), delta em [{cfg['put_delta_lo_bear']:.2f}, {cfg['put_delta_hi_bear']:.2f}]",
                f"moneyness_dist ≥ {cfg['put_min_otm_dist_bear']:.3f} (mais longe do spot)",
                "ranking: maior IV, maior prêmio e liquidez",
            ],
            ["Em baixa, vender put perto do dinheiro aumenta risco; por isso o filtro fica mais conservador."]
        )
    else:
        criteria_box(
            "Venda de PUT",
            [
                f"PUT OTM/ATM, delta em [{cfg['put_delta_lo']:.2f}, {cfg['put_delta_hi']:.2f}]",
                "ranking: maior IV, maior prêmio e liquidez",
            ],
            ["Em alta/neutra, venda de put tende a ter melhor assimetria (se você aceita comprar o ativo)."]
        )

    t2 = top_sell_puts(df2, expiry_sel, regime=regime, cfg=cfg)
    idx2 = selectable_table(format_table(t2), key="t2", label="Selecione uma operação para ver o payoff.")
    if idx2 is not None and not t2.empty:
        row = t2.iloc[idx2]
        op = {
            "label": f"Venda de PUT – SHORT PUT {row['option_symbol']} (K={row['strike']}, prémio={row['last_price']})",
            "legs": [{"type":"PUT","side":"SHORT","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": False,
        }
        plot_payoff(op, spot, multiplier, show_pct)

    st.markdown("### 5) Long Straddle (ATM) – volatilidade / rompimento")
    criteria_box(
        "Long Straddle",
        [
            "BUY CALL ATM + BUY PUT ATM (mesmo vencimento)",
            "ranking: maior liquidez e menor prêmio total",
        ],
        [
            "Em regime neutro, straddle pode capturar rompimentos; em alta/baixa, avalie se prefere direcional (travas)."
        ]
    )
    tS = top_straddles(df2, expiry_sel, regime=regime, cfg=cfg)
    idxS = selectable_table(tS, key="straddle", label="Selecione um straddle para ver o payoff.")
    if idxS is not None and not tS.empty:
        r = tS.iloc[idxS]
        op = {
            "label": (
                f"Long Straddle – BUY {r['call']} (K={r['K_call']},P={r['P_call']}) + "
                f"BUY {r['put']} (K={r['K_put']},P={r['P_put']})"
            ),
            "legs": [
                {"type":"CALL","side":"LONG","K":float(r["K_call"]),"premium":float(r["P_call"])},
                {"type":"PUT","side":"LONG","K":float(r["K_put"]),"premium":float(r["P_put"])},
            ],
            "include_stock": False,
        }
        plot_payoff(op, spot, multiplier, show_pct)

with colB:
    st.markdown("### 3) Travas (Bull Call / Bear Put) – Top RR")
    if regime == "Alta":
        criteria_box(
            "Travas",
            [
                "Prioridade: Bull Call (debit) em alta",
                "Bull: BUY CALL ATM, SELL CALL OTM",
                "Bear: BUY PUT ATM, SELL PUT OTM (mostrado, mas tende a ser contra tendência)",
                "ranking: melhor RR (max_profit/debit) e maior max_profit",
            ]
        )
    elif regime == "Baixa":
        criteria_box(
            "Travas",
            [
                "Prioridade: Bear Put (debit) em baixa",
                "Bear: BUY PUT ATM, SELL PUT OTM",
                "Bull: BUY CALL ATM, SELL CALL OTM (mostrado, mas tende a ser contra tendência)",
                "ranking: melhor RR (max_profit/debit) e maior max_profit",
            ]
        )
    else:
        criteria_box(
            "Travas",
            [
                "Bull: BUY CALL ATM, SELL CALL OTM",
                "Bear: BUY PUT ATM, SELL PUT OTM",
                "ranking: melhor RR (max_profit/debit) e maior max_profit",
            ]
        )

    bull = top_vertical_spreads(df2, expiry_sel, kind="bull_call", regime=regime, cfg=cfg)
    bear = top_vertical_spreads(df2, expiry_sel, kind="bear_put", regime=regime, cfg=cfg)

    # Order tabs according to regime
    if regime == "Baixa":
        tab_first, tab_second = "Bear Put Spread", "Bull Call Spread"
    else:
        tab_first, tab_second = "Bull Call Spread", "Bear Put Spread"

    tabs = st.tabs([tab_first, tab_second])

    def _render_spread(tab, name, data):
        with tab:
            idx_sp = selectable_table(data, key=f"sp_{name}", label="Selecione uma trava para ver o payoff.")
            if idx_sp is not None and not data.empty:
                r = data.iloc[idx_sp]
                if "Bull" in r["strategy"]:
                    op = {
                        "label": f"Bull Call Spread – BUY {r['buy']} | SELL {r['sell']} (debit={r['debit']:.2f})",
                        "legs":[
                            {"type":"CALL","side":"LONG","K":float(r["K_buy"]),"premium":float(r["P_buy"])},
                            {"type":"CALL","side":"SHORT","K":float(r["K_sell"]),"premium":float(r["P_sell"])},
                        ],
                        "include_stock": False,
                    }
                else:
                    op = {
                        "label": f"Bear Put Spread – BUY {r['buy']} | SELL {r['sell']} (debit={r['debit']:.2f})",
                        "legs":[
                            {"type":"PUT","side":"LONG","K":float(r["K_buy"]),"premium":float(r["P_buy"])},
                            {"type":"PUT","side":"SHORT","K":float(r["K_sell"]),"premium":float(r["P_sell"])},
                        ],
                        "include_stock": False,
                    }
                plot_payoff(op, spot, multiplier, show_pct)

    if tab_first == "Bull Call Spread":
        _render_spread(tabs[0], "bull", bull)
        _render_spread(tabs[1], "bear", bear)
    else:
        _render_spread(tabs[0], "bear", bear)
        _render_spread(tabs[1], "bull", bull)

    st.markdown("### 4) Venda coberta (1 ação + short CALL)")
    if regime == "Alta":
        criteria_box(
            "Venda coberta",
            [
                "Preferir CALL OTM (evita limitar upside em tendência de alta)",
                f"delta em [{cfg['cc_delta_up'][0]:.2f}, {cfg['cc_delta_up'][1]:.2f}]",
                "ranking: maior prêmio e liquidez",
            ]
        )
    elif regime == "Baixa":
        criteria_box(
            "Venda coberta",
            [
                "Permite OTM/ATM (melhor prêmio em baixa, mas maior risco de ficar travado)",
                f"delta em [{cfg['cc_delta_down'][0]:.2f}, {cfg['cc_delta_down'][1]:.2f}]",
                "ranking: maior prêmio e liquidez",
            ]
        )
    else:
        criteria_box(
            "Venda coberta",
            [
                "OTM/ATM",
                f"delta em [{cfg['cc_delta_neutral'][0]:.2f}, {cfg['cc_delta_neutral'][1]:.2f}]",
                "ranking: maior prêmio e liquidez",
            ]
        )

    t4 = top_covered_calls(df2, expiry_sel, regime=regime, cfg=cfg)
    idx4 = selectable_table(format_table(t4), key="t4", label="Selecione uma operação para ver o payoff.")
    if idx4 is not None and not t4.empty:
        row = t4.iloc[idx4]
        op = {
            "label": f"Venda Coberta – STOCK + SHORT CALL {row['option_symbol']} (K={row['strike']}, prémio={row['last_price']})",
            "legs": [{"type":"CALL","side":"SHORT","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": True,
            "stock_qty": 1.0
        }
        plot_payoff(op, spot, multiplier, show_pct)

    st.markdown("### 6) Short Condor Puro (CALLs / PUTs) – Top 3 (crédito)")
    criteria_box(
        "Short Condor (puro)",
        [
            "Condor de crédito: SELL K1, BUY K2, BUY K3, SELL K4 (todas CALLs ou todas PUTs)",
            "Seleção via bandas de delta (sells ~0.25 / buys ~0.10) ajustadas por regime",
            "ranking: maior credit-to-risk (cr) e maior crédito com liquidez",
        ],
        [
            "Em alta, CALL-condor é mais conservador (deltas menores => strikes mais OTM).",
            "Em baixa, PUT-condor é mais conservador (|delta| menor => strikes mais OTM)."
        ]
    )

    tab1, tab2 = st.tabs(["Short Call Condor", "Short Put Condor"])

    with tab1:
        tCC = top_short_call_condors(df2, expiry_sel, regime=regime, cfg=cfg)
        idxCC = selectable_table(tCC, key="short_call_condor", label="Selecione um short call condor para ver o payoff.")
        if idxCC is not None and not tCC.empty:
            r = tCC.iloc[idxCC]
            op = {
                "label": (
                    f"Short Call Condor – "
                    f"SELL {r['sell_1']} (K1={r['K1']},P={r['P_sell_1']}) | "
                    f"BUY {r['buy_2']} (K2={r['K2']},P={r['P_buy_2']}) | "
                    f"BUY {r['buy_3']} (K3={r['K3']},P={r['P_buy_3']}) | "
                    f"SELL {r['sell_4']} (K4={r['K4']},P={r['P_sell_4']}) | "
                    f"credit={r['credit']:.2f}"
                ),
                "legs": [
                    {"type":"CALL","side":"SHORT","K":float(r["K1"]),"premium":float(r["P_sell_1"])},
                    {"type":"CALL","side":"LONG","K":float(r["K2"]),"premium":float(r["P_buy_2"])},
                    {"type":"CALL","side":"LONG","K":float(r["K3"]),"premium":float(r["P_buy_3"])},
                    {"type":"CALL","side":"SHORT","K":float(r["K4"]),"premium":float(r["P_sell_4"])},
                ],
                "include_stock": False
            }
            plot_payoff(op, spot, multiplier, show_pct)

    with tab2:
        tPC = top_short_put_condors(df2, expiry_sel, regime=regime, cfg=cfg)
        idxPC = selectable_table(tPC, key="short_put_condor", label="Selecione um short put condor para ver o payoff.")
        if idxPC is not None and not tPC.empty:
            r = tPC.iloc[idxPC]
            op = {
                "label": (
                    f"Short Put Condor – "
                    f"SELL {r['sell_1']} (K1={r['K1']},P={r['P_sell_1']}) | "
                    f"BUY {r['buy_2']} (K2={r['K2']},P={r['P_buy_2']}) | "
                    f"BUY {r['buy_3']} (K3={r['K3']},P={r['P_buy_3']}) | "
                    f"SELL {r['sell_4']} (K4={r['K4']},P={r['P_sell_4']}) | "
                    f"credit={r['credit']:.2f}"
                ),
                "legs": [
                    {"type":"PUT","side":"SHORT","K":float(r["K1"]),"premium":float(r["P_sell_1"])},
                    {"type":"PUT","side":"LONG","K":float(r["K2"]),"premium":float(r["P_buy_2"])},
                    {"type":"PUT","side":"LONG","K":float(r["K3"]),"premium":float(r["P_buy_3"])},
                    {"type":"PUT","side":"SHORT","K":float(r["K4"]),"premium":float(r["P_sell_4"])},
                ],
                "include_stock": False
            }
            plot_payoff(op, spot, multiplier, show_pct)

# =========================
# Table + smiles
# =========================

st.subheader("Tabela completa (filtros: trades>0 e last_price>0)")
st.dataframe(format_table(df2), width="stretch", height=520, hide_index=True)

st.subheader("Sorriso de Volatilidade (IV vs Strike)")
cL, cR = st.columns(2)
with cL:
    plot_smile(df2, expiry_sel, "CALL", spot)
with cR:
    plot_smile(df2, expiry_sel, "PUT", spot)
