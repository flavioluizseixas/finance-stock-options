# app_streamlit_options.py
# ------------------------------------------------------------
# Streamlit dashboard for options (READ-ONLY) using Parquet files:
# - ITM/ATM/OTM tables, greeks, BSM/IV, last collection only
# - Strategy picker (Top 3) with payoff chart shown ONLY when user selects an operation.
# - Uses technical indicators from daily_bars to infer "trend/regime"
#
# Parquet expected tables (files):
# - data/assets.parquet
# - data/daily_bars.parquet
# - data/option_quote.parquet
# - data/option_model.parquet
#
# Expected columns (same as your DB schema):
# assets: id, ticker, is_active
# daily_bars: asset_id, trade_date, close, vol_annual,
#             sma_20, sma_50, sma_200, macd, macd_signal, macd_hist, rsi_14, atr_14
# option_quote: asset_id, trade_date, option_symbol, option_type, expiry_date,
#               strike, last_price, trades, volume, collected_at
# option_model: asset_id, trade_date, option_symbol, spot, rate_r, dividend_q, t_years,
#               iv, bsm_price, bsm_price_histvol, mispricing, mispricing_pct,
#               delta, gamma, vega, theta, rho, hist_vol_annual, collected_at
# ------------------------------------------------------------

import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finance Options Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

# Where parquet files live
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))

PATH_ASSETS = os.getenv("PATH_ASSETS", os.path.join(DATA_DIR, "assets.parquet"))
PATH_DAILY  = os.getenv("PATH_DAILY",  os.path.join(DATA_DIR, "daily_bars.parquet"))
PATH_QUOTE  = os.getenv("PATH_QUOTE",  os.path.join(DATA_DIR, "option_quote.parquet"))
PATH_MODEL  = os.getenv("PATH_MODEL",  os.path.join(DATA_DIR, "option_model.parquet"))

# =========================
# Parquet helpers (cache)
# =========================

def _require_file(path: str):
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        st.stop()

def _coerce_dates(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_assets():
    _require_file(PATH_ASSETS)
    df = pd.read_parquet(PATH_ASSETS)
    # normalize
    if "is_active" in df.columns:
        df = df[df["is_active"].astype(int) == 1]
    df = df[["id", "ticker"]].dropna().sort_values("ticker")
    df["id"] = df["id"].astype(int)
    return df.reset_index(drop=True)

@st.cache_data(ttl=300, show_spinner=False)
def load_option_quote_all():
    _require_file(PATH_QUOTE)
    df = pd.read_parquet(PATH_QUOTE)
    df = _coerce_dates(df, ["trade_date", "expiry_date"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_option_model_all():
    _require_file(PATH_MODEL)
    df = pd.read_parquet(PATH_MODEL)
    df = _coerce_dates(df, ["trade_date"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_daily_bars_all():
    _require_file(PATH_DAILY)
    df = pd.read_parquet(PATH_DAILY)
    df = _coerce_dates(df, ["trade_date"])
    return df

@st.cache_data(ttl=60, show_spinner=False)
def load_latest_trade_date(asset_id: int):
    oq = load_option_quote_all()
    d = oq.loc[oq["asset_id"].astype(int) == int(asset_id), "trade_date"]
    d = d.dropna()
    return max(d) if len(d) else None

@st.cache_data(ttl=60, show_spinner=False)
def load_daily_indicators(asset_id: int, trade_date: date):
    db = load_daily_bars_all()
    m = (db["asset_id"].astype(int) == int(asset_id)) & (db["trade_date"] == trade_date)
    d = db.loc[m].copy()
    if d.empty:
        return None
    # take first (should be unique per asset/date)
    row = d.iloc[0].to_dict()
    return row

@st.cache_data(ttl=60, show_spinner=False)
def load_chain(asset_id: int, trade_date: date):
    """
    Load chain and left join model (like your SQL).
    Filter: trades>0 and last_price>0.
    """
    oq = load_option_quote_all()
    om = load_option_model_all()

    # filter quote
    m = (oq["asset_id"].astype(int) == int(asset_id)) & (oq["trade_date"] == trade_date)
    q = oq.loc[m].copy()

    if q.empty:
        return pd.DataFrame()

    # base filters
    for c in ["trades", "last_price"]:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q[(q["trades"] > 0) & (q["last_price"] > 0)].copy()
    if q.empty:
        return pd.DataFrame()

    # join model
    m2 = (om["asset_id"].astype(int) == int(asset_id)) & (om["trade_date"] == trade_date)
    mm = om.loc[m2].copy()

    # disambiguate collected_at columns (match your output)
    if "collected_at" in q.columns:
        q = q.rename(columns={"collected_at": "quote_collected_at"})
    if "collected_at" in mm.columns:
        mm = mm.rename(columns={"collected_at": "model_collected_at"})

    # merge
    out = q.merge(
        mm,
        how="left",
        on=["asset_id", "trade_date", "option_symbol"],
        suffixes=("", "_m"),
    )

    # keep the same columns as before where possible
    # sort
    if "expiry_date" in out.columns and "option_type" in out.columns and "strike" in out.columns:
        out = out.sort_values(["expiry_date", "option_type", "strike"], ascending=[True, True, True])

    return out.reset_index(drop=True)

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

    # ATR (informativo)
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
# Strategy selectors + criteria
# =========================

def _liquidity_score(df):
    t = pd.to_numeric(df["trades"], errors="coerce").fillna(0)
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return t + np.log1p(v)

def top_itm_buy_calls(df, expiry, regime: str, cfg: dict) -> pd.DataFrame:
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
    if regime == "Baixa":
        d["cheapness"] = d["mispricing_pct"].fillna(0)
        return d.sort_values(["liq","cheapness","iv"], ascending=[False, True, True]).head(3)

    return d.sort_values(["liq","iv"], ascending=[False, True]).head(3)

def top_sell_puts(df, expiry, regime: str, cfg: dict) -> pd.DataFrame:
    d = df[(df["option_type"]=="PUT") & (df["expiry_date"]==expiry)].copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d["moneyness_dist"] = pd.to_numeric(d["moneyness_dist"], errors="coerce")

    lo, hi = cfg["put_delta_lo"], cfg["put_delta_hi"]

    if regime == "Baixa":
        lo = cfg["put_delta_lo_bear"]
        hi = cfg["put_delta_hi_bear"]
        min_dist = cfg["put_min_otm_dist_bear"]
    else:
        min_dist = 0.0

    d = d[(d["moneyness"].isin(["OTM","ATM"])) & (d["delta"].between(lo, hi)) & (d["iv"].notna()) & (d["last_price"]>0)].copy()
    if min_dist > 0:
        d = d[(d["moneyness"]=="OTM") & (d["moneyness_dist"]>=min_dist)].copy()

    if d.empty:
        return d

    d["liq"] = _liquidity_score(d)
    d["score"] = (d["iv"].fillna(0)*100.0) + (d["last_price"].fillna(0)*5.0) + d["liq"].fillna(0)
    return d.sort_values(["score"], ascending=[False]).head(3)

def top_covered_calls(df, expiry, regime: str, cfg: dict) -> pd.DataFrame:
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
    d["score"] = (d["last_price"].fillna(0)*10.0) + d["liq"].fillna(0)
    return d.sort_values(["score"], ascending=[False]).head(3)

def top_vertical_spreads(df, expiry, kind="bull_call", regime: str = "Neutra", cfg: dict = None) -> pd.DataFrame:
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

        if regime == "Baixa":
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

    return pd.DataFrame(rows).sort_values(["liq","premium_total"], ascending=[False, True]).head(3)

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

    if regime == "Alta":
        sell_band = cfg["condor_call_sell_band_up"]
        buy_band  = cfg["condor_call_buy_band_up"]
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
        sell_band = cfg["condor_put_sell_band_down"]
        buy_band  = cfg["condor_put_buy_band_down"]
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
    st.warning("Nenhum ativo em assets (is_active=1).")
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
    st.warning("Sem dados em option_quote para esse ativo.")
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
    st.warning("Sem opções com trades>0 e last_price>0 para esse pregão.")
    st.stop()

expiry_list = sorted(pd.Series(df["expiry_date"]).dropna().unique().tolist())
expiry_choice = st.sidebar.selectbox("Vencimento", [str(x) for x in expiry_list])
expiry_sel = pd.to_datetime(expiry_choice).date()

atm_mode_ui = st.sidebar.radio("ATM", options=["Faixa percentual","Mais próximo"], index=0)
atm_mode = "pct" if atm_mode_ui=="Faixa percentual" else "nearest"
atm_pct = st.sidebar.slider("Faixa ATM (|K−S|/S)", 0.001, 0.05, 0.01, 0.001, disabled=(atm_mode!="pct"))

df2 = df[df["expiry_date"]==expiry_sel].copy()

# spot fallback if daily_bars not available
if spot is None or not np.isfinite(spot):
    s2 = pd.to_numeric(df2.get("spot", np.nan), errors="coerce")
    s2 = s2.dropna()
    spot = float(s2.iloc[0]) if len(s2) else None

df2 = classify_moneyness(df2, spot=spot if spot else np.nan, atm_mode=atm_mode, atm_pct=atm_pct)

quote_ts = pd.to_datetime(df2.get("quote_collected_at", pd.NaT), errors="coerce").max()
model_ts = pd.to_datetime(df2.get("model_collected_at", pd.NaT), errors="coerce").max()

df2["rate_r"] = pd.to_numeric(df2.get("rate_r", np.nan), errors="coerce")
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

# Strategy configuration (regime-aware)
cfg = {
    "deep_itm_delta_up": 0.75,
    "deep_itm_delta_neutral": 0.80,
    "deep_itm_delta_down": 0.85,

    "put_delta_lo": -0.35,
    "put_delta_hi": -0.10,
    "put_delta_lo_bear": -0.25,
    "put_delta_hi_bear": -0.08,
    "put_min_otm_dist_bear": 0.015,

    "cc_delta_up": (0.12, 0.25),
    "cc_delta_neutral": (0.15, 0.30),
    "cc_delta_down": (0.20, 0.40),

    "condor_call_sell_band": (0.18, 0.32),
    "condor_call_buy_band": (0.06, 0.16),
    "condor_call_sell_band_up": (0.12, 0.22),
    "condor_call_buy_band_up": (0.04, 0.10),

    "condor_put_sell_band": (-0.32, -0.18),
    "condor_put_buy_band": (-0.16, -0.06),
    "condor_put_sell_band_down": (-0.22, -0.12),
    "condor_put_buy_band_down": (-0.10, -0.04),
}

# Strategies + Payoff on select
st.subheader("Top 3 por estratégia (seleção mostra payoff com break-even)")

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
                "CALL ITM (moneyness=ITM)",
                f"delta ≥ {cfg['deep_itm_delta_up']:.2f}",
                "ranking: maior liquidez (trades+log(1+volume)) e menor IV",
            ],
            ["Em alta, deep ITM dá exposição com menor vega relativa e menos ruído de IV."]
        )
    elif regime == "Baixa":
        criteria_box(
            "Compra ITM",
            [
                "CALL ITM (moneyness=ITM)",
                f"delta ≥ {cfg['deep_itm_delta_down']:.2f}",
                "ranking: maior liquidez e preferência por mispricing_pct mais negativo (mais 'barata' vs BSM)",
            ],
            ["Em baixa, call deep ITM é mais defensiva; ainda assim, pode estar contra tendência."]
        )
    else:
        criteria_box(
            "Compra ITM",
            [
                "CALL ITM (moneyness=ITM)",
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
                f"moneyness_dist ≥ {cfg['put_min_otm_dist_bear']:.3f}",
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
        ["Em regime neutro, straddle pode capturar rompimentos; em alta/baixa, avalie se prefere direcional (travas)."]
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

# Table + smiles
st.subheader("Tabela completa (filtros: trades>0 e last_price>0)")
st.dataframe(format_table(df2), width="stretch", height=520, hide_index=True)

st.subheader("Sorriso de Volatilidade (IV vs Strike)")
cL, cR = st.columns(2)
with cL:
    plot_smile(df2, expiry_sel, "CALL", spot)
with cR:
    plot_smile(df2, expiry_sel, "PUT", spot)
