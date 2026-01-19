# app_streamlit_options.py
# ------------------------------------------------------------
# Streamlit dashboard for options:
# - ITM/ATM/OTM tables, greeks, BSM/IV, last collection only
# - Strategy picker (Top N) with payoff chart shown ONLY when user selects an operation
#
# Features:
# 1) Expiry selection OPTIONAL: "(Todos)" aggregates across expiries.
# 2) Asset selection OPTIONAL: "(Todos)" aggregates across ALL active tickers.
#    - For each asset, we use its own latest trade_date (MAX in option_quote).
#    - We load daily_bars (close/indicators) for that same trade_date (if available).
#    - We compute regime PER asset and keep it in the dataset.
# 3) Top N.
# 4) Travas: debit and credit grouped together with tabs (comparability).
# 5) Liquidity criteria by leg + by pair (spreads/straddle):
#    - Single-leg: gates + class OK/ALERTA/RUIM, optional hard filter, penalty in score.
#    - Pair: min_liq + symmetry ratio, class OK/ALERTA/RUIM, hard filter + penalty.
#
# Universe filter (recommended):
# - abs(ln(strike/spot)) <= mny_log_max
# - abs(delta) >= delta_abs_min
# - last_price >= last_price_min
# - volume_fin >= vol_fin_min
#
# DB tables assumed:
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

TOP_N = 5

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

    details["ATR14"] = atr

    if up >= down + 2:
        return "Alta", up, down, details
    if down >= up + 2:
        return "Baixa", up, down, details
    return "Neutra", up, down, details

@st.cache_data(ttl=60)
def load_all_active_data(assets_df: pd.DataFrame):
    all_rows = []
    meta_rows = []

    for _, a in assets_df.iterrows():
        asset_id = int(a["id"])
        ticker = str(a["ticker"])

        td = load_latest_trade_date(asset_id)
        if not td:
            continue

        ind = load_daily_indicators(asset_id, td) or {}
        regime, up_score, down_score, reg_details = infer_regime(ind, rsi_hi=55, rsi_lo=45)

        close = _to_float(ind.get("close"))
        vol_annual = _to_float(ind.get("vol_annual"))

        df = load_chain(asset_id, td)
        if df is None or df.empty:
            continue

        if close is None or not np.isfinite(close) or close <= 0:
            s2 = pd.to_numeric(df.get("spot", np.nan), errors="coerce").dropna()
            close = float(s2.iloc[0]) if len(s2) else None

        df = df.copy()
        df["asset_id"] = asset_id
        df["ticker"] = ticker
        df["trade_date"] = td
        df["spot_ref"] = close
        df["hist_vol_annual_ref"] = vol_annual
        df["regime"] = regime
        df["regime_up_score"] = up_score
        df["regime_down_score"] = down_score

        all_rows.append(df)

        meta_rows.append({
            "asset_id": asset_id,
            "ticker": ticker,
            "trade_date": td,
            "spot_ref": close,
            "hist_vol_annual_ref": vol_annual,
            "regime": regime,
            "up_score": up_score,
            "down_score": down_score,
            "SMA_stack": reg_details.get("SMA_stack"),
            "Close_vs_SMA50": reg_details.get("Close_vs_SMA50"),
            "MACD_hist": reg_details.get("MACD_hist"),
            "RSI": reg_details.get("RSI"),
        })

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    meta_df = pd.DataFrame(meta_rows)
    return df_all, meta_df

# =========================
# Moneyness + formatting
# =========================

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

def format_table(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Ensure no duplicate column names (pyarrow fails)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    num_cols = [
        "strike","last_price","trades","volume","moneyness_dist",
        "iv","bsm_price","bsm_price_histvol","mispricing","mispricing_pct",
        "delta","gamma","vega","theta","rho","rate_r","spot_ref",
        "liq","liq_buy","liq_sell","liq_call","liq_put",
        "liq_min","liq_ratio","premium_total","debit","credit","rr","cr","rr_adj","cr_adj",
        "mny_log_abs","volume_fin"
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
        "ticker","trade_date",
        "option_symbol","option_type","expiry_date",
        "spot_ref",
        "strike","last_price","trades","volume",
        "moneyness","moneyness_dist",
        "mny_log_abs","volume_fin",
        "liq","liq_class",
        "rate_r","iv","iv_pct",
        "bsm_price","bsm_price_histvol",
        "mispricing","mispricing_pct",
        "delta","gamma","vega","vega_1pct",
        "theta","theta_day_365","theta_day_252","rho",
        "quote_collected_at","model_collected_at",
        "regime"
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()

    for c in ["spot_ref","strike","last_price","volume_fin"]:
        if c in out.columns:
            out[c] = out[c].round(4 if c == "spot_ref" else 2)
    for c in ["moneyness_dist","rate_r","liq","liq_buy","liq_sell","liq_call","liq_put","liq_min","liq_ratio","mny_log_abs"]:
        if c in out.columns:
            out[c] = out[c].round(6 if c == "rate_r" else 4)
    for c in ["iv","iv_pct","bsm_price","bsm_price_histvol","mispricing","mispricing_pct","delta","gamma","vega","vega_1pct","theta","theta_day_365","theta_day_252","rho"]:
        if c in out.columns:
            out[c] = out[c].round(6 if c == "gamma" else 4)

    return out

# =========================
# Universe filter (moneyness/delta/ultimo/volume_fin)
# =========================

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

    # Proxy de volume financeiro diário (R$):
    # last_price (R$ por ação) * volume (contratos) * multiplicador (100 ações/contrato na B3)
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

# =========================
# Payoff
# =========================

def payoff_long_call(ST, K, premium):  return np.maximum(ST - K, 0.0) - premium
def payoff_short_call(ST, K, premium): return premium - np.maximum(ST - K, 0.0)
def payoff_long_put(ST, K, premium):   return np.maximum(K - ST, 0.0) - premium
def payoff_short_put(ST, K, premium):  return premium - np.maximum(K - ST, 0.0)
def payoff_stock(ST, S0):              return ST - S0

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
    ST = np.linspace(lo, hi, 800)

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
# UI helpers
# =========================

def selectable_table(df: pd.DataFrame, key: str, label: str):
    if df is None or df.empty:
        st.info("Sem operações candidatas para esta estratégia.")
        return None

    disp = df.copy().reset_index(drop=True)
    st.caption(label)

    # Streamlit versions differ: some don't support on_select/selection_mode
    # + new API de width="stretch" (em vez de use_container_width=True)
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
        # Fallback (versões antigas)
        try:
            st.dataframe(disp, width="stretch", hide_index=True)
        except TypeError:
            st.dataframe(disp)  # fallback extremo

        options = [f"{i}: {disp.iloc[i].to_dict()}" for i in range(len(disp))]
        pick = st.selectbox("Escolha a operação para ver o payoff:", ["(nenhuma)"] + options, key=f"{key}_sb")
        if pick == "(nenhuma)":
            return None
        return int(pick.split(":")[0])

def criteria_box(title: str, bullets: list[str], notes: list[str] | None = None):
    st.markdown(f"**Critérios – {title}:**")
    st.markdown("\n".join([f"- {b}" for b in bullets]))
    if notes:
        st.markdown("\n".join([f"> {n}" for n in notes]))

def _ensure_date(x):
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return x

def _filter_expiry(df: pd.DataFrame, expiry: date | None):
    if expiry is None:
        return df.copy()
    return df[df["expiry_date"] == expiry].copy()

def _liquidity_score(df):
    t = pd.to_numeric(df.get("trades", 0), errors="coerce").fillna(0)
    v = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    return t + np.log1p(v)

# =========================
# Liquidity: leg + pair criteria
# =========================

def _liq_raw(trades, volume):
    t = float(trades) if np.isfinite(trades) else 0.0
    v = float(volume) if np.isfinite(volume) else 0.0
    return t + np.log1p(max(v, 0.0))

def _liq_leg_from_row(row):
    t = pd.to_numeric(row.get("trades", 0), errors="coerce")
    v = pd.to_numeric(row.get("volume", 0), errors="coerce")
    t = float(t) if pd.notna(t) else 0.0
    v = float(v) if pd.notna(v) else 0.0
    return _liq_raw(t, v), t, v

def _liq_class_single(liq, trades, volume, cfg):
    if trades < cfg["liq_min_trades"] or volume < cfg["liq_min_volume"]:
        return "RUIM"
    if liq >= cfg["liq_min_ok"]:
        return "OK"
    if liq >= cfg["liq_min_alert"]:
        return "ALERTA"
    return "RUIM"

def _liq_pair_metrics(liq1, liq2):
    mn = min(liq1, liq2)
    mx = max(liq1, liq2)
    ratio = (mn / mx) if mx > 0 else 0.0
    return mn, mx, ratio

def _liq_class_pair(min_liq, ratio, cfg):
    if (min_liq >= cfg["liq_pair_min_ok"]) and (ratio >= cfg["liq_pair_ratio_ok"]):
        return "OK"
    if (min_liq >= cfg["liq_pair_min_alert"]) and (ratio >= cfg["liq_pair_ratio_alert"]):
        return "ALERTA"
    return "RUIM"

def _liq_penalty(liq_class, cfg):
    if liq_class == "OK":
        return 0.0
    if liq_class == "ALERTA":
        return float(cfg["liq_penalty_alert"])
    return float(cfg["liq_penalty_bad"])

# =========================
# Strategies
# =========================

def top_itm_buy_calls(df, expiry: date | None, cfg: dict, top_n: int) -> pd.DataFrame:
    d = _filter_expiry(df, expiry)
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
        return d

    d["dmin"] = d["regime"].apply(dmin_for_reg)
    d = d[(d["delta"] >= d["dmin"]) & (d["iv"].notna()) & (d["last_price"] > 0)].copy()
    if d.empty:
        return d

    d["liq"] = _liquidity_score(d)

    d["liq_class"] = d.apply(lambda r: _liq_class_single(
        liq=float(pd.to_numeric(r.get("liq", 0), errors="coerce") or 0),
        trades=float(pd.to_numeric(r.get("trades", 0), errors="coerce") or 0),
        volume=float(pd.to_numeric(r.get("volume", 0), errors="coerce") or 0),
        cfg=cfg
    ), axis=1)

    if cfg.get("liq_single_filter_hard", True):
        d = d[d["liq_class"].isin(["OK", "ALERTA"])].copy()
        if d.empty:
            return d

    d["cheapness"] = d["mispricing_pct"].fillna(0)
    d["score"] = (
        d["liq"]
        - (d["iv"].fillna(0) * 10.0)
        + np.where(d["regime"] == "Baixa", (-d["cheapness"] * 0.5), 0.0)
        - d["liq_class"].map(lambda c: _liq_penalty(c, cfg))
    )

    out = d.sort_values(["score", "liq"], ascending=[False, False]).head(top_n)
    out["expiry_date"] = out["expiry_date"].apply(_ensure_date)
    return out.drop(columns=["dmin"], errors="ignore")

def top_sell_puts(df, expiry: date | None, cfg: dict, top_n: int) -> pd.DataFrame:
    d = _filter_expiry(df, expiry)
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
        return d

    d[["lo", "hi", "min_dist"]] = d["regime"].apply(lambda r: pd.Series(bounds(r)))
    d = d[(d["delta"].between(d["lo"], d["hi"])) & (d["iv"].notna()) & (d["last_price"] > 0)].copy()
    d = d[~((d["min_dist"] > 0) & ((d["moneyness"] != "OTM") | (d["moneyness_dist"] < d["min_dist"])))].copy()
    if d.empty:
        return d

    d["liq"] = _liquidity_score(d)
    d["liq_class"] = d.apply(lambda r: _liq_class_single(
        liq=float(pd.to_numeric(r.get("liq", 0), errors="coerce") or 0),
        trades=float(pd.to_numeric(r.get("trades", 0), errors="coerce") or 0),
        volume=float(pd.to_numeric(r.get("volume", 0), errors="coerce") or 0),
        cfg=cfg
    ), axis=1)

    if cfg.get("liq_single_filter_hard", True):
        d = d[d["liq_class"].isin(["OK", "ALERTA"])].copy()
        if d.empty:
            return d

    d["score"] = (
        (d["iv"].fillna(0) * 100.0)
        + (d["last_price"].fillna(0) * 5.0)
        + d["liq"].fillna(0)
        - d["liq_class"].map(lambda c: _liq_penalty(c, cfg))
    )

    out = d.sort_values(["score"], ascending=[False]).head(top_n)
    out["expiry_date"] = out["expiry_date"].apply(_ensure_date)
    return out.drop(columns=["lo", "hi", "min_dist"], errors="ignore")

def top_covered_calls(df, expiry: date | None, cfg: dict, top_n: int) -> pd.DataFrame:
    d = _filter_expiry(df, expiry)
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
        return d

    d["liq"] = _liquidity_score(d)
    d["liq_class"] = d.apply(lambda r: _liq_class_single(
        liq=float(pd.to_numeric(r.get("liq", 0), errors="coerce") or 0),
        trades=float(pd.to_numeric(r.get("trades", 0), errors="coerce") or 0),
        volume=float(pd.to_numeric(r.get("volume", 0), errors="coerce") or 0),
        cfg=cfg
    ), axis=1)

    if cfg.get("liq_single_filter_hard", True):
        d = d[d["liq_class"].isin(["OK", "ALERTA"])].copy()
        if d.empty:
            return d

    d["score"] = (
        (d["last_price"].fillna(0) * 10.0)
        + d["liq"].fillna(0)
        - d["liq_class"].map(lambda c: _liq_penalty(c, cfg))
    )

    out = d.sort_values(["score"], ascending=[False]).head(top_n)
    out["expiry_date"] = out["expiry_date"].apply(_ensure_date)
    return out.drop(columns=["mset", "dlo", "dhi"], errors="ignore")

def top_straddles(df, expiry: date | None, top_n: int, cfg: dict) -> pd.DataFrame:
    d = _filter_expiry(df, expiry)
    d["strike"] = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
    d["last_price"] = pd.to_numeric(d.get("last_price", np.nan), errors="coerce")
    d = d.dropna(subset=["strike", "last_price"]).copy()
    d = d[d["last_price"] > 0].copy()

    calls = d[(d["option_type"].str.upper() == "CALL") & (d["moneyness"] == "ATM")].copy()
    puts  = d[(d["option_type"].str.upper() == "PUT")  & (d["moneyness"] == "ATM")].copy()
    if calls.empty or puts.empty:
        return pd.DataFrame()

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

        liq_c, _, _ = _liq_leg_from_row(c)
        liq_p, _, _ = _liq_leg_from_row(p)
        min_liq, _, ratio = _liq_pair_metrics(liq_c, liq_p)
        liq_class = _liq_class_pair(min_liq, ratio, cfg)

        if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
            continue

        prem = float(c["last_price"]) + float(p["last_price"])
        rows.append({
            "strategy": "Long Straddle (ATM)",
            "ticker": c["ticker"],
            "trade_date": c["trade_date"],
            "expiry_date": _ensure_date(c["expiry_date"]),
            "spot_ref": float(c["spot_ref"]) if pd.notna(c["spot_ref"]) else np.nan,
            "call": c["option_symbol"],
            "put": p["option_symbol"],
            "K": float(c["strike"]),
            "P_call": float(c["last_price"]),
            "P_put": float(p["last_price"]),
            "premium_total": prem,
            "liq_call": liq_c,
            "liq_put": liq_p,
            "liq_min": min_liq,
            "liq_ratio": ratio,
            "liq_class": liq_class,
            "regime": c.get("regime", "N/A")
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["premium_total"] = pd.to_numeric(out["premium_total"], errors="coerce")
    out["liq_min"] = pd.to_numeric(out["liq_min"], errors="coerce")
    out = out.sort_values(["liq_min", "premium_total"], ascending=[False, True]).head(top_n)
    return out

# ---- Spreads: DEBIT

def _top_vertical_spreads_debit_one_expiry(df, expiry: date, kind="bull_call", top_n: int = 5, cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = {}

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

                liq_b, _, _ = _liq_leg_from_row(b)
                liq_s, _, _ = _liq_leg_from_row(s)
                min_liq, _, ratio = _liq_pair_metrics(liq_b, liq_s)
                liq_class = _liq_class_pair(min_liq, ratio, cfg)

                if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                    continue

                rows.append({
                    "strategy": "Bull Call Spread (Debit)",
                    "ticker": b["ticker"],
                    "trade_date": b["trade_date"],
                    "expiry_date": _ensure_date(expiry),
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
                    "liq_buy": liq_b,
                    "liq_sell": liq_s,
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

                liq_b, _, _ = _liq_leg_from_row(b)
                liq_s, _, _ = _liq_leg_from_row(s)
                min_liq, _, ratio = _liq_pair_metrics(liq_b, liq_s)
                liq_class = _liq_class_pair(min_liq, ratio, cfg)

                if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                    continue

                rows.append({
                    "strategy": "Bear Put Spread (Debit)",
                    "ticker": b["ticker"],
                    "trade_date": b["trade_date"],
                    "expiry_date": _ensure_date(expiry),
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
                    "liq_buy": liq_b,
                    "liq_sell": liq_s,
                    "liq_min": min_liq,
                    "liq_ratio": ratio,
                    "liq_class": liq_class,
                    "regime": b.get("regime", "N/A")
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["rr_adj"] = out["rr"] - out["liq_class"].map(lambda c: _liq_penalty(c, cfg) * 0.05)
    return out.sort_values(["rr_adj", "max_profit_est", "liq_min"], ascending=[False, False, False]).head(top_n)

def top_vertical_spreads_debit(df, expiry: date | None, top_n: int, cfg: dict):
    expiries = sorted(df["expiry_date"].dropna().unique().tolist())
    expiries = [pd.to_datetime(x).date() for x in expiries]

    if expiry is not None:
        bull = _top_vertical_spreads_debit_one_expiry(df, expiry, kind="bull_call", top_n=top_n, cfg=cfg)
        bear = _top_vertical_spreads_debit_one_expiry(df, expiry, kind="bear_put", top_n=top_n, cfg=cfg)
        return bull, bear

    all_bull, all_bear = [], []
    for ex in expiries:
        b = _top_vertical_spreads_debit_one_expiry(df, ex, kind="bull_call", top_n=top_n, cfg=cfg)
        p = _top_vertical_spreads_debit_one_expiry(df, ex, kind="bear_put", top_n=top_n, cfg=cfg)
        if not b.empty: all_bull.append(b)
        if not p.empty: all_bear.append(p)

    bull = pd.concat(all_bull, ignore_index=True) if all_bull else pd.DataFrame()
    bear = pd.concat(all_bear, ignore_index=True) if all_bear else pd.DataFrame()

    if not bull.empty:
        bull = bull.sort_values(["rr_adj", "max_profit_est", "liq_min"], ascending=[False, False, False]).head(top_n)
    if not bear.empty:
        bear = bear.sort_values(["rr_adj", "max_profit_est", "liq_min"], ascending=[False, False, False]).head(top_n)

    return bull, bear

# ---- Spreads: CREDIT

def _top_credit_spreads_one_expiry(df, expiry: date, kind="bull_put", cfg: dict | None = None, top_n: int = 5) -> pd.DataFrame:
    if cfg is None:
        cfg = {}

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

                    liq_s, _, _ = _liq_leg_from_row(s)
                    liq_b, _, _ = _liq_leg_from_row(b)
                    min_liq, _, ratio = _liq_pair_metrics(liq_s, liq_b)
                    liq_class = _liq_class_pair(min_liq, ratio, cfg)

                    if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                        continue

                    rows.append({
                        "strategy": "Bull Put Spread (Credit)",
                        "ticker": ticker,
                        "trade_date": s["trade_date"],
                        "expiry_date": _ensure_date(expiry),
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
                        "liq_sell": liq_s,
                        "liq_buy": liq_b,
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

                    liq_s, _, _ = _liq_leg_from_row(s)
                    liq_b, _, _ = _liq_leg_from_row(b)
                    min_liq, _, ratio = _liq_pair_metrics(liq_s, liq_b)
                    liq_class = _liq_class_pair(min_liq, ratio, cfg)

                    if (min_liq < cfg["liq_pair_hard_min"]) or (ratio < cfg["liq_pair_hard_ratio"]):
                        continue

                    rows.append({
                        "strategy": "Bear Call Spread (Credit)",
                        "ticker": ticker,
                        "trade_date": s["trade_date"],
                        "expiry_date": _ensure_date(expiry),
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
                        "liq_sell": liq_s,
                        "liq_buy": liq_b,
                        "liq_min": min_liq,
                        "liq_ratio": ratio,
                        "liq_class": liq_class,
                        "regime": reg
                    })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["cr_adj"] = out["cr"] - out["liq_class"].map(lambda c: _liq_penalty(c, cfg) * 0.05)
    out = out.sort_values(["cr_adj", "credit", "liq_min"], ascending=[False, False, False]).head(top_n)
    return out

def top_credit_spreads(df, expiry: date | None, cfg: dict, top_n: int):
    expiries = sorted(df["expiry_date"].dropna().unique().tolist())
    expiries = [pd.to_datetime(x).date() for x in expiries]

    if expiry is not None:
        bps = _top_credit_spreads_one_expiry(df, expiry, kind="bull_put", cfg=cfg, top_n=top_n)
        bcs = _top_credit_spreads_one_expiry(df, expiry, kind="bear_call", cfg=cfg, top_n=top_n)
        return bps, bcs

    all_bps, all_bcs = [], []
    for ex in expiries:
        a = _top_credit_spreads_one_expiry(df, ex, kind="bull_put", cfg=cfg, top_n=top_n)
        b = _top_credit_spreads_one_expiry(df, ex, kind="bear_call", cfg=cfg, top_n=top_n)
        if not a.empty: all_bps.append(a)
        if not b.empty: all_bcs.append(b)

    bps = pd.concat(all_bps, ignore_index=True) if all_bps else pd.DataFrame()
    bcs = pd.concat(all_bcs, ignore_index=True) if all_bcs else pd.DataFrame()

    if not bps.empty:
        bps = bps.sort_values(["cr_adj", "credit", "liq_min"], ascending=[False, False, False]).head(top_n)
    if not bcs.empty:
        bcs = bcs.sort_values(["cr_adj", "credit", "liq_min"], ascending=[False, False, False]).head(top_n)

    return bps, bcs

# =========================
# Smile plot
# =========================

def plot_smile(df: pd.DataFrame, ticker: str, expiry_date: date, option_type: str):
    d = df[(df["ticker"] == ticker) & (df["expiry_date"] == expiry_date) & (df["option_type"].str.upper() == option_type.upper())].copy()
    d["iv"] = pd.to_numeric(d.get("iv", np.nan), errors="coerce")
    d["strike"] = pd.to_numeric(d.get("strike", np.nan), errors="coerce")
    d = d.dropna(subset=["iv", "strike"]).sort_values("strike")
    if d.empty:
        st.info(f"Sem IV para {option_type} em {ticker} no vencimento {expiry_date}.")
        return
    spot = pd.to_numeric(d.get("spot_ref", np.nan), errors="coerce").dropna()
    spot = float(spot.iloc[0]) if len(spot) else np.nan

    fig = plt.figure()
    plt.plot(d["strike"], d["iv"], marker="o", linestyle="-")
    if np.isfinite(spot):
        plt.axvline(x=float(spot))
    plt.xlabel("Strike")
    plt.ylabel("IV")
    plt.title(f"Sorriso de Vol – {ticker} – {option_type} – {expiry_date}")
    st.pyplot(fig, clear_figure=True)

# =========================
# App
# =========================

st.title("Opções – Última Coleta (Greeks + BSM/IV + Estratégias + Payoff)")

assets = load_assets()
if assets.empty:
    st.warning("Nenhum ativo em assets. Rode o pipeline primeiro.")
    st.stop()

# Sidebar
st.sidebar.markdown("## Universo")
ticker_ui = st.sidebar.selectbox("Ativo (opcional)", ["(Todos)"] + assets["ticker"].tolist())

st.sidebar.markdown("## Payoff (config)")
show_pct = st.sidebar.toggle("Exibir payoff em % do spot", value=False)
mult_100 = st.sidebar.toggle("Payoff por 100 ações (multiplicador=100)", value=True)
multiplier = 100.0 if mult_100 else 1.0

st.sidebar.markdown("## Tendência (daily_bars)")
rsi_hi = st.sidebar.slider("RSI alto (força)", 50, 70, 55, 1)
rsi_lo = st.sidebar.slider("RSI baixo (fraqueza)", 30, 50, 45, 1)

st.sidebar.markdown("## ATM")
atm_mode_ui = st.sidebar.radio("ATM", options=["Faixa percentual", "Mais próximo"], index=0)
atm_mode = "pct" if atm_mode_ui == "Faixa percentual" else "nearest"
atm_pct = st.sidebar.slider("Faixa ATM (|K−S|/S)", 0.001, 0.05, 0.01, 0.001, disabled=(atm_mode != "pct"))

st.sidebar.markdown("## Liquidez (gates)")
liq_single_filter_hard = st.sidebar.toggle("Filtrar RUIM (perna única)", value=True)
liq_pair_filter_hard = st.sidebar.toggle("Filtrar RUIM (pares)", value=True)

# Universe filter (recommended)
st.sidebar.markdown("## Filtro do universo (recomendado)")
use_universe_filter = st.sidebar.toggle("Ativar filtro strike/delta/último/vol financeiro", value=True)

mny_log_max = st.sidebar.slider("|ln(K/S)| máx", 0.10, 1.00, 0.40, 0.05)
delta_abs_min = st.sidebar.slider("|delta| mín", 0.00, 0.50, 0.05, 0.01)
last_price_min = st.sidebar.number_input("Último (last_price) mín", min_value=0.0, value=0.05, step=0.01)

opt_contract_mult = st.sidebar.number_input("Multiplicador do contrato (p/ volume financeiro)", min_value=1, value=100, step=1)
vol_fin_min = st.sidebar.number_input("Volume financeiro mín (R$)", min_value=0.0, value=5000.0, step=500.0)

# Load data
if ticker_ui == "(Todos)":
    df_raw, meta = load_all_active_data(assets)
    if df_raw.empty:
        st.warning("Sem dados de opções (trades>0 e last_price>0) para os ativos ativos.")
        st.stop()
else:
    asset_id = int(assets.loc[assets["ticker"] == ticker_ui, "id"].iloc[0])
    td = load_latest_trade_date(asset_id)
    if not td:
        st.warning("Sem dados em option_quote para este ativo.")
        st.stop()
    ind = load_daily_indicators(asset_id, td) or {}
    regime, up_score, down_score, reg_details = infer_regime(ind, rsi_hi=float(rsi_hi), rsi_lo=float(rsi_lo))
    close = _to_float(ind.get("close"))
    vol_annual = _to_float(ind.get("vol_annual"))
    df_chain = load_chain(asset_id, td)
    if df_chain.empty:
        st.warning("Sem opções com trades>0 e last_price>0 para este ativo.")
        st.stop()
    if close is None or not np.isfinite(close) or close <= 0:
        s2 = pd.to_numeric(df_chain.get("spot", np.nan), errors="coerce").dropna()
        close = float(s2.iloc[0]) if len(s2) else None

    df_raw = df_chain.copy()
    df_raw["asset_id"] = asset_id
    df_raw["ticker"] = ticker_ui
    df_raw["trade_date"] = td
    df_raw["spot_ref"] = close
    df_raw["hist_vol_annual_ref"] = vol_annual
    df_raw["regime"] = regime
    df_raw["regime_up_score"] = up_score
    df_raw["regime_down_score"] = down_score

    meta = pd.DataFrame([{
        "asset_id": asset_id,
        "ticker": ticker_ui,
        "trade_date": td,
        "spot_ref": close,
        "hist_vol_annual_ref": vol_annual,
        "regime": regime,
        "up_score": up_score,
        "down_score": down_score,
        "SMA_stack": reg_details.get("SMA_stack"),
        "Close_vs_SMA50": reg_details.get("Close_vs_SMA50"),
        "MACD_hist": reg_details.get("MACD_hist"),
        "RSI": reg_details.get("RSI"),
    }])

# Expiry selection
expiry_list = sorted(df_raw["expiry_date"].dropna().unique().tolist())
expiry_list = [pd.to_datetime(x).date() for x in expiry_list]
expiry_ui = st.sidebar.selectbox("Vencimento (opcional)", ["(Todos)"] + [str(x) for x in expiry_list])
expiry_sel = None if expiry_ui == "(Todos)" else pd.to_datetime(expiry_ui).date()

# Moneyness
df2 = classify_moneyness_multi(df_raw.copy(), atm_mode=atm_mode, atm_pct=float(atm_pct))

# Apply universe filter
cfg_univ = {
    "mny_log_max": float(mny_log_max),
    "delta_abs_min": float(delta_abs_min),
    "last_price_min": float(last_price_min),
    "vol_fin_min": float(vol_fin_min),
    "opt_contract_mult": float(opt_contract_mult),
}
if use_universe_filter:
    df2 = apply_universe_filter(df2, cfg_univ)

# Header metrics
if ticker_ui == "(Todos)":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universo", f"{len(meta)} ativos")
    c2.metric("Opções (linhas)", f"{len(df2):,}".replace(",", "."))
    c3.metric("Vencimento", str(expiry_sel) if expiry_sel else "Todos")
    if not meta.empty and "regime" in meta.columns:
        c4.metric("Regimes", f"Alta={int((meta['regime']=='Alta').sum())} | Baixa={int((meta['regime']=='Baixa').sum())} | Neutra={int((meta['regime']=='Neutra').sum())}")
else:
    r = meta.iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ticker", ticker_ui)
    c2.metric("Pregão", str(r["trade_date"]))
    c3.metric("Spot (close)", f"{float(r['spot_ref']):.4f}" if pd.notna(r["spot_ref"]) else "N/A")
    c4.metric("Hist vol anual", f"{float(r['hist_vol_annual_ref']):.4f}" if pd.notna(r["hist_vol_annual_ref"]) else "N/A")
    c5.metric("Vencimento", str(expiry_sel) if expiry_sel else "Todos")
    c6.metric("Regime", str(r["regime"]))

with st.expander("Tendência / Indicadores (daily_bars) usados nas estratégias", expanded=True):
    if ticker_ui == "(Todos)":
        st.info("Modo (Todos os ativos): regimes e indicadores são mostrados por ativo.")
        if not meta.empty:
            st.dataframe(meta.sort_values(["ticker"]), width="stretch", hide_index=True)
        else:
            st.write("—")
    else:
        st.dataframe(meta, width="stretch", hide_index=True)

# Config
cfg = {
    # delta thresholds
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

    # credit spreads delta bands
    "bps_put_sell_band": (-0.30, -0.15),
    "bps_put_buy_band":  (-0.15, -0.05),
    "bps_put_sell_band_down": (-0.25, -0.12),
    "bps_put_buy_band_down":  (-0.12, -0.04),

    "bcs_call_sell_band": (0.15, 0.30),
    "bcs_call_buy_band":  (0.05, 0.15),
    "bcs_call_sell_band_up": (0.12, 0.22),
    "bcs_call_buy_band_up":  (0.04, 0.10),

    # liquidity single-leg gates
    "liq_min_trades": 5,
    "liq_min_volume": 50,
    "liq_min_ok": 8.0,
    "liq_min_alert": 5.0,
    "liq_penalty_alert": 1.0,
    "liq_penalty_bad": 3.0,
    "liq_single_filter_hard": bool(liq_single_filter_hard),

    # liquidity pair gates
    "liq_pair_min_ok": 6.0,
    "liq_pair_min_alert": 4.0,
    "liq_pair_ratio_ok": 0.40,
    "liq_pair_ratio_alert": 0.25,
    "liq_pair_hard_min": 3.0 if liq_pair_filter_hard else 0.0,
    "liq_pair_hard_ratio": 0.20 if liq_pair_filter_hard else 0.0,
}

st.subheader(f"Top {TOP_N} por estratégia (considerando {'todos os ativos' if ticker_ui=='(Todos)' else 'o ativo selecionado'})")

colA, colB = st.columns(2)

with colA:
    st.markdown("### 1) Compra bem ITM (CALL deep ITM)")
    criteria_box(
        "Compra ITM",
        [
            "CALL ITM (moneyness=ITM)",
            "delta mínimo depende do regime por ativo (Alta/Neutra/Baixa)",
            "ranking: liquidez − 10×IV (e leve ajuste por mispricing em baixa) − penalidade liquidez",
            f"gates liquidez: trades≥{cfg['liq_min_trades']} e volume≥{cfg['liq_min_volume']}",
            "vencimento: " + ("Todos" if expiry_sel is None else str(expiry_sel)),
        ]
    )
    t1 = top_itm_buy_calls(df2, expiry_sel, cfg=cfg, top_n=TOP_N)
    idx = selectable_table(format_table(t1), key="t1", label="Selecione uma operação para ver o payoff.")
    if idx is not None and not t1.empty:
        row = t1.iloc[idx]
        spot = float(row["spot_ref"]) if pd.notna(row.get("spot_ref", np.nan)) else None
        op = {
            "label": f"{row['ticker']} | Compra ITM – LONG CALL {row['option_symbol']} (exp={_ensure_date(row['expiry_date'])}, K={row['strike']}, P={row['last_price']})",
            "legs": [{"type":"CALL","side":"LONG","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": False,
        }
        plot_payoff(op, spot, multiplier, show_pct)

    st.markdown("### 2) Venda de PUTs (renda)")
    criteria_box(
        "Venda de PUT",
        [
            "PUT OTM/ATM com delta em banda regime-aware (mais conservadora em baixa)",
            "ranking: 100×IV + 5×prêmio + liq − penalidade liquidez",
            "vencimento: " + ("Todos" if expiry_sel is None else str(expiry_sel)),
        ]
    )
    t2 = top_sell_puts(df2, expiry_sel, cfg=cfg, top_n=TOP_N)
    idx2 = selectable_table(format_table(t2), key="t2", label="Selecione uma operação para ver o payoff.")
    if idx2 is not None and not t2.empty:
        row = t2.iloc[idx2]
        spot = float(row["spot_ref"]) if pd.notna(row.get("spot_ref", np.nan)) else None
        op = {
            "label": f"{row['ticker']} | Venda de PUT – SHORT PUT {row['option_symbol']} (exp={_ensure_date(row['expiry_date'])}, K={row['strike']}, P={row['last_price']})",
            "legs": [{"type":"PUT","side":"SHORT","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": False,
        }
        plot_payoff(op, spot, multiplier, show_pct)

    st.markdown("### 5) Long Straddle (ATM)")
    criteria_box(
        "Long Straddle",
        [
            "BUY CALL ATM + BUY PUT ATM (mesmo ticker e vencimento)",
            "filtro: min_liq e simetria de liquidez (ratio)",
            "ranking: maior liq_min e menor prêmio total",
        ]
    )
    tS = top_straddles(df2, expiry_sel, top_n=TOP_N, cfg=cfg)
    idxS = selectable_table(tS, key="straddle", label="Selecione um straddle para ver o payoff.")
    if idxS is not None and not tS.empty:
        r = tS.iloc[idxS]
        spot = float(r["spot_ref"]) if pd.notna(r.get("spot_ref", np.nan)) else None
        op = {
            "label": f"{r['ticker']} | Long Straddle – exp={_ensure_date(r['expiry_date'])} | BUY {r['call']} + BUY {r['put']}",
            "legs": [
                {"type":"CALL","side":"LONG","K":float(r["K"]),"premium":float(r["P_call"])},
                {"type":"PUT","side":"LONG","K":float(r["K"]),"premium":float(r["P_put"])},
            ],
            "include_stock": False,
        }
        plot_payoff(op, spot, multiplier, show_pct)

with colB:
    st.markdown("### 3) Travas (Débito e Crédito) – comparáveis em abas")
    criteria_box(
        "Travas",
        [
            "Débito: Bull Call / Bear Put (rank por RR ajustado por liquidez)",
            "Crédito: Bull Put / Bear Call (rank por CR ajustado por liquidez)",
            "filtro: min_liq e simetria de liquidez (ratio)",
        ]
    )

    tab_debit, tab_credit = st.tabs(["Travas no DÉBITO (RR)", "Travas no CRÉDITO (CR)"])

    with tab_debit:
        bull_deb, bear_deb = top_vertical_spreads_debit(df2, expiry_sel, top_n=TOP_N, cfg=cfg)
        tA, tB = st.tabs(["Bull Call (Debit)", "Bear Put (Debit)"])

        def _render_debit(tab, data):
            with tab:
                idx_sp = selectable_table(data, key=f"debit_{tab}", label="Selecione uma trava (débito) para ver o payoff.")
                if idx_sp is not None and not data.empty:
                    r = data.iloc[idx_sp]
                    spot = float(r["spot_ref"]) if pd.notna(r.get("spot_ref", np.nan)) else None
                    if "Bull Call" in r["strategy"]:
                        op = {
                            "label": f"{r['ticker']} | Bull Call (Debit) – exp={_ensure_date(r['expiry_date'])} | BUY {r['buy']} | SELL {r['sell']} (debit={r['debit']:.2f})",
                            "legs": [
                                {"type":"CALL","side":"LONG","K":float(r["K_buy"]),"premium":float(r["P_buy"])},
                                {"type":"CALL","side":"SHORT","K":float(r["K_sell"]),"premium":float(r["P_sell"])},
                            ],
                            "include_stock": False,
                        }
                    else:
                        op = {
                            "label": f"{r['ticker']} | Bear Put (Debit) – exp={_ensure_date(r['expiry_date'])} | BUY {r['buy']} | SELL {r['sell']} (debit={r['debit']:.2f})",
                            "legs": [
                                {"type":"PUT","side":"LONG","K":float(r["K_buy"]),"premium":float(r["P_buy"])},
                                {"type":"PUT","side":"SHORT","K":float(r["K_sell"]),"premium":float(r["P_sell"])},
                            ],
                            "include_stock": False,
                        }
                    plot_payoff(op, spot, multiplier, show_pct)

        _render_debit(tA, bull_deb)
        _render_debit(tB, bear_deb)

    with tab_credit:
        bps, bcs = top_credit_spreads(df2, expiry_sel, cfg=cfg, top_n=TOP_N)
        tC, tD = st.tabs(["Bull Put (Credit)", "Bear Call (Credit)"])

        def _render_credit(tab, data):
            with tab:
                idx_cs = selectable_table(data, key=f"credit_{tab}", label="Selecione uma trava (crédito) para ver o payoff.")
                if idx_cs is not None and not data.empty:
                    r = data.iloc[idx_cs]
                    spot = float(r["spot_ref"]) if pd.notna(r.get("spot_ref", np.nan)) else None
                    if "Bull Put" in r["strategy"]:
                        op = {
                            "label": f"{r['ticker']} | Bull Put (Credit) – exp={_ensure_date(r['expiry_date'])} | SELL {r['sell']} | BUY {r['buy']} (credit={r['credit']:.2f})",
                            "legs": [
                                {"type":"PUT","side":"SHORT","K":float(r["K_sell"]),"premium":float(r["P_sell"])},
                                {"type":"PUT","side":"LONG", "K":float(r["K_buy"]), "premium":float(r["P_buy"])},
                            ],
                            "include_stock": False,
                        }
                    else:
                        op = {
                            "label": f"{r['ticker']} | Bear Call (Credit) – exp={_ensure_date(r['expiry_date'])} | SELL {r['sell']} | BUY {r['buy']} (credit={r['credit']:.2f})",
                            "legs": [
                                {"type":"CALL","side":"SHORT","K":float(r["K_sell"]),"premium":float(r["P_sell"])},
                                {"type":"CALL","side":"LONG", "K":float(r["K_buy"]), "premium":float(r["P_buy"])},
                            ],
                            "include_stock": False,
                        }
                    plot_payoff(op, spot, multiplier, show_pct)

        _render_credit(tC, bps)
        _render_credit(tD, bcs)

    st.markdown("### 4) Venda coberta (1 ação + short CALL)")
    criteria_box(
        "Venda coberta",
        [
            "CALL OTM/ATM (conforme regime por ativo) com delta em banda regime-aware",
            "ranking: 10×prêmio + liq − penalidade liquidez",
        ]
    )
    t4 = top_covered_calls(df2, expiry_sel, cfg=cfg, top_n=TOP_N)
    idx4 = selectable_table(format_table(t4), key="t4", label="Selecione uma operação para ver o payoff.")
    if idx4 is not None and not t4.empty:
        row = t4.iloc[idx4]
        spot = float(row["spot_ref"]) if pd.notna(row.get("spot_ref", np.nan)) else None
        op = {
            "label": f"{row['ticker']} | Venda Coberta – exp={_ensure_date(row['expiry_date'])} | STOCK + SHORT CALL {row['option_symbol']} (K={row['strike']}, P={row['last_price']})",
            "legs": [{"type":"CALL","side":"SHORT","K":float(row["strike"]),"premium":float(row["last_price"])}],
            "include_stock": True,
            "stock_qty": 1.0
        }
        plot_payoff(op, spot, multiplier, show_pct)

st.subheader("Tabela completa (filtros: trades>0 e last_price>0) – universo carregado")
st.dataframe(format_table(df2), width="stretch", height=520, hide_index=True)

st.subheader("Sorriso de Volatilidade (IV vs Strike)")
if ticker_ui == "(Todos)":
    st.info("Para plotar sorriso de vol, selecione um ativo específico.")
elif expiry_sel is None:
    st.info("Selecione um vencimento específico para plotar o sorriso de vol (CALL/PUT).")
else:
    cL, cR = st.columns(2)
    with cL:
        plot_smile(df2, ticker_ui, expiry_sel, "CALL")
    with cR:
        plot_smile(df2, ticker_ui, expiry_sel, "PUT")

