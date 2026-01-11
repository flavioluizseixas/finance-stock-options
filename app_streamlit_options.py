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
def load_spot_histvol(asset_id: int, trade_date: date):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT close AS spot, vol_annual AS hist_vol_annual
                FROM daily_bars
                WHERE asset_id=%s AND trade_date=%s
                """,
                (asset_id, trade_date),
            )
            row = cur.fetchone()
    if not row:
        return None, None
    return row.get("spot"), row.get("hist_vol_annual")

@st.cache_data(ttl=60)
def load_curve(trade_date: date):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, vertex_bd, rate, source, collected_at
                FROM yield_curve
                WHERE trade_date=%s
                ORDER BY vertex_bd
                """,
                (trade_date,),
            )
            rows = cur.fetchall() or []
    return pd.DataFrame(rows)

@st.cache_data(ttl=60)
def load_chain(asset_id: int, trade_date: date):
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
    out = df.copy()

    num_cols = [
        "strike","last_price","trades","volume","moneyness_dist",
        "iv","bsm_price","bsm_price_histvol","mispricing","mispricing_pct",
        "delta","gamma","vega","theta","rho","rate_r"
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["theta_day_365"] = out["theta"] / 365.0
    out["theta_day_252"] = out["theta"] / 252.0
    out["vega_1pct"] = out["vega"] * 0.01
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
    out = out[cols].copy()

    # arredondamento
    out["strike"] = out["strike"].round(2)
    out["last_price"] = out["last_price"].round(2)
    out["moneyness_dist"] = out["moneyness_dist"].round(4)
    out["rate_r"] = out["rate_r"].round(6)

    out["iv"] = out["iv"].round(4)
    out["iv_pct"] = out["iv_pct"].round(2)

    out["bsm_price"] = out["bsm_price"].round(4)
    out["bsm_price_histvol"] = out["bsm_price_histvol"].round(4)
    out["mispricing"] = out["mispricing"].round(4)
    out["mispricing_pct"] = out["mispricing_pct"].round(4)

    out["delta"] = out["delta"].round(4)
    out["gamma"] = out["gamma"].round(6)
    out["vega"] = out["vega"].round(4)
    out["vega_1pct"] = out["vega_1pct"].round(4)

    out["theta"] = out["theta"].round(4)
    out["theta_day_365"] = out["theta_day_365"].round(4)
    out["theta_day_252"] = out["theta_day_252"].round(4)

    out["rho"] = out["rho"].round(4)
    return out

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
# Estratégias (Top 3)
# =========================

def _liquidity_score(df):
    # score simples: trades + log(volume)
    t = pd.to_numeric(df["trades"], errors="coerce").fillna(0)
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return t + np.log1p(v)

def top_itm_buy_calls(df, spot, expiry) -> pd.DataFrame:
    d = df[(df["option_type"]=="CALL") & (df["expiry_date"]==expiry)].copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d["moneyness_dist"] = pd.to_numeric(d["moneyness_dist"], errors="coerce")
    d = d[(d["moneyness"]=="ITM") & (d["delta"]>=0.75) & (d["iv"].notna())].copy()
    if d.empty: return d
    d["liq"] = _liquidity_score(d)
    # preferir mais líquido e menor iv (entrada “menos cara” relativa)
    d = d.sort_values(["liq","iv"], ascending=[False, True]).head(3)
    return d

def top_sell_puts(df, expiry) -> pd.DataFrame:
    d = df[(df["option_type"]=="PUT") & (df["expiry_date"]==expiry)].copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d = d[(d["moneyness"].isin(["OTM","ATM"])) & (d["delta"].between(-0.35, -0.10)) & (d["iv"].notna())].copy()
    if d.empty: return d
    d["liq"] = _liquidity_score(d)
    # preferir IV alta (prêmio melhor) e liquidez
    d = d.sort_values(["iv","liq"], ascending=[False, False]).head(3)
    return d

def top_vertical_spreads(df, expiry, kind="bull_call") -> pd.DataFrame:
    # gera 3 travas por heurística simples:
    # bull_call: compra CALL ATM e vende CALL OTM (delta menor)
    # bear_put: compra PUT ATM e vende PUT OTM (mais distante)
    d = df[df["expiry_date"]==expiry].copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["last_price"], errors="coerce")
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d = d.dropna(subset=["strike","last_price"]).copy()

    rows = []
    if kind == "bull_call":
        calls = d[d["option_type"]=="CALL"].copy()
        buy = calls[calls["moneyness"]=="ATM"].copy()
        sell = calls[calls["moneyness"]=="OTM"].copy()
        for _, b in buy.iterrows():
            # escolhe vendedor com strike maior e delta mais baixo
            cand = sell[sell["strike"] > b["strike"]].copy()
            if cand.empty: continue
            cand = cand.sort_values(["strike"]).head(8)
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
                    "K_buy": b["strike"], "K_sell": s["strike"],
                    "debit": debit, "max_profit": max_profit, "rr": rr
                })
    else:
        puts = d[d["option_type"]=="PUT"].copy()
        buy = puts[puts["moneyness"]=="ATM"].copy()
        sell = puts[puts["moneyness"]=="OTM"].copy()
        for _, b in buy.iterrows():
            cand = sell[sell["strike"] < b["strike"]].copy()
            if cand.empty: continue
            cand = cand.sort_values(["strike"], ascending=False).head(8)
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
                    "K_buy": b["strike"], "K_sell": s["strike"],
                    "debit": debit, "max_profit": max_profit, "rr": rr
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # pega as 3 melhores por rr (com controle de débito não absurdo)
    out = out.sort_values(["rr","max_profit"], ascending=[False, False]).head(3)
    return out

def top_covered_calls(df, spot, expiry) -> pd.DataFrame:
    # venda coberta: CALL OTM com delta ~0.15..0.30 e boa liquidez
    d = df[(df["option_type"]=="CALL") & (df["expiry_date"]==expiry)].copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
    d = d[(d["moneyness"].isin(["OTM","ATM"])) & (d["delta"].between(0.15, 0.30)) & (d["iv"].notna())].copy()
    if d.empty: return d
    d["liq"] = _liquidity_score(d)
    d = d.sort_values(["last_price","liq"], ascending=[False, False]).head(3)
    return d

# =========================
# UI
# =========================

st.title("Opções – Última Coleta (Curva DI + Greeks + Estratégias)")

assets = load_assets()
if assets.empty:
    st.warning("Nenhum ativo em assets. Rode o pipeline primeiro.")
    st.stop()

ticker = st.sidebar.selectbox("Ativo", assets["ticker"].tolist())
asset_id = int(assets.loc[assets["ticker"]==ticker, "id"].iloc[0])

trade_date = load_latest_trade_date(asset_id)
if not trade_date:
    st.warning("Sem dados em option_quote.")
    st.stop()

spot, hist_vol_annual = load_spot_histvol(asset_id, trade_date)
df = load_chain(asset_id, trade_date)
if df.empty:
    st.warning("Sem opções com trades>0 e last_price>0.")
    st.stop()

# curve
df_curve = load_curve(trade_date)
curve_source = df_curve["source"].iloc[0] if not df_curve.empty else "N/A"
curve_ts = pd.to_datetime(df_curve["collected_at"], errors="coerce").max() if not df_curve.empty else None

expiry_list = sorted(df["expiry_date"].dropna().unique().tolist())
expiry_choice = st.sidebar.selectbox("Vencimento", [str(x) for x in expiry_list])
expiry_sel = pd.to_datetime(expiry_choice).date()

atm_mode_ui = st.sidebar.radio("ATM", options=["Faixa percentual","Mais próximo"], index=0)
atm_mode = "pct" if atm_mode_ui=="Faixa percentual" else "nearest"
atm_pct = st.sidebar.slider("Faixa ATM (|K−S|/S)", 0.001, 0.05, 0.01, 0.001, disabled=(atm_mode!="pct"))

df2 = df[df["expiry_date"]==expiry_sel].copy()

# spot fallback
if spot is None or not np.isfinite(spot):
    s2 = pd.to_numeric(df2["spot"], errors="coerce").dropna()
    spot = float(s2.iloc[0]) if len(s2) else None

df2 = classify_moneyness(df2, spot=spot if spot else np.nan, atm_mode=atm_mode, atm_pct=atm_pct)

quote_ts = pd.to_datetime(df2["quote_collected_at"], errors="coerce").max()
model_ts = pd.to_datetime(df2["model_collected_at"], errors="coerce").max()

# r exibido (pega mediana de rate_r do vencimento)
df2["rate_r"] = pd.to_numeric(df2["rate_r"], errors="coerce")
r_expiry = float(df2["rate_r"].dropna().median()) if df2["rate_r"].notna().any() else np.nan

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Ticker", ticker)
c2.metric("Pregão", str(trade_date))
c3.metric("Spot (close)", f"{spot:.4f}" if spot is not None else "N/A")
c4.metric("Hist vol anual", f"{hist_vol_annual:.4f}" if hist_vol_annual is not None else "N/A")
c5.metric("Vencimento", str(expiry_sel))
c6.metric("r (curva)", f"{r_expiry:.4%}" if np.isfinite(r_expiry) else "N/A")

st.caption(
    f"Coleta quotes: {quote_ts} | Coleta model: {model_ts} | Curva: {curve_source} | Curva coletada: {curve_ts}"
)

with st.expander("Ver curva de juros (vértices)", expanded=False):
    if df_curve.empty:
        st.info("Sem yield_curve para esta data (rode o pipeline).")
    else:
        tmp = df_curve.copy()
        tmp["rate_pct"] = (pd.to_numeric(tmp["rate"], errors="coerce") * 100).round(4)
        st.dataframe(tmp[["vertex_bd","rate","rate_pct","source","collected_at"]], width="stretch")

st.subheader("Top 3 por estratégia (heurística)")
colA, colB = st.columns(2)

with colA:
    st.markdown("### 1) Compra bem ITM (CALL ITM delta ≥ 0.75)")
    t1 = top_itm_buy_calls(df2, spot, expiry_sel)
    st.dataframe(format_table(t1) if not t1.empty else pd.DataFrame(), width="stretch")

    st.markdown("### 2) Venda de PUTs (PUT OTM/ATM delta −0.35..−0.10)")
    t2 = top_sell_puts(df2, expiry_sel)
    st.dataframe(format_table(t2) if not t2.empty else pd.DataFrame(), width="stretch")

with colB:
    st.markdown("### 3) Travas (Bull Call / Bear Put) – Top RR")
    bull = top_vertical_spreads(df2, expiry_sel, "bull_call")
    bear = top_vertical_spreads(df2, expiry_sel, "bear_put")
    st.markdown("**Bull Call Spread**")
    st.dataframe(bull if not bull.empty else pd.DataFrame(), width="stretch")
    st.markdown("**Bear Put Spread**")
    st.dataframe(bear if not bear.empty else pd.DataFrame(), width="stretch")

    st.markdown("### 4) Venda coberta (CALL OTM/ATM delta 0.15..0.30)")
    t4 = top_covered_calls(df2, spot, expiry_sel)
    st.dataframe(format_table(t4) if not t4.empty else pd.DataFrame(), width="stretch")

st.subheader("Tabela completa (filtros: trades>0 e last_price>0)")
st.dataframe(format_table(df2), width="stretch", height=520)

st.subheader("Sorriso de Volatilidade (IV vs Strike)")
col1, col2 = st.columns(2)
with col1:
    plot_smile(df2, expiry_sel, "CALL", spot)
with col2:
    plot_smile(df2, expiry_sel, "PUT", spot)
