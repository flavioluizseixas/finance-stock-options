# app_streamlit_covered_roll_pro.py
# ------------------------------------------------------------
# Streamlit (venda coberta) focado em "rolagem profissional"
# com regras:
#
# R1) Rolar quando extrínseco < X% do prêmio inicial
# R2) Rolar quando delta > 0.65 e DTE < 10
# R3) Rolar quando yield anualizado (nova) > yield anualizado (atual) por Y pontos
#     (pontos = pontos percentuais, ex: 0.50pp)
# + "Tendência" via indicadores técnicos da tabela daily_bars:
#   - SMA20 vs SMA50 vs SMA200
#   - MACD hist
#   - RSI14
#   - filtro opcional para evitar rolar contra tendência (configurável)
#
# Também exibe uma sumarização das regras usadas na tela.
#
# Pré-requisito de schema (conforme você mandou):
# CREATE TABLE daily_bars (
#   asset_id INT NOT NULL,
#   trade_date DATE NOT NULL,
#   open DOUBLE NULL, high DOUBLE NULL, low DOUBLE NULL, close DOUBLE NULL,
#   adj_close DOUBLE NULL, volume BIGINT NULL,
#   log_ret DOUBLE NULL, vol_annual DOUBLE NULL,
#   sma_20 DOUBLE NULL, sma_50 DOUBLE NULL, sma_200 DOUBLE NULL,
#   ema_12 DOUBLE NULL, ema_26 DOUBLE NULL, macd DOUBLE NULL, macd_signal DOUBLE NULL, macd_hist DOUBLE NULL,
#   rsi_14 DOUBLE NULL, atr_14 DOUBLE NULL,
#   PRIMARY KEY(asset_id, trade_date)
# );
#
# Obs:
# - Este app assume que option_quote e option_model existem e que option_model.delta, t_years etc. estão preenchidos.
# - "rolagem" aqui = fechar a call atual (buy-to-close) + abrir outra (sell-to-open)
# - Se você quiser travar "up & out" (K_novo>=K_atual e expiry_novo>expiry_atual), tem toggle.
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

st.set_page_config(page_title="Covered Call Rolling (Pro)", layout="wide")

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
# DB
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
def latest_trade_date(asset_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(trade_date) AS trade_date FROM option_quote WHERE asset_id=%s", (asset_id,))
            row = cur.fetchone()
    return row["trade_date"] if row else None

@st.cache_data(ttl=60)
def load_dailybar_indicators(asset_id: int, trade_date: date):
    """
    Carrega indicadores técnicos do daily_bars no pregão atual (trade_date).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  trade_date, close,
                  sma_20, sma_50, sma_200,
                  macd, macd_signal, macd_hist,
                  rsi_14, atr_14
                FROM daily_bars
                WHERE asset_id=%s AND trade_date=%s
                """,
                (asset_id, trade_date),
            )
            row = cur.fetchone()
    return row

@st.cache_data(ttl=60)
def load_calls_chain(asset_id: int, trade_date: date):
    """
    Cadeia de CALLs (último pregão), com dados de mercado (quote) e modelo (delta, t_years).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  oq.trade_date,
                  oq.option_symbol,
                  oq.option_type,
                  oq.expiry_date,
                  oq.strike,
                  oq.last_price,
                  oq.trades,
                  oq.volume,
                  oq.collected_at AS quote_collected_at,

                  om.spot,
                  om.delta,
                  om.iv,
                  om.theta,
                  om.rate_r,
                  om.t_years,
                  om.collected_at AS model_collected_at
                FROM option_quote oq
                LEFT JOIN option_model om
                  ON om.asset_id=oq.asset_id
                 AND om.trade_date=oq.trade_date
                 AND om.option_symbol=oq.option_symbol
                WHERE oq.asset_id=%s
                  AND oq.trade_date=%s
                  AND oq.option_type='CALL'
                  AND oq.trades > 0
                  AND oq.last_price > 0
                ORDER BY oq.expiry_date, oq.strike;
                """,
                (asset_id, trade_date),
            )
            rows = cur.fetchall() or []
    return pd.DataFrame(rows)


# =========================
# Helpers / metrics
# =========================
def add_covered_metrics(df: pd.DataFrame, spot: float):
    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date
    d["expiry_date"] = pd.to_datetime(d["expiry_date"]).dt.date

    for c in ["strike","last_price","delta","iv","theta","t_years","spot","rate_r","trades","volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    s0 = float(spot) if (spot is not None and np.isfinite(spot) and spot > 0) else np.nan
    d["spot"] = d["spot"].fillna(s0)

    d["DTE_days"] = (pd.to_datetime(d["expiry_date"]) - pd.to_datetime(d["trade_date"])).dt.days
    d["DTE_years"] = d["DTE_days"] / 365.0

    # intrínseco/extrínseco (CALL)
    d["intrinsic"] = np.maximum(d["spot"] - d["strike"], 0.0)
    d["extrinsic"] = d["last_price"] - d["intrinsic"]

    # moneyness
    d["moneyness_dist"] = (d["strike"] - d["spot"]).abs() / d["spot"]
    d["moneyness"] = np.where(d["strike"] < d["spot"], "ITM", "OTM")
    d.loc[d["moneyness_dist"] <= 0.01, "moneyness"] = "ATM"

    # yield (covered call): prêmio / spot
    d["yield"] = d["last_price"] / d["spot"]

    # annualized yield approx
    d["yield_ann_252"] = np.where(d["DTE_days"] > 0, d["yield"] * (252.0 / d["DTE_days"]), np.nan)

    # liquidity score
    d["liq"] = d["trades"].fillna(0) + np.log1p(d["volume"].fillna(0))

    return d

def trend_label(ind: dict) -> str:
    """
    Tendência simples usando SMA + MACD hist + RSI.
    """
    if not ind:
        return "N/A"

    close = ind.get("close")
    sma20 = ind.get("sma_20")
    sma50 = ind.get("sma_50")
    sma200 = ind.get("sma_200")
    macdh = ind.get("macd_hist")
    rsi = ind.get("rsi_14")

    # defaults
    up_votes = 0
    down_votes = 0

    # SMA stack
    if (sma20 is not None) and (sma50 is not None) and (sma200 is not None):
        if sma20 > sma50 > sma200:
            up_votes += 2
        elif sma20 < sma50 < sma200:
            down_votes += 2

    # close vs SMA50
    if (close is not None) and (sma50 is not None):
        if close > sma50:
            up_votes += 1
        elif close < sma50:
            down_votes += 1

    # MACD hist
    if macdh is not None:
        if macdh > 0:
            up_votes += 1
        elif macdh < 0:
            down_votes += 1

    # RSI regime
    if rsi is not None:
        if rsi >= 55:
            up_votes += 1
        elif rsi <= 45:
            down_votes += 1

    if up_votes >= down_votes + 2:
        return "Alta"
    if down_votes >= up_votes + 2:
        return "Baixa"
    return "Neutra"

def format_calls_table(d: pd.DataFrame):
    if d is None or d.empty:
        return pd.DataFrame()

    out = d.copy()
    # rounding
    for c in ["strike","last_price","intrinsic","extrinsic"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    for c in ["delta","iv","yield","yield_ann_252","moneyness_dist"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    cols = [
        "option_symbol","expiry_date","strike","moneyness","DTE_days",
        "last_price","intrinsic","extrinsic",
        "delta","iv",
        "yield","yield_ann_252",
        "trades","volume","liq"
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values(["expiry_date","strike"])

def rule_summary_box(X_extrinsic: float, Y_pp: float, use_trend: bool, trend: str, upout_only: bool):
    st.markdown("### Regras usadas (rolagem profissional)")
    st.markdown(
        f"""
- **R1 (Extrínseco baixo)**: sinal quando **extrínseco < {X_extrinsic:.0%} do prêmio inicial**  
- **R2 (Risco ITM/assignment)**: sinal quando **delta > 0.65** e **DTE < 10 dias**  
- **R3 (Melhoria de yield)**: sinal quando **yield anualizado da nova call > yield atual + {Y_pp:.2f} pp**  
- **Tendência (daily_bars)**: {"**ATIVADA**" if use_trend else "**DESATIVADA**"} — regime estimado: **{trend}**  
  - Heurística: SMA20/SMA50/SMA200 + MACD_hist + RSI14.
- **Filtro Up & Out**: {"**ATIVADO**" if upout_only else "**DESATIVADO**"} — exigir K_novo ≥ K_atual e vencimento novo > vencimento atual.
        """.strip()
    )

def compute_roll_candidates(
    calls_today: pd.DataFrame,
    current_symbol: str,
    current_entry_premium: float,
    X_extrinsic: float,
    Y_pp: float,
    upout_only: bool,
    use_trend_filter: bool,
    trend: str
) -> tuple[pd.DataFrame, dict]:
    """
    Retorna:
      - tabela com candidatos de rolagem (fechar atual + abrir nova)
      - flags dos gatilhos (R1, R2, R3)
    """
    d = calls_today.copy()
    cur = d[d["option_symbol"] == current_symbol].head(1)
    if cur.empty:
        return pd.DataFrame(), {}

    cur = cur.iloc[0]
    cur_price_close = float(cur["last_price"])
    cur_extrinsic = float(cur["extrinsic"])
    cur_delta = float(cur["delta"]) if np.isfinite(cur["delta"]) else np.nan
    cur_dte = int(cur["DTE_days"]) if np.isfinite(cur["DTE_days"]) else None
    cur_y_ann = float(cur["yield_ann_252"]) if np.isfinite(cur["yield_ann_252"]) else np.nan
    cur_K = float(cur["strike"])
    cur_exp = cur["expiry_date"]

    entry = float(current_entry_premium) if current_entry_premium and current_entry_premium > 0 else np.nan

    # --- R1
    R1 = False
    if np.isfinite(entry) and entry > 0 and np.isfinite(cur_extrinsic):
        R1 = cur_extrinsic < (X_extrinsic * entry)

    # --- R2
    R2 = False
    if (cur_dte is not None) and np.isfinite(cur_delta):
        R2 = (cur_delta > 0.65) and (cur_dte < 10)

    # --- R3 depende da comparação com candidatos; aqui só sinalizamos depois.
    flags = {"R1_extrinsic_low": R1, "R2_itm_risk": R2, "R3_yield_improve": False}

    # Build candidates
    cand = d[d["option_symbol"] != current_symbol].copy()

    # Up & Out
    if upout_only:
        cand = cand[(cand["expiry_date"] > cur_exp) & (cand["strike"] >= cur_K)].copy()

    # Trend filter (opcional):
    # Em tendência de ALTA, evitamos vender ITM e preferimos OTM/ATM com delta menor.
    # Em tendência de BAIXA, o app não impede, mas sinaliza que covered call pode limitar upside.
    if use_trend_filter and trend == "Alta":
        cand = cand[cand["moneyness"].isin(["OTM","ATM"])].copy()
        cand = cand[(cand["delta"].isna()) | (cand["delta"] <= 0.35)].copy()

    # crédito líquido: prêmio novo - custo de fechar
    cand["net_credit"] = cand["last_price"] - cur_price_close
    cand["net_yield"] = cand["net_credit"] / cand["spot"]
    cand["net_yield_ann_252"] = np.where(cand["DTE_days"] > 0, cand["net_yield"] * (252.0 / cand["DTE_days"]), np.nan)

    # melhoria de yield (R3): comparar yield anualizado da NOVA com yield anualizado da ATUAL
    # Y_pp em pontos percentuais => converter para fração
    Y = float(Y_pp) / 100.0
    cand["yield_improve"] = cand["yield_ann_252"] - cur_y_ann
    cand["rule_R3"] = cand["yield_improve"] > Y

    # R3 signal: existe alguma rolagem com melhoria suficiente?
    flags["R3_yield_improve"] = bool(cand["rule_R3"].fillna(False).any())

    # score do roll: prioriza net_credit, depois regra R3, depois yield_ann_252 e liquidez
    cand["liq"] = cand["liq"].fillna(0)
    cand["roll_score"] = (
        (cand["net_credit"].fillna(-999) * 10.0)
        + (cand["rule_R3"].fillna(False).astype(int) * 50.0)
        + (cand["yield_ann_252"].fillna(0) * 100.0)
        + (cand["liq"])
    )

    # ordenar e selecionar top
    cand = cand.sort_values(["roll_score"], ascending=False)

    # tabela final
    show_cols = [
        "option_symbol","expiry_date","strike","moneyness","DTE_days",
        "last_price","delta","iv",
        "yield","yield_ann_252",
        "net_credit","net_yield_ann_252",
        "yield_improve","rule_R3",
        "intrinsic","extrinsic","trades","volume","liq"
    ]
    show_cols = [c for c in show_cols if c in cand.columns]
    out = cand[show_cols].head(20).copy()

    # formatting
    for c in ["strike","last_price","net_credit","intrinsic","extrinsic"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    for c in ["delta","iv","yield","yield_ann_252","net_yield_ann_252","yield_improve"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    return out, flags


# =========================
# App UI
# =========================
st.title("Venda Coberta — Rolagem Profissional (Extrínseco, Delta/DTE, Yield, Tendência)")

assets = load_assets()
if assets.empty:
    st.warning("Nenhum ativo em assets. Rode o pipeline primeiro.")
    st.stop()

ticker = st.sidebar.selectbox("Ativo", assets["ticker"].tolist())
asset_id = int(assets.loc[assets["ticker"] == ticker, "id"].iloc[0])

td = latest_trade_date(asset_id)
if not td:
    st.warning("Sem dados em option_quote para este ativo.")
    st.stop()

calls = load_calls_chain(asset_id, td)
if calls.empty:
    st.warning("Sem CALLs com trades>0 e last_price>0 no último pregão.")
    st.stop()

# Indicadores técnicos do dia
ind = load_dailybar_indicators(asset_id, td)
spot = float(ind["close"]) if ind and ind.get("close") is not None else None
trend = trend_label(ind)

# Covered metrics
calls2 = add_covered_metrics(calls, spot=spot if spot else np.nan)

# =========================
# Sidebar: regras profissionais
# =========================
st.sidebar.markdown("## Regras profissionais (rolagem)")
X_extrinsic = st.sidebar.slider("X: Extrínseco < X% do prêmio inicial", 0.01, 0.80, 0.20, 0.01)
Y_pp = st.sidebar.slider("Y: melhora mínima de yield anualizado (p.p.)", 0.05, 5.00, 0.50, 0.05)

use_trend = st.sidebar.toggle("Usar tendência (daily_bars) para filtrar rolagens", value=True)
upout_only = st.sidebar.toggle("Up & Out (K_novo>=K_atual e expiry_novo>expiry_atual)", value=True)

# Preferência do usuário (definição do "melhor momento")
st.sidebar.markdown("## Preferência (seu estilo)")
target_delta_low = st.sidebar.slider("Meta delta da nova call (máx) em tendência de alta", 0.10, 0.45, 0.30, 0.01)
min_roll_dte = st.sidebar.slider("Meta DTE mínimo da nova call", 7, 60, 30, 1)
max_roll_dte = st.sidebar.slider("Meta DTE máximo da nova call", 15, 180, 60, 1)

# payoff toggles (opcional só para rolagem visual)
st.sidebar.markdown("## Payoff (opcional)")
show_pct = st.sidebar.toggle("Exibir payoff em % do spot", value=False)
mult_100 = st.sidebar.toggle("Payoff por 100 ações", value=True)
multiplier = 100.0 if mult_100 else 1.0

# Header summary
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ticker", ticker)
c2.metric("Pregão", str(td))
c3.metric("Spot (close)", f"{spot:.4f}" if spot else "N/A")
c4.metric("Tendência (heurística)", trend)
c5.metric("Qtd CALLs", str(len(calls2)))

rule_summary_box(X_extrinsic=X_extrinsic, Y_pp=Y_pp, use_trend=use_trend, trend=trend, upout_only=upout_only)

with st.expander("Indicadores técnicos (daily_bars) do último pregão", expanded=False):
    if not ind:
        st.info("Sem daily_bars/indicadores para esta data.")
    else:
        show = {k: ind.get(k) for k in ["trade_date","close","sma_20","sma_50","sma_200","macd_hist","rsi_14","atr_14"]}
        st.json(show)

# ============================================================
# 1) Cadeia atual e seleção da call vendida
# ============================================================
st.subheader("1) Selecionar a CALL atual (vendida) e avaliar gatilhos de rolagem")

# filtros de cadeia para exibir
colf1, colf2, colf3, colf4 = st.columns(4)
with colf1:
    mfilter = st.multiselect("Moneyness", ["OTM","ATM","ITM"], default=["OTM","ATM"])
with colf2:
    dte_min_view = st.number_input("DTE min (tabela)", value=int(min_roll_dte), min_value=0, step=1)
with colf3:
    dte_max_view = st.number_input("DTE max (tabela)", value=int(max_roll_dte), min_value=1, step=1)
with colf4:
    delta_max_view = st.number_input("Delta máx (tabela)", value=float(target_delta_low), min_value=0.0, max_value=1.0, step=0.01)

view = calls2.copy()
view = view[view["moneyness"].isin(mfilter)]
view = view[(view["DTE_days"] >= dte_min_view) & (view["DTE_days"] <= dte_max_view)]
view = view[(view["delta"].isna()) | (view["delta"] <= delta_max_view)]
view = view.sort_values(["expiry_date","strike"])

st.dataframe(format_calls_table(view), width="stretch", hide_index=True, height=420)

pos_symbol = st.selectbox("Qual CALL você está vendido (ou pretende vender)?", sorted(calls2["option_symbol"].unique().tolist()))

pos_row = calls2[calls2["option_symbol"] == pos_symbol].head(1)
if pos_row.empty:
    st.info("Não encontrei o option_symbol selecionado.")
    st.stop()

pos_row = pos_row.iloc[0]
pos_last = float(pos_row["last_price"])
pos_extrinsic = float(pos_row["extrinsic"])
pos_delta = float(pos_row["delta"]) if np.isfinite(pos_row["delta"]) else np.nan
pos_dte = int(pos_row["DTE_days"]) if np.isfinite(pos_row["DTE_days"]) else None
pos_y_ann = float(pos_row["yield_ann_252"]) if np.isfinite(pos_row["yield_ann_252"]) else np.nan

colp1, colp2, colp3 = st.columns(3)
with colp1:
    entry_premium = st.number_input("Prêmio inicial recebido (R$ por ação)", min_value=0.0, value=float(pos_last), step=0.01)
with colp2:
    st.write("**Extrínseco atual (aprox.)**")
    st.write(f"{pos_extrinsic:.2f}")
with colp3:
    st.write("**Delta / DTE**")
    st.write(f"delta={pos_delta:.2f} | DTE={pos_dte}")

# Gatilhos (R1/R2) para a posição atual
R1_now = (pos_extrinsic < (X_extrinsic * entry_premium)) if (entry_premium > 0 and np.isfinite(pos_extrinsic)) else False
R2_now = ((pos_delta > 0.65) and (pos_dte is not None and pos_dte < 10)) if np.isfinite(pos_delta) else False

st.markdown("#### Diagnóstico (posição atual)")
d1, d2, d3, d4 = st.columns(4)
d1.metric("R1 Extrínseco baixo", "SIM" if R1_now else "NÃO")
d2.metric("R2 Delta>0.65 & DTE<10", "SIM" if R2_now else "NÃO")
d3.metric("Yield anualizado atual", f"{pos_y_ann:.2%}" if np.isfinite(pos_y_ann) else "N/A")
d4.metric("Tendência", trend)

# ============================================================
# 2) Sugestões de rolagem (R3 + score + tendência)
# ============================================================
st.subheader("2) Sugestões de rolagem (fechar atual + abrir nova)")

# reforçar preferências no universo de candidatos:
# - DTE dentro de janela
cand2 = calls2.copy()
cand2 = cand2[(cand2["DTE_days"] >= min_roll_dte) & (cand2["DTE_days"] <= max_roll_dte)].copy()

# se tendência de alta e filtro ativo => preferir OTM/ATM e deltas menores
if use_trend and trend == "Alta":
    cand2 = cand2[cand2["moneyness"].isin(["OTM","ATM"])]
    cand2 = cand2[(cand2["delta"].isna()) | (cand2["delta"] <= target_delta_low)]

roll_table, flags = compute_roll_candidates(
    calls_today=cand2,
    current_symbol=pos_symbol,
    current_entry_premium=entry_premium,
    X_extrinsic=X_extrinsic,
    Y_pp=Y_pp,
    upout_only=upout_only,
    use_trend_filter=use_trend,
    trend=trend
)

# sinal R3 e regra completa (qualquer R aciona rolagem)
R3_now = bool(flags.get("R3_yield_improve", False))
should_roll = (R1_now or R2_now or R3_now)

st.markdown("#### Sinal consolidado")
s1, s2, s3, s4 = st.columns(4)
s1.metric("R3 (há rolagem melhor em yield)", "SIM" if R3_now else "NÃO")
s2.metric("Recomendação heurística", "ROLAR" if should_roll else "MANTER")
s3.metric("Preço p/ fechar (aprox.)", f"{pos_last:.2f}")
s4.metric("Extrínseco / prêmio inicial", f"{(pos_extrinsic/entry_premium):.2%}" if entry_premium>0 else "N/A")

st.caption("Nota: R3 é calculado comparando o yield anualizado da opção nova com o yield anualizado da opção atual. Ajuste Y conforme seu custo/risco.")

if roll_table.empty:
    st.info("Sem candidatos de rolagem com os filtros atuais.")
else:
    st.dataframe(roll_table, width="stretch", hide_index=True, height=520)

# ============================================================
# 3) Payoff opcional (rolagem selecionada)
# ============================================================
st.subheader("3) Payoff (opcional) para uma rolagem escolhida")

if roll_table.empty:
    st.info("Escolha filtros que gerem candidatos para visualizar payoff.")
else:
    pick_symbol = st.selectbox("Escolha a nova CALL (sell-to-open) para visualizar payoff:", ["(nenhuma)"] + roll_table["option_symbol"].tolist())
    if pick_symbol != "(nenhuma)":
        new_row = calls2[calls2["option_symbol"] == pick_symbol].head(1)
        if new_row.empty:
            st.info("Não consegui encontrar essa opção na cadeia completa.")
        else:
            new_row = new_row.iloc[0]
            newK = float(new_row["strike"])
            newP = float(new_row["last_price"])

            # payoff da posição coberta após rolagem:
            # Você mantém 1 ação + vende 1 call nova, e “paga” para fechar a call antiga.
            # Aproximação: prêmio líquido = newP - pos_last
            net_premium = newP - pos_last

            s0 = float(spot) if spot else float(new_row["spot"])

            lo = max(0.01, 0.5 * s0)
            hi = 1.5 * s0
            ST = np.linspace(lo, hi, 700)

            # payoff: (ST-S0) + [net_premium - max(ST-newK,0)]
            pnl = ((ST - s0) + (net_premium - np.maximum(ST - newK, 0.0))) * float(multiplier)

            denom = s0 * float(multiplier)
            pnl_plot = (pnl / denom) * 100.0 if show_pct else pnl

            # break-even
            be = []
            y = pnl
            x = ST
            sgn = np.sign(y)
            for i in range(len(x)-1):
                if sgn[i] == 0:
                    be.append(float(x[i]))
                if sgn[i] * sgn[i+1] < 0:
                    x0, x1 = x[i], x[i+1]
                    y0, y1 = y[i], y[i+1]
                    xb = x0 - y0 * (x1 - x0) / (y1 - y0)
                    be.append(float(xb))
            be = sorted(set([round(v, 6) for v in be]))

            fig = plt.figure()
            plt.plot(ST, pnl_plot, linestyle="-")
            plt.axhline(0.0)
            plt.axvline(s0)
            for b in be:
                plt.axvline(b, linestyle="--")
            plt.title(f"Payoff rolagem: fechar {pos_symbol} e vender {pick_symbol} (net={net_premium:.2f})")
            plt.xlabel("Preço do ativo no vencimento (ST)")
            plt.ylabel("Payoff (% do spot)" if show_pct else "Payoff (P&L no vencimento)")
            st.pyplot(fig, clear_figure=True)

            cbe1, cbe2, cbe3, cbe4 = st.columns(4)
            cbe1.write("**Break-even(s)**")
            cbe1.write(", ".join([f"{v:.2f}" for v in be]) if be else "—")
            cbe2.write("**Net premium (rolagem)**")
            cbe2.write(f"{net_premium:.2f} por ação")
            cbe3.write("**K novo**")
            cbe3.write(f"{newK:.2f}")
            cbe4.write("**Multiplicador**")
            cbe4.write(f"{multiplier:g}")

st.divider()
st.caption(
    "Este painel usa heurísticas e dados coletados. Não é recomendação. "
    "Ajuste X/Y e filtros de tendência conforme sua política (dividendos, custos, impostos, margem, risco de exercício)."
)
