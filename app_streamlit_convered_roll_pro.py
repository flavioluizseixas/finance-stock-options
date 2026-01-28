# app_streamlit_covered_roll_pro.py
# ------------------------------------------------------------
# Covered Call Rolling (Pro) — PARQUET-first
# Lê estes arquivos em DATA_DIR (default ./data):
#   assets.parquet
#   daily_bars.parquet
#   option_quote.parquet
#   option_model.parquet
#
# Recursos pedidos:
# - Abas: "Rolar para cima" e "Rolar para o lado" (rankings diferentes)
# - Toggle: "Somente rolagens com crédito líquido" (net_credit >= 0)
#
# Regras revisadas:
# R1) Captura de prêmio: fechar por <= (1 - CAPTURE)% do prêmio inicial
#     (CAPTURE=0.70 => fechar por <= 30% do prêmio inicial)
#     + opcional confirmação: extrínseco <= EXTR_PCT% do prêmio inicial
#
# R2) Risco ITM/assignment: delta > DELTA_TH e DTE < DTE_TH
#
# R3) Melhoria: yield_ann(nova) > yield_ann(atual) + Y_pp
#     + guardrail: net_credit >= NET_CREDIT_MIN
#
# Observação:
# - "Rolar para cima": exige K_novo > K_atual (e opcionalmente expiry >= atual)
# - "Rolar para o lado": K_novo ~ K_atual (tolerância) e expiry > atual
# ------------------------------------------------------------

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="Covered Call Rolling (Pro) — Parquet", layout="wide")

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

ASSETS_FP = DATA_DIR / "assets.parquet"
DAILY_FP  = DATA_DIR / "daily_bars.parquet"
OQ_FP     = DATA_DIR / "option_quote.parquet"
OM_FP     = DATA_DIR / "option_model.parquet"


# =========================
# Utils
# =========================
def _to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def _num(s):
    return pd.to_numeric(s, errors="coerce")


# =========================
# Parquet loaders
# =========================
@st.cache_data(ttl=60)
def load_assets() -> pd.DataFrame:
    if not ASSETS_FP.exists():
        return pd.DataFrame(columns=["id", "ticker", "is_active"])
    df = pd.read_parquet(ASSETS_FP)

    if "is_active" not in df.columns:
        df["is_active"] = 1
    if "ticker" not in df.columns:
        raise ValueError("assets.parquet precisa ter coluna 'ticker'.")

    # id/asset_id
    if "id" not in df.columns and "asset_id" in df.columns:
        df = df.rename(columns={"asset_id": "id"})
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)

    df = df[df["is_active"] == 1].copy()
    df = df.sort_values("ticker").reset_index(drop=True)
    return df


@st.cache_data(ttl=60)
def load_option_quote(asset_id: int) -> pd.DataFrame:
    if not OQ_FP.exists():
        return pd.DataFrame()

    df = pd.read_parquet(OQ_FP)

    if "asset_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "asset_id"})

    need = {"asset_id","trade_date","expiry_date","option_symbol","option_type","strike","last_price","trades","volume"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"option_quote.parquet sem colunas obrigatórias: {missing}")

    df = df[df["asset_id"] == asset_id].copy()
    if df.empty:
        return df

    df["trade_date"] = _to_date_series(df["trade_date"])
    df["expiry_date"] = _to_date_series(df["expiry_date"])

    for c in ["strike","last_price","trades","volume"]:
        df[c] = _num(df[c])

    return df


@st.cache_data(ttl=60)
def load_option_model(asset_id: int) -> pd.DataFrame:
    if not OM_FP.exists():
        return pd.DataFrame()

    df = pd.read_parquet(OM_FP)

    if "asset_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "asset_id"})

    need = {"asset_id","trade_date","option_symbol"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"option_model.parquet sem colunas obrigatórias: {missing}")

    df = df[df["asset_id"] == asset_id].copy()
    if df.empty:
        return df

    df["trade_date"] = _to_date_series(df["trade_date"])

    for c in ["spot","delta","iv","theta","rate_r","t_years"]:
        if c in df.columns:
            df[c] = _num(df[c])
        else:
            df[c] = np.nan

    return df


@st.cache_data(ttl=60)
def load_daily_bars(asset_id: int) -> pd.DataFrame:
    if not DAILY_FP.exists():
        return pd.DataFrame()

    df = pd.read_parquet(DAILY_FP)

    if "asset_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "asset_id"})

    need = {"asset_id","trade_date","close"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"daily_bars.parquet sem colunas obrigatórias: {missing}")

    df = df[df["asset_id"] == asset_id].copy()
    if df.empty:
        return df

    df["trade_date"] = _to_date_series(df["trade_date"])
    for c in ["close","sma_20","sma_50","sma_200","macd_hist","rsi_14","atr_14"]:
        if c in df.columns:
            df[c] = _num(df[c])
        else:
            df[c] = np.nan

    return df


@st.cache_data(ttl=60)
def latest_trade_date(asset_id: int):
    oq = load_option_quote(asset_id)
    if oq.empty:
        return None
    return oq["trade_date"].max()


@st.cache_data(ttl=60)
def load_dailybar_indicators(asset_id: int, trade_date: date):
    db = load_daily_bars(asset_id)
    if db.empty:
        return None
    row = db.loc[db["trade_date"] == trade_date].tail(1)
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    return {
        "trade_date": r.get("trade_date"),
        "close": r.get("close"),
        "sma_20": r.get("sma_20"),
        "sma_50": r.get("sma_50"),
        "sma_200": r.get("sma_200"),
        "macd_hist": r.get("macd_hist"),
        "rsi_14": r.get("rsi_14"),
        "atr_14": r.get("atr_14"),
    }


@st.cache_data(ttl=60)
def load_calls_chain(asset_id: int, trade_date: date) -> pd.DataFrame:
    oq = load_option_quote(asset_id)
    if oq.empty:
        return pd.DataFrame()

    oq = oq[oq["trade_date"] == trade_date].copy()
    oq = oq[(oq["option_type"] == "CALL") & (oq["trades"] > 0) & (oq["last_price"] > 0)].copy()
    if oq.empty:
        return pd.DataFrame()

    om = load_option_model(asset_id)
    if not om.empty:
        om = om[om["trade_date"] == trade_date].copy()
        keep = ["trade_date","option_symbol","spot","delta","iv","theta","rate_r","t_years"]
        df = oq.merge(om[keep], on=["trade_date","option_symbol"], how="left")
    else:
        df = oq.copy()
        for c in ["spot","delta","iv","theta","rate_r","t_years"]:
            df[c] = np.nan

    return df.sort_values(["expiry_date","strike"]).reset_index(drop=True)


# =========================
# Helpers / metrics
# =========================
def add_covered_metrics(df: pd.DataFrame, spot: float):
    d = df.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce").dt.date
    d["expiry_date"] = pd.to_datetime(d["expiry_date"], errors="coerce").dt.date

    for c in ["strike","last_price","delta","iv","theta","t_years","spot","rate_r","trades","volume"]:
        if c in d.columns:
            d[c] = _num(d[c])

    s0 = float(spot) if (spot is not None and np.isfinite(spot) and spot > 0) else np.nan
    d["spot"] = d["spot"].fillna(s0)

    d["DTE_days"] = (pd.to_datetime(d["expiry_date"]) - pd.to_datetime(d["trade_date"])).dt.days
    d["DTE_years"] = d["DTE_days"] / 365.0

    d["intrinsic"] = np.maximum(d["spot"] - d["strike"], 0.0)
    d["extrinsic"] = d["last_price"] - d["intrinsic"]

    d["moneyness_dist"] = (d["strike"] - d["spot"]).abs() / d["spot"]
    d["moneyness"] = np.where(d["strike"] < d["spot"], "ITM", "OTM")
    d.loc[d["moneyness_dist"] <= 0.01, "moneyness"] = "ATM"

    d["yield"] = d["last_price"] / d["spot"]
    d["yield_ann_252"] = np.where(d["DTE_days"] > 0, d["yield"] * (252.0 / d["DTE_days"]), np.nan)

    d["liq"] = d["trades"].fillna(0) + np.log1p(d["volume"].fillna(0))
    return d


def trend_label(ind: dict) -> str:
    if not ind:
        return "N/A"

    close = ind.get("close")
    sma20 = ind.get("sma_20")
    sma50 = ind.get("sma_50")
    sma200 = ind.get("sma_200")
    macdh = ind.get("macd_hist")
    rsi = ind.get("rsi_14")

    up_votes = 0
    down_votes = 0

    if np.isfinite(sma20) and np.isfinite(sma50) and np.isfinite(sma200):
        if sma20 > sma50 > sma200:
            up_votes += 2
        elif sma20 < sma50 < sma200:
            down_votes += 2

    if np.isfinite(close) and np.isfinite(sma50):
        if close > sma50:
            up_votes += 1
        elif close < sma50:
            down_votes += 1

    if np.isfinite(macdh):
        if macdh > 0:
            up_votes += 1
        elif macdh < 0:
            down_votes += 1

    if np.isfinite(rsi):
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
    for c in ["strike","last_price","intrinsic","extrinsic"]:
        if c in out.columns:
            out[c] = _num(out[c]).round(2)
    for c in ["delta","iv","yield","yield_ann_252","moneyness_dist","liq"]:
        if c in out.columns:
            out[c] = _num(out[c]).round(4)

    cols = [
        "option_symbol","expiry_date","strike","moneyness","DTE_days",
        "last_price","intrinsic","extrinsic",
        "delta","iv",
        "yield","yield_ann_252",
        "trades","volume","liq"
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values(["expiry_date","strike"])


def compute_candidates_base(
    calls_all: pd.DataFrame,
    pos_symbol: str,
    entry_premium: float,
):
    cur = calls_all[calls_all["option_symbol"] == pos_symbol].head(1)
    if cur.empty:
        return None, pd.DataFrame()

    cur = cur.iloc[0]
    cur_price_close = float(cur["last_price"])
    cur_extrinsic = float(cur["extrinsic"])
    cur_delta = float(cur["delta"]) if np.isfinite(cur["delta"]) else np.nan
    cur_dte = int(cur["DTE_days"]) if np.isfinite(cur["DTE_days"]) else None
    cur_y_ann = float(cur["yield_ann_252"]) if np.isfinite(cur["yield_ann_252"]) else np.nan
    cur_K = float(cur["strike"])
    cur_exp = cur["expiry_date"]
    spot = float(cur["spot"]) if np.isfinite(cur["spot"]) else np.nan

    entry = float(entry_premium) if entry_premium and entry_premium > 0 else np.nan

    cand = calls_all[calls_all["option_symbol"] != pos_symbol].copy()
    cand["net_credit"] = cand["last_price"] - cur_price_close
    cand["net_yield"] = cand["net_credit"] / cand["spot"]
    cand["net_yield_ann_252"] = np.where(cand["DTE_days"] > 0, cand["net_yield"] * (252.0 / cand["DTE_days"]), np.nan)

    # comparação de yield
    cand["yield_improve"] = cand["yield_ann_252"] - cur_y_ann

    cur_info = dict(
        option_symbol=pos_symbol,
        price_close=cur_price_close,
        extrinsic=cur_extrinsic,
        delta=cur_delta,
        dte=cur_dte,
        y_ann=cur_y_ann,
        K=cur_K,
        exp=cur_exp,
        spot=spot,
        entry=entry,
    )
    return cur_info, cand


def apply_rules_and_score(
    cur_info: dict,
    cand: pd.DataFrame,
    *,
    # R1
    capture_pct: float,
    use_extrinsic_confirm: bool,
    extrinsic_pct: float,
    # R2
    delta_th: float,
    dte_th: int,
    # R3
    Y_pp: float,
    net_credit_min: float,
):
    entry = cur_info["entry"]
    cur_price_close = cur_info["price_close"]
    cur_extrinsic = cur_info["extrinsic"]
    cur_delta = cur_info["delta"]
    cur_dte = cur_info["dte"]

    # R1
    R1_now = (cur_price_close <= (1.0 - capture_pct) * entry) if (np.isfinite(entry) and entry > 0) else False
    R1b_now = (cur_extrinsic <= extrinsic_pct * entry) if (use_extrinsic_confirm and np.isfinite(entry) and entry > 0) else False
    R1_final = (R1_now and R1b_now) if use_extrinsic_confirm else R1_now

    # R2
    R2_now = ((cur_delta > delta_th) and (cur_dte is not None and cur_dte < dte_th)) if np.isfinite(cur_delta) else False

    # R3 por candidato
    Y = float(Y_pp) / 100.0
    cand = cand.copy()
    cand["rule_R3"] = (cand["yield_improve"] > Y) & (cand["net_credit"] >= float(net_credit_min))

    flags = {
        "R1": bool(R1_final),
        "R2": bool(R2_now),
        "R3_any": bool(cand["rule_R3"].fillna(False).any()),
    }
    return flags, cand


def rank_up(cur_info: dict, cand: pd.DataFrame):
    # "para cima": K_novo > K_atual (e normalmente expiry >= atual)
    c = cand.copy()
    c = c[(c["strike"] > cur_info["K"]) & (c["expiry_date"] >= cur_info["exp"])].copy()

    # score prioriza: aumentar strike + net_credit + liquidez + (bonus R3) + yield
    c["K_gain"] = (c["strike"] - cur_info["K"]) / cur_info["K"]
    c["roll_score_up"] = (
        c["K_gain"].fillna(0) * 200.0
        + c["net_credit"].fillna(-999) * 10.0
        + c["rule_R3"].fillna(False).astype(int) * 50.0
        + c["yield_ann_252"].fillna(0) * 100.0
        + c["liq"].fillna(0)
    )
    return c.sort_values("roll_score_up", ascending=False)


def rank_side(cur_info: dict, cand: pd.DataFrame, strike_tol: float):
    # "para o lado": strike próximo do atual (tolerância) e expiry > atual
    c = cand.copy()
    c = c[(c["expiry_date"] > cur_info["exp"])].copy()
    c["K_dist"] = (c["strike"] - cur_info["K"]).abs() / cur_info["K"]
    c = c[c["K_dist"] <= strike_tol].copy()

    # score prioriza: net_credit + net_yield_ann + liquidez + (bonus R3)
    c["roll_score_side"] = (
        c["net_credit"].fillna(-999) * 20.0
        + c["net_yield_ann_252"].fillna(0) * 120.0
        + c["rule_R3"].fillna(False).astype(int) * 40.0
        + c["liq"].fillna(0)
        - c["K_dist"].fillna(0) * 50.0
    )
    return c.sort_values("roll_score_side", ascending=False)


# =========================
# App
# =========================
st.title("Venda Coberta — Rolagem Profissional (Parquet)")

with st.expander("🔧 Debug (paths / arquivos)", expanded=False):
    st.write("BASE_DIR:", str(BASE_DIR))
    st.write("DATA_DIR:", str(DATA_DIR))
    st.write("assets.parquet:", ASSETS_FP.exists())
    st.write("daily_bars.parquet:", DAILY_FP.exists())
    st.write("option_quote.parquet:", OQ_FP.exists())
    st.write("option_model.parquet:", OM_FP.exists())

if not ASSETS_FP.exists() or not OQ_FP.exists():
    st.error(f"Arquivos obrigatórios ausentes em {DATA_DIR}. Precisa de assets.parquet e option_quote.parquet.")
    st.stop()

# ---- assets
assets = load_assets()
if assets.empty:
    st.error("assets.parquet não tem ativos ativos.")
    st.stop()

ticker = st.sidebar.selectbox("Ativo", assets["ticker"].tolist())
asset_id = int(assets.loc[assets["ticker"] == ticker, "id"].iloc[0])

td = latest_trade_date(asset_id)
if not td:
    st.warning("Sem dados de option_quote para este ativo.")
    st.stop()

calls = load_calls_chain(asset_id, td)
if calls.empty:
    st.warning("Sem CALLs com trades>0 e last_price>0 no último pregão.")
    st.stop()

# ---- daily indicators
ind = load_dailybar_indicators(asset_id, td)
spot = float(ind["close"]) if ind and ind.get("close") is not None else None
trend = trend_label(ind)

calls2 = add_covered_metrics(calls, spot=spot if spot else np.nan)

# ---- Sidebar rules + style
st.sidebar.markdown("## Regras / Filtros")

capture_pct = st.sidebar.slider("R1: Captura mínima do prêmio (%)", 0.10, 0.95, 0.70, 0.01)
use_extrinsic_confirm = st.sidebar.toggle("R1b: exigir extrínseco baixo (confirmação)", value=False)
extrinsic_pct = st.sidebar.slider("R1b: extrínseco ≤ X% do prêmio inicial", 0.01, 0.80, 0.20, 0.01)

delta_th = st.sidebar.slider("R2: Delta gatilho (>)", 0.30, 0.90, 0.65, 0.01)
dte_th   = st.sidebar.slider("R2: DTE gatilho (< dias)", 3, 30, 10, 1)

Y_pp = st.sidebar.slider("R3: melhora mínima de yield anualizado (p.p.)", 0.05, 5.00, 0.50, 0.05)
net_credit_min = st.sidebar.number_input("R3: net_credit mínimo (R$/ação)", value=0.00, step=0.01)

only_credit = st.sidebar.toggle("Somente rolagens com crédito líquido (net_credit ≥ 0)", value=True)

use_trend = st.sidebar.toggle("Usar tendência para filtrar (quando Alta)", value=True)
target_delta_low = st.sidebar.slider("Em Alta: delta máx preferido (nova call)", 0.10, 0.60, 0.30, 0.01)

min_roll_dte = st.sidebar.slider("DTE mínimo (candidatos)", 7, 90, 30, 1)
max_roll_dte = st.sidebar.slider("DTE máximo (candidatos)", 15, 240, 60, 1)

strike_tol = st.sidebar.slider("Rolagem p/ o lado: tolerância de strike (±%)", 0.001, 0.05, 0.01, 0.001)

st.sidebar.markdown("## Payoff (opcional)")
show_pct = st.sidebar.toggle("Payoff em % do spot", value=False)
mult_100 = st.sidebar.toggle("Payoff por 100 ações", value=True)
multiplier = 100.0 if mult_100 else 1.0

# ---- header
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ticker", ticker)
c2.metric("Pregão", str(td))
c3.metric("Spot (close)", f"{spot:.4f}" if spot else "N/A")
c4.metric("Tendência", trend)
c5.metric("Qtd CALLs", str(len(calls2)))

# ---- indicators
with st.expander("Indicadores técnicos (daily_bars) do último pregão", expanded=False):
    if not ind:
        st.info("Sem daily_bars/indicadores para esta data.")
    else:
        st.json({k: ind.get(k) for k in ["trade_date","close","sma_20","sma_50","sma_200","macd_hist","rsi_14","atr_14"]})

# ============================================================
# 1) Cadeia e seleção da posição
# ============================================================
st.subheader("1) Selecionar a CALL atual (vendida)")

# view filters
colf1, colf2, colf3, colf4 = st.columns(4)
with colf1:
    mfilter = st.multiselect("Moneyness (tabela)", ["OTM","ATM","ITM"], default=["OTM","ATM"])
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

pos_symbol = st.selectbox("CALL vendida (option_symbol)", sorted(calls2["option_symbol"].unique().tolist()))

pos_row = calls2[calls2["option_symbol"] == pos_symbol].head(1)
if pos_row.empty:
    st.error("Option_symbol selecionado não encontrado.")
    st.stop()

pos_row = pos_row.iloc[0]
pos_last = float(pos_row["last_price"])
pos_extrinsic = float(pos_row["extrinsic"])
pos_delta = float(pos_row["delta"]) if np.isfinite(pos_row["delta"]) else np.nan
pos_dte = int(pos_row["DTE_days"]) if np.isfinite(pos_row["DTE_days"]) else None
pos_y_ann = float(pos_row["yield_ann_252"]) if np.isfinite(pos_row["yield_ann_252"]) else np.nan

colp1, colp2, colp3, colp4 = st.columns(4)
with colp1:
    entry_premium = st.number_input("Prêmio inicial recebido (R$/ação)", min_value=0.0, value=float(pos_last), step=0.01)
with colp2:
    st.write("**Extrínseco atual**")
    st.write(f"{pos_extrinsic:.2f}")
with colp3:
    st.write("**Delta / DTE**")
    st.write(f"delta={pos_delta:.2f} | DTE={pos_dte}")
with colp4:
    st.write("**Yield anualizado atual**")
    st.write(f"{pos_y_ann:.2%}" if np.isfinite(pos_y_ann) else "N/A")

# ============================================================
# 2) Candidatos base + regras + filtros
# ============================================================
st.subheader("2) Sugestões de rolagem (duas abas)")

cur_info, cand = compute_candidates_base(calls2, pos_symbol, entry_premium)
if cur_info is None:
    st.error("Não consegui identificar a posição atual na cadeia.")
    st.stop()

# filtro DTE candidatos
cand = cand[(cand["DTE_days"] >= min_roll_dte) & (cand["DTE_days"] <= max_roll_dte)].copy()

# filtro tendência (em Alta)
if use_trend and trend == "Alta":
    cand = cand[cand["moneyness"].isin(["OTM","ATM"])].copy()
    cand = cand[(cand["delta"].isna()) | (cand["delta"] <= target_delta_low)].copy()

# aplica regras
flags, cand = apply_rules_and_score(
    cur_info, cand,
    capture_pct=capture_pct,
    use_extrinsic_confirm=use_extrinsic_confirm,
    extrinsic_pct=extrinsic_pct,
    delta_th=delta_th,
    dte_th=dte_th,
    Y_pp=Y_pp,
    net_credit_min=net_credit_min,
)

# filtro crédito líquido
if only_credit:
    cand = cand[cand["net_credit"] >= 0].copy()

# resumo sinal
R1 = flags["R1"]
R2 = flags["R2"]
R3 = flags["R3_any"]
should_roll = R1 or R2 or R3

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("R1 (captura)", "SIM" if R1 else "NÃO")
s2.metric("R2 (delta/DTE)", "SIM" if R2 else "NÃO")
s3.metric("R3 (há melhora)", "SIM" if R3 else "NÃO")
s4.metric("Heurística", "ROLAR" if should_roll else "MANTER")
s5.metric("Filtro crédito líquido", "ON" if only_credit else "OFF")

# ============================================================
# 2b) Abas: UP vs SIDE
# ============================================================
tab_up, tab_side = st.tabs(["⬆️ Rolar para cima (Up)", "➡️ Rolar para o lado (Side)"])

with tab_up:
    up = rank_up(cur_info, cand).head(25).copy()
    if up.empty:
        st.info("Sem candidatos de rolagem PARA CIMA com os filtros atuais.")
    else:
        show_cols = [
            "option_symbol","expiry_date","strike","moneyness","DTE_days",
            "last_price","delta","iv",
            "net_credit","net_yield_ann_252",
            "yield_ann_252","yield_improve","rule_R3",
            "liq"
        ]
        show_cols = [c for c in show_cols if c in up.columns]
        out = up[show_cols].copy()
        for c in ["strike","last_price","net_credit"]:
            if c in out.columns: out[c] = _num(out[c]).round(2)
        for c in ["delta","iv","yield_ann_252","net_yield_ann_252","yield_improve","liq"]:
            if c in out.columns: out[c] = _num(out[c]).round(4)
        st.dataframe(out, width="stretch", hide_index=True, height=520)

with tab_side:
    side = rank_side(cur_info, cand, strike_tol=strike_tol).head(25).copy()
    if side.empty:
        st.info("Sem candidatos de rolagem PARA O LADO com os filtros atuais.")
    else:
        show_cols = [
            "option_symbol","expiry_date","strike","moneyness","DTE_days",
            "last_price","delta","iv",
            "net_credit","net_yield_ann_252",
            "yield_ann_252","yield_improve","rule_R3",
            "liq"
        ]
        show_cols = [c for c in show_cols if c in side.columns]
        out = side[show_cols].copy()
        for c in ["strike","last_price","net_credit"]:
            if c in out.columns: out[c] = _num(out[c]).round(2)
        for c in ["delta","iv","yield_ann_252","net_yield_ann_252","yield_improve","liq"]:
            if c in out.columns: out[c] = _num(out[c]).round(4)
        st.dataframe(out, width="stretch", hide_index=True, height=520)


# ============================================================
# 3) Payoff opcional (escolhe dentre a união dos top candidatos)
# ============================================================
st.subheader("3) Payoff (opcional) para uma rolagem escolhida")

pick_pool = pd.concat([up.head(10), side.head(10)], axis=0, ignore_index=True) if ("up" in locals() and "side" in locals()) else pd.DataFrame()
pick_pool = pick_pool.drop_duplicates(subset=["option_symbol"]) if not pick_pool.empty else pick_pool

if pick_pool.empty:
    st.info("Gere candidatos nas abas acima para visualizar payoff.")
else:
    pick_symbol = st.selectbox("Escolha a nova CALL (sell-to-open) para payoff:", ["(nenhuma)"] + pick_pool["option_symbol"].tolist())
    if pick_symbol != "(nenhuma)":
        new_row = calls2[calls2["option_symbol"] == pick_symbol].head(1)
        if new_row.empty:
            st.warning("Não encontrei essa opção na cadeia completa.")
        else:
            new_row = new_row.iloc[0]
            newK = float(new_row["strike"])
            newP = float(new_row["last_price"])

            net_premium = newP - cur_info["price_close"]
            s0 = float(spot) if spot else float(new_row["spot"])

            lo = max(0.01, 0.5 * s0)
            hi = 1.5 * s0
            ST = np.linspace(lo, hi, 700)

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
            plt.title(f"Payoff rolagem: fechar {cur_info['option_symbol']} e vender {pick_symbol} (net={net_premium:.2f})")
            plt.xlabel("Preço do ativo no vencimento (ST)")
            plt.ylabel("Payoff (% do spot)" if show_pct else "Payoff (P&L no vencimento)")
            st.pyplot(fig, clear_figure=True)

            cbe1, cbe2, cbe3, cbe4 = st.columns(4)
            cbe1.write("**Break-even(s)**")
            cbe1.write(", ".join([f'{v:.2f}' for v in be]) if be else "—")
            cbe2.write("**Net premium (rolagem)**")
            cbe2.write(f"{net_premium:.2f} por ação")
            cbe3.write("**K novo**")
            cbe3.write(f"{newK:.2f}")
            cbe4.write("**Multiplicador**")
            cbe4.write(f"{multiplier:g}")

st.divider()
st.caption(
    "Heurísticas para estudo. Não é recomendação. "
    "Considere custos, impostos, dividendos, risco de exercício e liquidez."
)
