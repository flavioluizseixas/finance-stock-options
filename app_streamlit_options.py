import os
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from config import DEFAULT_CFG, DEFAULT_TOP_N
from db.conn import load_env, get_db_config
from db.repo import load_assets, load_latest_trade_date, load_daily_indicators, load_chain

from features.common import to_float
from features.regime import infer_regime
from features.moneyness import classify_moneyness_multi, apply_universe_filter

from ui.sidebar import build_sidebar
from ui.widgets import selectable_table, format_table, criteria_box

from strategies import get_strategies
from features.pricing.renderer import render_payoff

st.set_page_config(page_title="Finance Options Dashboard (Modular)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_env(BASE_DIR)
host, port, user, pwd, name = get_db_config()
if not pwd:
    st.error("DB_PASSWORD/DB_PASS não definido no .env")
    st.stop()

TOP_N = DEFAULT_TOP_N
st.title("Opções – Última Coleta (Modular)")

assets = load_assets()
if assets.empty:
    st.warning("Nenhum ativo em assets. Rode o pipeline primeiro.")
    st.stop()

# Carregamento (Todos) ou ticker específico
ticker_ui = None
df_raw = None
meta = None

# primeiro, precisamos de expiry_list para o sidebar, mas depende do modo.
# Solução: carregar um ticker "default" para capturar expiries, e o sidebar oferece (Todos) como antes.
# Se o usuário escolher (Todos), recalculamos depois.
def _expiry_list_from_df(df):
    expiry_list = sorted(df["expiry_date"].dropna().unique().tolist()) if df is not None and not df.empty else []
    return [pd.to_datetime(x).date() for x in expiry_list]

# Carrega um ticker default para obter expiries iniciais
default_asset_id = int(assets.iloc[0]["id"])
td0 = load_latest_trade_date(default_asset_id)
df0 = load_chain(default_asset_id, td0) if td0 else pd.DataFrame()
expiry_list0 = _expiry_list_from_df(df0)

# Sidebar controls
sb = build_sidebar(assets, expiry_list0)
ticker_ui = sb["ticker_ui"]
expiry_sel = sb["expiry_sel"]
show_pct = sb["show_pct"]
multiplier = sb["multiplier"]
payoff_cfg = {
    "range_mode": ("manual" if sb.get("payoff_range_mode")=="manual" else "auto"),
    "lo_mult": float(sb.get("payoff_lo_mult", 0.5)),
    "hi_mult": float(sb.get("payoff_hi_mult", 1.5)),
}

# Config runtime (merge defaults + flags do sidebar)
cfg = dict(DEFAULT_CFG)
cfg["liq_single_filter_hard"] = sb["liq_single_filter_hard"]
if sb["liq_pair_filter_hard"]:
    cfg["liq_pair_hard_min"] = 3.0
    cfg["liq_pair_hard_ratio"] = 0.20
else:
    cfg["liq_pair_hard_min"] = 0.0
    cfg["liq_pair_hard_ratio"] = 0.0

# Universo
cfg_univ = {
    "mny_log_max": sb["mny_log_max"],
    "delta_abs_min": sb["delta_abs_min"],
    "last_price_min": sb["last_price_min"],
    "vol_fin_min": sb["vol_fin_min"],
    "opt_contract_mult": sb["opt_contract_mult"],
}

# ===== Load data =====
if ticker_ui == "(Todos)":
    all_rows = []
    meta_rows = []
    for _, a in assets.iterrows():
        asset_id = int(a["id"])
        ticker = str(a["ticker"])
        td = load_latest_trade_date(asset_id)
        if not td:
            continue
        ind = load_daily_indicators(asset_id, td) or {}
        regime, up_score, down_score, reg_details = infer_regime(ind, rsi_hi=sb["rsi_hi"], rsi_lo=sb["rsi_lo"])

        close = to_float(ind.get("close"))
        vol_annual = to_float(ind.get("vol_annual"))

        df_chain = load_chain(asset_id, td)
        if df_chain is None or df_chain.empty:
            continue

        if close is None or not np.isfinite(close) or close <= 0:
            s2 = pd.to_numeric(df_chain.get("spot", np.nan), errors="coerce").dropna()
            close = float(s2.iloc[0]) if len(s2) else None

        df_chain = df_chain.copy()
        df_chain["asset_id"] = asset_id
        df_chain["ticker"] = ticker
        df_chain["trade_date"] = td
        df_chain["spot_ref"] = close
        df_chain["hist_vol_annual_ref"] = vol_annual
        df_chain["regime"] = regime
        df_chain["regime_up_score"] = up_score
        df_chain["regime_down_score"] = down_score
        all_rows.append(df_chain)

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

    df_raw = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    meta = pd.DataFrame(meta_rows)
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
    regime, up_score, down_score, reg_details = infer_regime(ind, rsi_hi=sb["rsi_hi"], rsi_lo=sb["rsi_lo"])
    close = to_float(ind.get("close"))
    vol_annual = to_float(ind.get("vol_annual"))
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

# Atualiza expiry_list real (para UI)
expiry_list = sorted(df_raw["expiry_date"].dropna().unique().tolist())
expiry_list = [pd.to_datetime(x).date() for x in expiry_list]

# Moneyness
df2 = classify_moneyness_multi(df_raw.copy(), atm_mode=sb["atm_mode"], atm_pct=float(sb["atm_pct"]))

# Universe filter
if sb["use_universe_filter"]:
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

st.subheader(f"Top {TOP_N} por estratégia")

strategies = get_strategies()

# Render: cada estratégia em um expander
for stg in strategies:
    with st.expander(f"{stg.name}", expanded=(stg.key in ["buy_deep_itm_call","sell_put","booster_puts"])):
        res = stg.candidates(df2, expiry_sel, cfg=cfg, top_n=TOP_N)
        tbl = res.table
        idx = selectable_table(format_table(tbl), key=stg.key, label="Selecione uma operação para ver o payoff.")
        if idx is not None and tbl is not None and not tbl.empty:
            row = tbl.iloc[idx]
            spec = stg.payoff_spec(row, cfg)
            render_payoff(spec, multiplier=multiplier, show_pct=show_pct, payoff_cfg=payoff_cfg)

st.subheader("Tabela completa – universo carregado")
st.dataframe(format_table(df2), width="stretch", height=520, hide_index=True)
