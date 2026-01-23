import os
import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_CFG, DEFAULT_TOP_N
from db.conn import load_env, get_db_config
from db.repo import load_assets, load_latest_trade_date, load_daily_indicators, load_chain

from features.common import to_float
from features.regime import infer_regime
from features.moneyness import classify_moneyness_multi, apply_universe_filter

from ui.sidebar import build_sidebar
from ui.widgets import selectable_table, format_table

from strategies import get_strategies
from features.pricing.renderer import render_payoff

st.set_page_config(page_title="Finance Options Dashboard (Modular — Parquet)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_env(BASE_DIR)
_ = get_db_config()

TOP_N = DEFAULT_TOP_N
st.title("Opções – Última Coleta (Modular — Parquet)")

assets = load_assets()
if assets.empty:
    st.warning("Nenhum ativo em assets. Verifique assets.parquet (is_active=1).")
    st.stop()

def _expiry_list_from_df(df):
    expiry_list = sorted(df["expiry_date"].dropna().unique().tolist()) if df is not None and not df.empty else []
    return [pd.to_datetime(x).date() for x in expiry_list]

default_asset_id = int(assets.iloc[0]["id"])
td0 = load_latest_trade_date(default_asset_id)
df0 = load_chain(default_asset_id, td0) if td0 else pd.DataFrame()
expiry_list0 = _expiry_list_from_df(df0)

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

cfg = dict(DEFAULT_CFG)
cfg["liq_single_filter_hard"] = sb["liq_single_filter_hard"]
cfg["liq_pair_filter_hard"] = sb["liq_pair_filter_hard"]
if cfg["liq_pair_filter_hard"]:
    cfg["liq_pair_hard_min"] = 3.0
    cfg["liq_pair_hard_ratio"] = 0.20

cfg_univ = {
    "mny_log_max": sb["mny_log_max"],
    "delta_abs_min": sb["delta_abs_min"],
    "last_price_min": sb["last_price_min"],
    "vol_fin_min": sb["vol_fin_min"],
    "opt_contract_mult": sb["opt_contract_mult"],
}

df_raw=None
meta=None

if ticker_ui == "(Todos)":
    all_rows=[]
    meta_rows=[]
    for _, a in assets.iterrows():
        asset_id=int(a["id"]); ticker=str(a["ticker"])
        td=load_latest_trade_date(asset_id)
        if not td: 
            continue
        ind=load_daily_indicators(asset_id, td) or {}
        regime, up_score, down_score, reg_details = infer_regime(ind, rsi_hi=sb["rsi_hi"], rsi_lo=sb["rsi_lo"])
        close=to_float(ind.get("close"))
        vol_annual=to_float(ind.get("vol_annual"))
        df_chain=load_chain(asset_id, td)
        if df_chain is None or df_chain.empty:
            continue

        # Prefer spot from option_model (parquet) when daily_bars close is missing or inconsistent
        spot_m = None
        if "spot" in df_chain.columns:
            s2 = pd.to_numeric(df_chain["spot"], errors="coerce").dropna()
            spot_m = float(s2.median()) if len(s2) else None

        if close is None or not np.isfinite(close) or close<=0:
            close = spot_m
        elif spot_m is not None and np.isfinite(spot_m) and spot_m>0:
            if abs(float(close)-float(spot_m))/float(spot_m) > 0.12:
                close = spot_m
        df_chain=df_chain.copy()
        df_chain["asset_id"]=asset_id
        df_chain["ticker"]=ticker
        df_chain["trade_date"]=td
        df_chain["spot_ref"]=close
        df_chain["hist_vol_annual_ref"]=vol_annual
        df_chain["regime"]=regime
        df_chain["regime_up_score"]=up_score
        df_chain["regime_down_score"]=down_score
        all_rows.append(df_chain)
        meta_rows.append({"asset_id":asset_id,"ticker":ticker,"trade_date":td,"spot_ref":close,"hist_vol_annual_ref":vol_annual,"regime":regime})
    df_raw=pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    meta=pd.DataFrame(meta_rows)
    if df_raw.empty:
        st.warning("Sem dados de opções para os ativos ativos.")
        st.stop()
else:
    asset_id=int(assets.loc[assets["ticker"]==ticker_ui, "id"].iloc[0])
    td=load_latest_trade_date(asset_id)
    if not td:
        st.warning("Sem dados em option_quote para este ativo.")
        st.stop()
    ind=load_daily_indicators(asset_id, td) or {}
    regime, up_score, down_score, reg_details = infer_regime(ind, rsi_hi=sb["rsi_hi"], rsi_lo=sb["rsi_lo"])
    close=to_float(ind.get("close"))
    vol_annual=to_float(ind.get("vol_annual"))
    df_chain=load_chain(asset_id, td)
    if df_chain.empty:
        st.warning("Sem opções com trades>0 e last_price>0 para este ativo.")
        st.stop()

    # Prefer spot from option_model (parquet) when daily_bars close is missing or inconsistent
    spot_m = None
    if "spot" in df_chain.columns:
        s2 = pd.to_numeric(df_chain["spot"], errors="coerce").dropna()
        spot_m = float(s2.median()) if len(s2) else None

    if close is None or not np.isfinite(close) or close<=0:
        close = spot_m
    elif spot_m is not None and np.isfinite(spot_m) and spot_m>0:
        if abs(float(close)-float(spot_m))/float(spot_m) > 0.12:
            close = spot_m
    df_raw=df_chain.copy()
    df_raw["asset_id"]=asset_id
    df_raw["ticker"]=ticker_ui
    df_raw["trade_date"]=td
    df_raw["spot_ref"]=close
    df_raw["hist_vol_annual_ref"]=vol_annual
    df_raw["regime"]=regime
    df_raw["regime_up_score"]=up_score
    df_raw["regime_down_score"]=down_score
    meta=pd.DataFrame([{"asset_id":asset_id,"ticker":ticker_ui,"trade_date":td,"spot_ref":close,"hist_vol_annual_ref":vol_annual,"regime":regime}])

df2 = classify_moneyness_multi(df_raw.copy(), atm_mode=sb["atm_mode"], atm_pct=float(sb["atm_pct"]))
if sb["use_universe_filter"]:
    df2 = apply_universe_filter(df2, cfg_univ)

st.subheader(f"Top {TOP_N} por estratégia")

for stg in get_strategies():
    with st.expander(stg.name, expanded=True):
        res = stg.candidates(df2, expiry_sel, cfg=cfg, top_n=TOP_N)
        tbl = res.table
        idx = selectable_table(tbl, key=stg.key, label="Selecione uma operação para ver o payoff.")
        if idx is not None and tbl is not None and not tbl.empty:
            row = tbl.iloc[idx]
            spec = stg.payoff_spec(row, cfg)
            render_payoff(spec, multiplier=multiplier, show_pct=show_pct, payoff_cfg=payoff_cfg)

st.subheader("Tabela completa – universo carregado")
st.dataframe(format_table(df2), use_container_width=True, height=520, hide_index=True)