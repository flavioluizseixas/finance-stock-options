import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_CFG, DEFAULT_TOP_N
from features.common import to_float
from features.moneyness import apply_universe_filter, classify_moneyness_multi
from features.pricing.renderer import render_payoff
from features.regime import infer_regime
from finance_options_pipeline import load_cfg, run_market_update
from parquet_store import env_paths, load_table
from strategies import get_strategies
from ui.widgets import format_table, selectable_table


st.set_page_config(page_title="Finance Options Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = env_paths(BASE_DIR)


def _require_file(path: str):
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        st.stop()


def _coerce_dates(df: pd.DataFrame, cols: list[str]):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_assets() -> pd.DataFrame:
    df = load_table("assets", BASE_DIR)
    if df.empty:
        return pd.DataFrame(columns=["id", "ticker"])
    if "is_active" in df.columns:
        df = df[pd.to_numeric(df["is_active"], errors="coerce").fillna(1).astype(int) == 1]
    return df[["id", "ticker"]].dropna().sort_values("ticker").reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def load_option_quote_all() -> pd.DataFrame:
    _require_file(PATHS["PATH_QUOTE"])
    return _coerce_dates(pd.read_parquet(PATHS["PATH_QUOTE"]), ["trade_date", "expiry_date"])


@st.cache_data(ttl=60, show_spinner=False)
def load_option_model_all() -> pd.DataFrame:
    _require_file(PATHS["PATH_MODEL"])
    return _coerce_dates(pd.read_parquet(PATHS["PATH_MODEL"]), ["trade_date"])


@st.cache_data(ttl=60, show_spinner=False)
def load_daily_bars_all() -> pd.DataFrame:
    _require_file(PATHS["PATH_DAILY"])
    return _coerce_dates(pd.read_parquet(PATHS["PATH_DAILY"]), ["trade_date"])


def clear_caches():
    load_assets.clear()
    load_option_quote_all.clear()
    load_option_model_all.clear()
    load_daily_bars_all.clear()


@st.cache_data(ttl=60, show_spinner=False)
def latest_trade_date_by_asset() -> pd.DataFrame:
    oq = load_option_quote_all()
    if oq.empty:
        return pd.DataFrame(columns=["asset_id", "trade_date"])
    out = (
        oq.groupby("asset_id", as_index=False)["trade_date"]
        .max()
        .rename(columns={"trade_date": "latest_trade_date"})
    )
    return out


def load_latest_trade_date(asset_id: int):
    d = latest_trade_date_by_asset()
    row = d.loc[pd.to_numeric(d["asset_id"], errors="coerce") == int(asset_id)]
    return None if row.empty else row.iloc[0]["latest_trade_date"]


def load_daily_indicators(asset_id: int, trade_date: date):
    db = load_daily_bars_all()
    if db.empty:
        return None
    mask = (pd.to_numeric(db["asset_id"], errors="coerce") == int(asset_id)) & (db["trade_date"] == trade_date)
    row = db.loc[mask].tail(1)
    return None if row.empty else row.iloc[0].to_dict()


def load_chain(asset_id: int, trade_date: date) -> pd.DataFrame:
    oq = load_option_quote_all()
    om = load_option_model_all()
    mask_q = (pd.to_numeric(oq["asset_id"], errors="coerce") == int(asset_id)) & (oq["trade_date"] == trade_date)
    q = oq.loc[mask_q].copy()
    if q.empty:
        return pd.DataFrame()

    for col in ["trades", "last_price", "strike", "volume"]:
        if col in q.columns:
            q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q[(q["trades"] > 0) & (q["last_price"] > 0)].copy()
    if q.empty:
        return pd.DataFrame()

    mask_m = (pd.to_numeric(om["asset_id"], errors="coerce") == int(asset_id)) & (om["trade_date"] == trade_date)
    mm = om.loc[mask_m].copy()
    if "collected_at" in q.columns:
        q = q.rename(columns={"collected_at": "quote_collected_at"})
    if "collected_at" in mm.columns:
        mm = mm.rename(columns={"collected_at": "model_collected_at"})

    out = q.merge(mm, how="left", on=["asset_id", "trade_date", "option_symbol"], suffixes=("", "_m"))
    return out.sort_values(["expiry_date", "option_type", "strike"]).reset_index(drop=True)


def build_universe(selected_ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = load_assets()
    rows = []
    meta_rows = []
    iterable = assets.itertuples(index=False)

    for asset in iterable:
        if selected_ticker != "(Todos)" and asset.ticker != selected_ticker:
            continue

        trade_date = load_latest_trade_date(int(asset.id))
        if not trade_date:
            continue

        indicators = load_daily_indicators(int(asset.id), trade_date) or {}
        regime, up_score, down_score, _ = infer_regime(indicators)
        chain = load_chain(int(asset.id), trade_date)
        if chain.empty:
            continue

        close = to_float(indicators.get("close"))
        vol_annual = to_float(indicators.get("vol_annual"))
        if vol_annual is None and "hist_vol_annual" in chain.columns:
            hv_chain = pd.to_numeric(chain["hist_vol_annual"], errors="coerce").dropna()
            hv_chain = hv_chain[hv_chain > 0]
            if not hv_chain.empty:
                vol_annual = float(hv_chain.iloc[-1])
        spot_from_model = pd.to_numeric(chain.get("spot"), errors="coerce").dropna()
        spot_ref = float(spot_from_model.median()) if len(spot_from_model) else close
        if close and spot_ref and abs(float(close) - float(spot_ref)) / float(spot_ref) <= 0.12:
            spot_ref = close

        chain["asset_id"] = int(asset.id)
        chain["ticker"] = asset.ticker
        chain["trade_date"] = trade_date
        chain["spot_ref"] = spot_ref
        chain["hist_vol_annual_ref"] = vol_annual
        chain["regime"] = regime
        chain["regime_up_score"] = up_score
        chain["regime_down_score"] = down_score
        rows.append(chain)
        meta_rows.append(
            {
                "asset_id": int(asset.id),
                "ticker": asset.ticker,
                "trade_date": trade_date,
                "spot_ref": spot_ref,
                "hist_vol_annual_ref": vol_annual,
                "regime": regime,
            }
        )

    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(meta_rows))


def recommendation_score(strategy_key: str, table: pd.DataFrame, regime: str) -> float:
    if table is None or table.empty:
        return float("-inf")
    row = table.iloc[0]
    base = 0.0
    for col, weight in [("cr", 120.0), ("rr", 120.0), ("credit", 40.0), ("max_profit", 30.0), ("liq", 3.0), ("liq_min", 3.0), ("last_price", 8.0)]:
        val = to_float(row.get(col))
        if val is not None and np.isfinite(val):
            base += min(float(val), 100.0) * weight

    alignment = {
        "Alta": {"bull_put_credit_spread": 50, "covered_call": 45, "buy_deep_itm_call": 35, "bull_call_spread": 30},
        "Baixa": {"bear_put_spread": 50, "bear_call_credit_spread": 45, "short_call_condor": 30},
        "Neutra": {"short_call_condor": 45, "short_put_condor": 45, "covered_call": 15, "long_straddle_atm": 10},
        "N/A": {},
    }
    return base + alignment.get(regime, {}).get(strategy_key, 0)


st.title("Opções e Estratégias")

tab_update, tab_recommendation, tab_universe = st.tabs(
    ["Atualização de Dados", "Melhor Estratégia", "Universo Completo"]
)

with tab_update:
    cfg = load_cfg()
    st.subheader("Atualização parquet-first")
    st.caption("Os tickers são lidos do `.env.bsm` e os dados são gravados diretamente em `data/*.parquet`.")
    st.code(", ".join(cfg.tickers), language="text")

    selected = st.multiselect("Tickers para atualizar", options=cfg.tickers, default=cfg.tickers)
    col1, col2, col3 = st.columns(3)

    if col1.button("Atualizar cotações do ativo", width="stretch"):
        with st.spinner("Baixando histórico do Yahoo Finance e recalculando indicadores..."):
            summary = run_market_update(update_quotes=True, update_options=False, selected_tickers=selected)
        clear_caches()
        st.success("Cotações e indicadores atualizados.")
        st.dataframe(pd.DataFrame(summary["tickers"]), width="stretch", hide_index=True)

    if col2.button("Atualizar opções e gregas", width="stretch"):
        with st.spinner("Coletando opções no opcoes.net e recalculando gregas..."):
            summary = run_market_update(update_quotes=True, update_options=True, selected_tickers=selected)
        clear_caches()
        st.success("Cotações, opções e gregas atualizadas.")
        st.dataframe(pd.DataFrame(summary["tickers"]), width="stretch", hide_index=True)

    if col3.button("Recarregar arquivos parquet", width="stretch"):
        clear_caches()
        st.info("Caches locais do Streamlit foram limpos.")

    st.markdown("**Arquivos usados**")
    st.write(PATHS)

with tab_recommendation:
    assets = load_assets()
    if assets.empty:
        st.warning("Nenhum ativo ativo encontrado em `assets.parquet`.")
        st.stop()

    ticker_ui = st.selectbox("Ativo", ["(Todos)"] + assets["ticker"].tolist(), index=0)
    top_n = st.slider("Top N por estratégia", min_value=1, max_value=10, value=DEFAULT_TOP_N)
    show_pct = st.toggle("Exibir payoff em % do spot", value=False)
    multiplier = 100.0 if st.toggle("Payoff por 100 ações", value=True) else 1.0

    df_raw, meta = build_universe(ticker_ui)
    if df_raw.empty:
        st.warning("Sem dados de opções disponíveis para o filtro selecionado.")
        st.stop()

    df2 = classify_moneyness_multi(df_raw.copy(), atm_mode="pct", atm_pct=0.01)
    df2 = apply_universe_filter(
        df2,
        {
            "mny_log_max": 6.0,
            "delta_abs_min": 0.05,
            "last_price_min": 0.01,
            "vol_fin_min": 0.0,
            "opt_contract_mult": 100.0,
        },
    )

    regime = "Neutra"
    if not meta.empty and ticker_ui != "(Todos)":
        regime = str(meta.iloc[0]["regime"])
    elif not meta.empty:
        regime = meta["regime"].mode().iat[0]

    st.write(f"**Regime predominante:** {regime}")

    cfg = dict(DEFAULT_CFG)
    payoff_cfg = {"range_mode": "auto", "lo_mult": 0.5, "hi_mult": 1.5}
    results = []

    for strategy in get_strategies():
        result = strategy.candidates(df2, None, cfg=cfg, top_n=top_n)
        score = recommendation_score(strategy.key, result.table, regime)
        results.append(
            {
                "strategy": strategy,
                "result": result,
                "score": score,
            }
        )

    ranked = sorted(results, key=lambda item: item["score"], reverse=True)
    best = ranked[0]

    st.subheader("Estratégia sugerida")
    if best["result"].table.empty:
        st.info("Nenhuma estratégia elegível no universo atual.")
    else:
        best_row = best["result"].table.iloc[0]
        st.success(f"{best['strategy'].name}")
        st.caption("A sugestão combina regime do ativo com o melhor candidato disponível entre as estratégias modeladas.")
        st.dataframe(best["result"].table.head(3), width="stretch", hide_index=True)
        render_payoff(best["strategy"].payoff_spec(best_row, cfg), multiplier=multiplier, show_pct=show_pct, payoff_cfg=payoff_cfg)

    st.subheader("Ranking das estratégias")
    ranking_df = pd.DataFrame(
        [
            {
                "estrategia": item["strategy"].name,
                "score": (round(item["score"], 2) if np.isfinite(item["score"]) else None),
                "qtd_candidatos": 0 if item["result"].table is None else len(item["result"].table),
            }
            for item in ranked
        ]
    )
    st.dataframe(ranking_df, width="stretch", hide_index=True)

    st.subheader("Explorar candidatos por estratégia")
    for item in ranked:
        strategy = item["strategy"]
        result = item["result"]
        with st.expander(f"{strategy.name} | score={item['score']:.2f}", expanded=False):
            idx = selectable_table(result.table, key=strategy.key, label="Selecione uma linha para visualizar o payoff.")
            if result.criteria:
                st.caption("Critérios: " + " | ".join([c for c in result.criteria if c]))
            if idx is not None and result.table is not None and not result.table.empty:
                spec = strategy.payoff_spec(result.table.iloc[idx], cfg)
                render_payoff(spec, multiplier=multiplier, show_pct=show_pct, payoff_cfg=payoff_cfg)

with tab_universe:
    assets = load_assets()
    if assets.empty:
        st.warning("Nenhum ativo ativo encontrado em `assets.parquet`.")
        st.stop()

    ticker_all = st.selectbox("Ativo para tabela completa", ["(Todos)"] + assets["ticker"].tolist(), index=0, key="ticker_all")
    df_raw, meta = build_universe(ticker_all)
    if df_raw.empty:
        st.warning("Sem dados para montar o universo.")
        st.stop()
    df2 = classify_moneyness_multi(df_raw.copy(), atm_mode="pct", atm_pct=0.01)
    st.dataframe(format_table(df2), width="stretch", height=580, hide_index=True)
