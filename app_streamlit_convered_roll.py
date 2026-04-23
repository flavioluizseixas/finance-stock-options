import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from finance_options_pipeline import load_cfg, run_market_update
from parquet_store import env_paths, read_parquet_safe, table_path, upsert_parquet, utcnow_ts, write_parquet


st.set_page_config(page_title="Rolagem Coberta", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = env_paths(BASE_DIR)
ASSETS_PATH = table_path("assets", BASE_DIR)
QUOTE_PATH = table_path("option_quote", BASE_DIR)
MODEL_PATH = table_path("option_model", BASE_DIR)
DAILY_PATH = table_path("daily_bars", BASE_DIR)
POSITIONS_PATH = table_path("covered_call_positions", BASE_DIR)

MIN_DTE_DAYS = 180
MAX_DTE_DAYS = 365


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _normalize_id_scalar(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, pd.Timestamp):
        return int(value.value)
    if isinstance(value, np.datetime64):
        return int(pd.Timestamp(value).value)
    try:
        return int(float(value))
    except Exception:
        return np.nan


def _normalize_id_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_id_scalar)


def _safe_float(value, default: float | None = None) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return float(number)


def _require_file(path: str):
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def load_assets() -> pd.DataFrame:
    df = read_parquet_safe(ASSETS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["id", "ticker"])
    if "id" in df.columns:
        df["id"] = _normalize_id_series(df["id"])
    if "is_active" in df.columns:
        df = df[pd.to_numeric(df["is_active"], errors="coerce").fillna(1).astype(int) == 1]
    return df[["id", "ticker"]].dropna().sort_values("ticker").reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def load_option_quote_all() -> pd.DataFrame:
    _require_file(str(QUOTE_PATH))
    df = read_parquet_safe(QUOTE_PATH)
    if df.empty:
        return df
    if "asset_id" in df.columns:
        df["asset_id"] = _normalize_id_series(df["asset_id"])
    df["trade_date"] = _to_date(df["trade_date"])
    df["expiry_date"] = _to_date(df["expiry_date"])
    for col in ["strike", "last_price", "trades", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_option_model_all() -> pd.DataFrame:
    _require_file(str(MODEL_PATH))
    df = read_parquet_safe(MODEL_PATH)
    if df.empty:
        return df
    if "asset_id" in df.columns:
        df["asset_id"] = _normalize_id_series(df["asset_id"])
    df["trade_date"] = _to_date(df["trade_date"])
    for col in ["spot", "delta", "iv", "theta", "t_years", "mispricing_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_daily_bars_all() -> pd.DataFrame:
    _require_file(str(DAILY_PATH))
    df = read_parquet_safe(DAILY_PATH)
    if df.empty:
        return df
    if "asset_id" in df.columns:
        df["asset_id"] = _normalize_id_series(df["asset_id"])
    df["trade_date"] = _to_date(df["trade_date"])
    for col in ["close", "vol_annual"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_positions() -> pd.DataFrame:
    df = read_parquet_safe(POSITIONS_PATH)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "position_id",
                "ticker",
                "asset_id",
                "trade_date",
                "asof_trade_date",
                "option_symbol",
                "option_expiry",
                "option_strike",
                "sale_price",
                "contracts",
                "quantity_multiplier",
                "underlying_entry_price",
                "notes",
                "status",
                "created_at",
                "updated_at",
            ]
        )
    for col in ["position_id", "asset_id", "contracts", "quantity_multiplier"]:
        if col in df.columns:
            df[col] = _normalize_id_series(df[col])
    for col in ["trade_date", "asof_trade_date", "option_expiry"]:
        if col in df.columns:
            df[col] = _to_date(df[col])
    for col in ["created_at", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["option_strike", "sale_price", "underlying_entry_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" not in df.columns:
        df["status"] = "OPEN"
    return df.sort_values(["status", "trade_date", "position_id"], ascending=[True, False, False]).reset_index(drop=True)


def clear_caches():
    load_assets.clear()
    load_option_quote_all.clear()
    load_option_model_all.clear()
    load_daily_bars_all.clear()
    load_positions.clear()


def latest_trade_snapshot(asset_id: int) -> tuple[date | None, float | None, float | None]:
    daily = load_daily_bars_all()
    if daily.empty:
        return None, None, None
    asset_daily = daily.loc[pd.to_numeric(daily["asset_id"], errors="coerce") == int(asset_id)].copy()
    asset_daily = asset_daily.sort_values("trade_date")
    if asset_daily.empty:
        return None, None, None
    row = asset_daily.iloc[-1]
    return row.get("trade_date"), _safe_float(row.get("close")), _safe_float(row.get("vol_annual"))


def load_calls_chain(asset_id: int, trade_date: date | None = None) -> pd.DataFrame:
    quotes = load_option_quote_all()
    models = load_option_model_all()
    if quotes.empty:
        return pd.DataFrame()

    asset_quotes = quotes.loc[
        (pd.to_numeric(quotes["asset_id"], errors="coerce") == int(asset_id)) & (quotes["option_type"] == "CALL")
    ].copy()
    if asset_quotes.empty:
        return pd.DataFrame()

    trade_date = trade_date or asset_quotes["trade_date"].max()
    asset_quotes = asset_quotes.loc[asset_quotes["trade_date"] == trade_date].copy()
    asset_quotes = asset_quotes[(asset_quotes["trades"] > 0) & (asset_quotes["last_price"] > 0)].copy()
    if asset_quotes.empty:
        return pd.DataFrame()

    model_keep = ["asset_id", "trade_date", "option_symbol", "spot", "delta", "iv", "theta", "t_years", "mispricing_pct"]
    asset_models = models.loc[
        (pd.to_numeric(models["asset_id"], errors="coerce") == int(asset_id)) & (models["trade_date"] == trade_date),
        [col for col in model_keep if col in models.columns],
    ].copy()
    merged = asset_quotes.merge(asset_models, how="left", on=["asset_id", "trade_date", "option_symbol"])
    merged["DTE_days"] = (pd.to_datetime(merged["expiry_date"]) - pd.to_datetime(merged["trade_date"])).dt.days
    merged["liq"] = merged["trades"].fillna(0) + np.log1p(merged["volume"].fillna(0))
    return merged.sort_values(["expiry_date", "strike", "option_symbol"]).reset_index(drop=True)


def option_row_from_symbol(calls: pd.DataFrame, option_symbol: str) -> pd.Series | None:
    if calls.empty or not option_symbol:
        return None
    row = calls.loc[calls["option_symbol"] == option_symbol].head(1)
    return None if row.empty else row.iloc[0]


def next_position_id(df: pd.DataFrame) -> int:
    if df.empty or "position_id" not in df.columns:
        return 1
    current = _normalize_id_series(df["position_id"]).max()
    return 1 if pd.isna(current) else int(current) + 1


def lookup_position(df: pd.DataFrame, position_id) -> pd.Series | None:
    if df.empty or "position_id" not in df.columns:
        return None
    normalized_id = _normalize_id_scalar(position_id)
    if pd.isna(normalized_id):
        return None
    row = df.loc[_normalize_id_series(df["position_id"]) == int(normalized_id)].head(1)
    return None if row.empty else row.iloc[0]


def save_position(payload: dict):
    positions = load_positions()
    now = utcnow_ts()
    position_id = next_position_id(positions)
    row = {
        "position_id": position_id,
        "ticker": payload["ticker"],
        "asset_id": int(payload["asset_id"]),
        "trade_date": pd.Timestamp(payload["trade_date"]),
        "asof_trade_date": pd.Timestamp(payload["asof_trade_date"]) if payload.get("asof_trade_date") else pd.NaT,
        "option_symbol": payload["option_symbol"],
        "option_expiry": pd.Timestamp(payload["option_expiry"]),
        "option_strike": float(payload["option_strike"]),
        "sale_price": float(payload["sale_price"]),
        "contracts": int(payload["contracts"]),
        "quantity_multiplier": int(payload.get("quantity_multiplier", 100)),
        "underlying_entry_price": float(payload["underlying_entry_price"]),
        "notes": payload.get("notes", ""),
        "status": str(payload.get("status", "OPEN")).upper(),
        "created_at": now,
        "updated_at": now,
    }
    upsert_parquet(POSITIONS_PATH, pd.DataFrame([row]), ["position_id"], ["position_id"])
    load_positions.clear()


def update_position_status(position_id: int, status: str):
    positions = load_positions()
    if positions.empty:
        return
    mask = _normalize_id_series(positions["position_id"]) == int(position_id)
    if not mask.any():
        return
    positions.loc[mask, "status"] = str(status).upper()
    positions.loc[mask, "updated_at"] = utcnow_ts()
    upsert_parquet(POSITIONS_PATH, positions, ["position_id"], ["position_id"])
    load_positions.clear()


def delete_position(position_id: int):
    positions = load_positions()
    if positions.empty:
        return
    positions = positions.loc[_normalize_id_series(positions["position_id"]) != int(position_id)].copy()
    write_parquet(POSITIONS_PATH, positions)
    load_positions.clear()


def equivalent_monthly_rate(period_return: float, dte_days: float) -> float | None:
    if not np.isfinite(period_return) or not np.isfinite(dte_days) or dte_days <= 0 or period_return <= -0.9999:
        return None
    try:
        return float((1.0 + float(period_return)) ** (30.4375 / float(dte_days)) - 1.0)
    except Exception:
        return None


def score_delta(delta: float) -> float:
    if not np.isfinite(delta):
        return np.nan
    if 0.25 <= delta <= 0.35:
        return 1.0
    if 0.15 <= delta < 0.25:
        return 0.75
    if 0.35 < delta <= 0.50:
        return 0.55
    if 0.50 < delta <= 0.60:
        return 0.20
    return -0.40


def score_monthly_yield(monthly_yield: float) -> float:
    if not np.isfinite(monthly_yield):
        return np.nan
    if monthly_yield >= 0.02:
        return 1.0
    if monthly_yield >= 0.01:
        return 0.75
    if monthly_yield >= 0.008:
        return 0.45
    if monthly_yield >= 0.0:
        return 0.10
    return -1.0


def build_roll_candidates(position: pd.Series) -> tuple[pd.DataFrame, dict]:
    asset_id = int(position["asset_id"])
    latest_trade_date, spot_ref, hist_vol = latest_trade_snapshot(asset_id)
    calls = load_calls_chain(asset_id, latest_trade_date)
    if calls.empty:
        return pd.DataFrame(), {"trade_date": latest_trade_date, "spot_ref": spot_ref, "hist_vol": hist_vol}

    current_row = option_row_from_symbol(calls, str(position.get("option_symbol", "")))
    current_mark = _safe_float(position.get("sale_price"), 0.0)
    current_mark_source = "preco_registrado"
    current_dte = None
    if current_row is not None:
        marked = _safe_float(current_row.get("last_price"))
        if marked is not None:
            current_mark = marked
            current_mark_source = "mercado_atual"
        current_dte = _safe_float(current_row.get("DTE_days"))

    spot_ref = _safe_float(spot_ref, _safe_float(position.get("underlying_entry_price"), 0.0))
    entry_spot = _safe_float(position.get("underlying_entry_price"), spot_ref or 0.0)
    current_strike = _safe_float(position.get("option_strike"), 0.0)
    contracts = int(_safe_float(position.get("contracts"), 1) or 1)
    multiplier = int(_safe_float(position.get("quantity_multiplier"), 100) or 100)

    candidates = calls.copy()
    candidates = candidates[
        (candidates["DTE_days"] >= MIN_DTE_DAYS)
        & (candidates["DTE_days"] <= MAX_DTE_DAYS)
        & (candidates["expiry_date"] > pd.to_datetime(position["option_expiry"]).date())
    ].copy()
    for col in ["delta", "iv", "theta", "last_price", "strike", "trades", "volume", "DTE_days"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    before_model_filter = len(candidates)
    candidates = candidates.dropna(subset=["delta", "iv"]).copy()
    candidates = candidates[(candidates["iv"] > 0) & (candidates["delta"] > 0) & (candidates["delta"] < 1)].copy()
    removed_missing_model = before_model_filter - len(candidates)
    if candidates.empty:
        return pd.DataFrame(), {
            "trade_date": latest_trade_date,
            "spot_ref": spot_ref,
            "hist_vol": hist_vol,
            "current_mark": current_mark,
            "current_mark_source": current_mark_source,
            "current_dte": current_dte,
            "removed_missing_model": removed_missing_model,
        }

    candidates["spot_ref"] = float(spot_ref or 0.0)
    candidates["entry_spot"] = float(entry_spot or 0.0)
    candidates["net_credit"] = candidates["last_price"] - float(current_mark or 0.0)
    candidates["cash_in_roll"] = candidates["net_credit"] * contracts * multiplier
    candidates["strike_gain"] = candidates["strike"] - float(current_strike or 0.0)
    candidates["strike_gain_pct"] = np.where(
        candidates["spot_ref"] > 0,
        candidates["strike_gain"] / candidates["spot_ref"],
        np.nan,
    )
    candidates["yield_on_spot"] = np.where(
        candidates["spot_ref"] > 0,
        candidates["net_credit"] / candidates["spot_ref"],
        np.nan,
    )
    candidates["yield_on_entry_spot"] = np.where(
        candidates["entry_spot"] > 0,
        candidates["net_credit"] / candidates["entry_spot"],
        np.nan,
    )
    candidates["assignment_upside_pct"] = np.where(
        candidates["spot_ref"] > 0,
        (candidates["strike"] - candidates["spot_ref"]) / candidates["spot_ref"],
        np.nan,
    )
    candidates["months_to_expiry"] = candidates["DTE_days"] / 30.4375
    candidates["monthly_simple_yield"] = np.where(
        candidates["months_to_expiry"] > 0,
        candidates["yield_on_spot"] / candidates["months_to_expiry"],
        np.nan,
    )
    candidates["monthly_equiv_yield"] = [
        equivalent_monthly_rate(period_return, dte_days)
        for period_return, dte_days in zip(candidates["yield_on_spot"], candidates["DTE_days"])
    ]
    candidates["delta_score"] = candidates["delta"].map(score_delta)
    candidates["monthly_yield_score"] = candidates["monthly_equiv_yield"].map(score_monthly_yield)
    candidates["iv_score"] = candidates["iv"].rank(pct=True, method="average")
    candidates["liq_score"] = candidates["liq"].rank(pct=True, method="average")
    candidates["strike_score"] = np.select(
        [
            candidates["assignment_upside_pct"] >= 0.10,
            candidates["assignment_upside_pct"] >= 0.03,
            candidates["assignment_upside_pct"] >= 0.0,
            candidates["assignment_upside_pct"] < 0.0,
        ],
        [1.0, 0.75, 0.40, -0.60],
        default=0.0,
    )
    candidates["credit_score"] = np.where(candidates["net_credit"] >= 0, 0.50, -1.0)
    candidates["score"] = (
        candidates["delta_score"].fillna(-1.0) * 30.0
        + candidates["iv_score"].fillna(0.0) * 20.0
        + candidates["monthly_yield_score"].fillna(-1.0) * 20.0
        + candidates["liq_score"].fillna(0.0) * 15.0
        + candidates["strike_score"].fillna(-1.0) * 10.0
        + candidates["credit_score"].fillna(-1.0) * 5.0
    )
    candidates["score"] = np.where(candidates["delta"] >= 0.60, candidates["score"] - 25.0, candidates["score"])
    candidates = candidates.sort_values(["score", "net_credit", "liq"], ascending=[False, False, False]).reset_index(drop=True)

    context = {
        "trade_date": latest_trade_date,
        "spot_ref": spot_ref,
        "hist_vol": hist_vol,
        "current_mark": current_mark,
        "current_mark_source": current_mark_source,
        "current_dte": current_dte,
        "removed_missing_model": removed_missing_model,
    }
    return candidates, context


st.title("Rolagem de Venda Coberta")
st.caption("Atualiza os parquets, registra vendas cobertas em parquet e sugere rolagens longas de 6 a 12 meses.")

tab_update, tab_register, tab_roll = st.tabs(
    ["Atualização de Dados", "Cadastro da Venda Coberta", "Rolagens de Longo Prazo"]
)

with tab_update:
    cfg = load_cfg()
    st.subheader("Atualização parquet-first")
    st.caption("Os tickers são lidos do `.env.bsm` e os dados são gravados diretamente em `data/*.parquet`.")
    st.code(", ".join(cfg.tickers), language="text")

    selected = st.multiselect("Tickers para atualizar", options=cfg.tickers, default=cfg.tickers)
    col1, col2, col3 = st.columns(3)

    if col1.button("Atualizar cotações do ativo", use_container_width=True):
        with st.spinner("Baixando histórico do Yahoo Finance e recalculando indicadores..."):
            summary = run_market_update(update_quotes=True, update_options=False, selected_tickers=selected)
        clear_caches()
        st.success("Cotações e indicadores atualizados.")
        st.dataframe(pd.DataFrame(summary["tickers"]), use_container_width=True, hide_index=True)

    if col2.button("Atualizar opções e gregas", use_container_width=True):
        with st.spinner("Coletando opções e recalculando gregas..."):
            summary = run_market_update(update_quotes=True, update_options=True, selected_tickers=selected)
        clear_caches()
        st.success("Cotações, opções e gregas atualizadas.")
        st.dataframe(pd.DataFrame(summary["tickers"]), use_container_width=True, hide_index=True)

    if col3.button("Recarregar arquivos parquet", use_container_width=True):
        clear_caches()
        st.info("Caches locais do Streamlit foram limpos.")

    st.markdown("**Arquivos usados**")
    st.write(PATHS)

with tab_register:
    assets = load_assets()
    if assets.empty:
        st.warning("Nenhum ativo ativo encontrado em `assets.parquet`.")
        st.stop()

    st.subheader("Registrar venda coberta")
    ticker = st.selectbox("Ativo subjacente", assets["ticker"].tolist(), key="register_ticker")
    asset_id = int(assets.loc[assets["ticker"] == ticker, "id"].iloc[0])
    asof_trade_date, latest_spot, hist_vol = latest_trade_snapshot(asset_id)
    calls = load_calls_chain(asset_id, asof_trade_date)

    st.caption(
        f"Pregão base: {asof_trade_date or 'N/A'} | Spot atual: {latest_spot if latest_spot is not None else 'N/A'} | Vol anual: {hist_vol if hist_vol is not None else 'N/A'}"
    )

    if calls.empty:
        st.info("Sem CALLs líquidas disponíveis para este ativo no parquet atual. O cadastro é manual.")

    with st.form("covered_call_register_form", clear_on_submit=False):
        option_symbol = st.text_input(
            "Option symbol",
            placeholder="Ex.: ITUBD429W4",
            help="Pode ser digitado livremente, mesmo que a opção não esteja no parquet atual.",
        )

        col0, col1, col2 = st.columns(3)
        option_expiry = col0.date_input("Vencimento da opção", value=asof_trade_date or date.today())
        trade_date = col1.date_input("Data da venda", value=date.today())
        contracts = col2.number_input("Contratos", min_value=1, value=1, step=1)

        col3, col4, col5 = st.columns(3)
        option_strike = col3.number_input("Strike", min_value=0.0, value=0.0, step=0.01)
        quantity_multiplier = col4.number_input("Multiplicador", min_value=1, value=100, step=1)
        sale_price = col5.number_input(
            "Preço de venda da opção",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )

        underlying_entry_price = st.number_input(
            "Preço do ativo subjacente",
            min_value=0.0,
            value=float(latest_spot or 0.0),
            step=0.01,
        )

        notes = st.text_area("Notas", placeholder="Ex.: venda coberta para gerar renda e aceitar rolagem para vencimentos longos.")
        submitted = st.form_submit_button("Salvar venda coberta", type="primary", use_container_width=True)

        if submitted:
            option_symbol_clean = str(option_symbol).strip().upper()
            if not option_symbol_clean:
                st.warning("Informe o option symbol da CALL vendida.")
            else:
                save_position(
                    {
                        "ticker": ticker,
                        "asset_id": asset_id,
                        "trade_date": trade_date,
                        "asof_trade_date": asof_trade_date,
                        "option_symbol": option_symbol_clean,
                        "option_expiry": option_expiry,
                        "option_strike": float(option_strike),
                        "sale_price": float(sale_price),
                        "contracts": int(contracts),
                        "quantity_multiplier": int(quantity_multiplier),
                        "underlying_entry_price": float(underlying_entry_price),
                        "notes": notes,
                        "status": "OPEN",
                    }
                )
                st.success("Venda coberta registrada em parquet.")
                st.rerun()

    st.subheader("Posições registradas")
    positions = load_positions()
    if positions.empty:
        st.info("Ainda não há vendas cobertas registradas.")
    else:
        st.dataframe(
            positions[
                [
                    "position_id",
                    "ticker",
                    "trade_date",
                    "option_symbol",
                    "option_expiry",
                    "option_strike",
                    "sale_price",
                    "contracts",
                    "underlying_entry_price",
                    "status",
                    "notes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        manage_col1, manage_col2, manage_col3 = st.columns(3)
        selected_position_id = manage_col1.selectbox(
            "Posição para gestão",
            positions["position_id"].tolist(),
            format_func=lambda position_id: (
                lambda row: (
                    f"#{int(row['position_id'])} | {row['ticker']} | {row['option_symbol']} | {row['status']}"
                    if row is not None
                    else str(position_id)
                )
            )(lookup_position(positions, position_id)),
        )
        selected_position = lookup_position(positions, selected_position_id)
        if selected_position is None:
            st.warning("Não foi possível localizar a posição selecionada.")
            st.stop()
        next_status = "CLOSED" if str(selected_position["status"]).upper() == "OPEN" else "OPEN"

        if manage_col2.button(f"Marcar como {next_status}", use_container_width=True):
            update_position_status(int(selected_position_id), next_status)
            st.success("Status atualizado.")
            st.rerun()

        confirm_delete = manage_col3.checkbox("Confirmar exclusão", key="confirm_delete_position")
        if st.button("Excluir posição selecionada", disabled=not confirm_delete, use_container_width=True):
            delete_position(int(selected_position_id))
            st.success("Posição removida do parquet.")
            st.rerun()

with tab_roll:
    positions = load_positions()
    open_positions = positions.loc[positions["status"].astype(str).str.upper() == "OPEN"].copy()
    if open_positions.empty:
        st.info("Registre ao menos uma venda coberta aberta para calcular as rolagens.")
        st.stop()

    st.subheader("Melhores rolagens de longo prazo")
    selected_position_id = st.selectbox(
        "Selecione a venda coberta",
        open_positions["position_id"].tolist(),
        format_func=lambda position_id: (
            lambda row: (
                f"#{int(row['position_id'])} | {row['ticker']} | {row['option_symbol']} | strike {row['option_strike']:.2f} | venc {row['option_expiry']}"
                if row is not None
                else str(position_id)
            )
        )(lookup_position(open_positions, position_id)),
    )
    position = lookup_position(open_positions, selected_position_id)
    if position is None:
        st.warning("Não foi possível localizar a venda coberta selecionada.")
        st.stop()
    candidates, context = build_roll_candidates(position)

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Spot de referência", f"{context.get('spot_ref'):.2f}" if context.get("spot_ref") else "N/A")
    info_col2.metric("Mark atual da CALL", f"{context.get('current_mark'):.2f}" if context.get("current_mark") is not None else "N/A")
    info_col3.metric("Fonte do mark", str(context.get("current_mark_source", "N/A")).replace("_", " ").title())
    info_col4.metric("Pregão analisado", str(context.get("trade_date") or "N/A"))

    st.caption(
        f"Filtro aplicado: CALLs com vencimento entre {MIN_DTE_DAYS} e {MAX_DTE_DAYS} dias, posteriores ao vencimento atual, com Delta e IV válidos."
    )
    if context.get("removed_missing_model"):
        st.info(f"{int(context['removed_missing_model'])} opções foram removidas por falta de Delta ou IV.")
    st.caption(
        "Score: prioriza Delta 0.25-0.35, IV relativa alta, retorno mensal equivalente acima de 0.8%-2%, liquidez, strike acima do spot e crédito líquido positivo."
    )

    if candidates.empty:
        st.warning("Não encontrei candidatos longos de rolagem para essa posição com o parquet atual.")
    else:
        display = candidates[
            [
                "option_symbol",
                "expiry_date",
                "DTE_days",
                "strike",
                "last_price",
                "net_credit",
                "cash_in_roll",
                "yield_on_spot",
                "monthly_simple_yield",
                "monthly_equiv_yield",
                "assignment_upside_pct",
                "strike_gain_pct",
                "delta",
                "iv",
                "delta_score",
                "monthly_yield_score",
                "iv_score",
                "liq_score",
                "liq",
                "score",
            ]
        ].copy()

        pct_cols = [
            "yield_on_spot",
            "monthly_simple_yield",
            "monthly_equiv_yield",
            "assignment_upside_pct",
            "strike_gain_pct",
            "iv",
        ]
        for col in pct_cols:
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "expiry_date": st.column_config.DateColumn("Vencimento", format="YYYY-MM-DD"),
                "DTE_days": st.column_config.NumberColumn("Dias até venc.", format="%d"),
                "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "last_price": st.column_config.NumberColumn("Prêmio atual", format="%.2f"),
                "net_credit": st.column_config.NumberColumn("Crédito líquido", format="%.2f"),
                "cash_in_roll": st.column_config.NumberColumn("Caixa na rolagem", format="%.2f"),
                "yield_on_spot": st.column_config.NumberColumn("Retorno / spot", format="%.2f%%"),
                "monthly_simple_yield": st.column_config.NumberColumn("Juro mensal simples", format="%.2f%%"),
                "monthly_equiv_yield": st.column_config.NumberColumn("Juro mensal equivalente", format="%.2f%%"),
                "assignment_upside_pct": st.column_config.NumberColumn("Upside até strike", format="%.2f%%"),
                "strike_gain_pct": st.column_config.NumberColumn("Ganho de strike", format="%.2f%%"),
                "delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                "iv": st.column_config.NumberColumn("IV", format="%.2f%%"),
                "delta_score": st.column_config.NumberColumn("Score Delta", format="%.2f"),
                "monthly_yield_score": st.column_config.NumberColumn("Score Retorno", format="%.2f"),
                "iv_score": st.column_config.NumberColumn("Score IV", format="%.2f"),
                "liq_score": st.column_config.NumberColumn("Score Liq.", format="%.2f"),
                "liq": st.column_config.NumberColumn("Liquidez", format="%.2f"),
                "score": st.column_config.NumberColumn("Score", format="%.2f"),
            },
        )

        best = candidates.iloc[0]
        st.markdown("**Melhor candidata pela pontuação atual**")
        st.write(
            {
                "nova_call": best["option_symbol"],
                "vencimento": best["expiry_date"],
                "strike": round(float(best["strike"]), 2),
                "premio": round(float(best["last_price"]), 2),
                "credito_liquido": round(float(best["net_credit"]), 2),
                "juros_mensal_simples_pct": round(float(best["monthly_simple_yield"]) * 100.0, 3)
                if pd.notna(best["monthly_simple_yield"])
                else None,
                "juros_mensal_equivalente_pct": round(float(best["monthly_equiv_yield"]) * 100.0, 3)
                if pd.notna(best["monthly_equiv_yield"])
                else None,
            }
        )
