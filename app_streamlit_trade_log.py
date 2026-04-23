import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from parquet_store import env_paths, read_parquet_safe, table_path, upsert_parquet, utcnow_ts, write_parquet


st.set_page_config(page_title="Trade Log Estruturado", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = env_paths(BASE_DIR)
TRADES_PATH = table_path("structured_trades", BASE_DIR)
LEGS_PATH = table_path("structured_trade_legs", BASE_DIR)
ASSETS_PATH = table_path("assets", BASE_DIR)
QUOTE_PATH = table_path("option_quote", BASE_DIR)
DAILY_PATH = table_path("daily_bars", BASE_DIR)


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


def load_assets() -> pd.DataFrame:
    df = read_parquet_safe(ASSETS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["id", "ticker"])
    if "id" in df.columns:
        df["id"] = _normalize_id_series(df["id"])
    if "is_active" in df.columns:
        df = df[pd.to_numeric(df["is_active"], errors="coerce").fillna(1).astype(int) == 1]
    return df[["id", "ticker"]].sort_values("ticker").reset_index(drop=True)


def load_quotes(asset_id: int) -> pd.DataFrame:
    df = read_parquet_safe(QUOTE_PATH)
    if df.empty:
        return df
    if "asset_id" in df.columns:
        df["asset_id"] = _normalize_id_series(df["asset_id"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
    for col in ["strike", "last_price", "trades", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[(df["asset_id"] == int(asset_id)) & (df["trades"] > 0) & (df["last_price"] > 0)].copy()


def latest_spot(asset_id: int):
    df = read_parquet_safe(DAILY_PATH)
    if df.empty:
        return None, None
    if "asset_id" in df.columns:
        df["asset_id"] = _normalize_id_series(df["asset_id"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df = df[df["asset_id"] == int(asset_id)].sort_values("trade_date")
    if df.empty:
        return None, None
    row = df.iloc[-1]
    return row.get("trade_date"), row.get("close")


def load_trade_tables():
    trades = read_parquet_safe(TRADES_PATH)
    legs = read_parquet_safe(LEGS_PATH)
    if not trades.empty:
        for col in ["trade_id", "asset_id", "quantity_multiplier"]:
            if col in trades.columns:
                trades[col] = _normalize_id_series(trades[col])
        for col in ["trade_date", "asof_trade_date", "closed_date", "created_at", "updated_at"]:
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col], errors="coerce")
    if not legs.empty:
        for col in ["leg_id", "trade_id", "leg_no", "contracts"]:
            if col in legs.columns:
                legs[col] = _normalize_id_series(legs[col])
        for col in ["expiry", "created_at", "updated_at"]:
            if col in legs.columns:
                legs[col] = pd.to_datetime(legs[col], errors="coerce")
        for col in ["strike", "entry_price", "exit_price", "ref_last_price"]:
            if col in legs.columns:
                legs[col] = pd.to_numeric(legs[col], errors="coerce")
    return trades, legs


def next_id(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 1
    val = _normalize_id_series(df[col]).max()
    return 1 if pd.isna(val) else int(val) + 1


def infer_defaults(quotes: pd.DataFrame, spot: float | None, opt_type: str):
    if quotes.empty:
        return None, pd.DataFrame(), None
    today = pd.Timestamp.today().date()
    quotes = quotes.copy()
    quotes["strike_dist"] = (pd.to_numeric(quotes["strike"], errors="coerce") - float(spot or 0.0)).abs()
    expiries = sorted([d for d in quotes["expiry_date"].dropna().unique().tolist() if d >= today] or quotes["expiry_date"].dropna().unique().tolist())
    default_expiry = expiries[0] if expiries else None
    subset = quotes[(quotes["expiry_date"] == default_expiry) & (quotes["option_type"] == opt_type)].copy()
    if subset.empty:
        return default_expiry, subset, None
    subset = subset.sort_values(["strike_dist", "trades", "volume"], ascending=[True, False, False])
    default_symbol = subset.iloc[0]["option_symbol"]
    return default_expiry, subset, default_symbol


def direction_from_legs(legs: list[dict], multiplier: int = 100) -> str:
    net_cash = 0.0
    for leg in legs:
        side = str(leg.get("side", "")).upper()
        sign = 1.0 if side == "SELL" else -1.0
        net_cash += sign * float(leg.get("entry_price", 0.0)) * float(leg.get("contracts", 0)) * multiplier
    return "CREDIT" if net_cash > 0 else "DEBIT"


def calc_leg_pnl(leg: pd.Series, multiplier: int) -> float | None:
    if pd.isna(leg.get("exit_price")):
        return None
    entry = float(leg.get("entry_price", 0.0))
    exit_price = float(leg.get("exit_price", 0.0))
    contracts = float(leg.get("contracts", 0))
    if str(leg.get("side", "")).upper() == "SELL":
        return (entry - exit_price) * contracts * multiplier
    return (exit_price - entry) * contracts * multiplier


def save_trade(header: dict, legs_payload: list[dict]):
    trades, legs = load_trade_tables()
    trade_id = next_id(trades, "trade_id")
    now = utcnow_ts()
    multiplier = int(header.get("quantity_multiplier", 100))
    direction = direction_from_legs(legs_payload, multiplier)

    trade_row = {
        "trade_id": trade_id,
        "trade_date": pd.Timestamp(header["trade_date"]),
        "asof_trade_date": pd.Timestamp(header["asof_trade_date"]) if header.get("asof_trade_date") else pd.NaT,
        "strategy": header["strategy"],
        "asset_id": int(header["asset_id"]),
        "ticker": header["ticker"],
        "direction": direction,
        "status": "OPEN",
        "quantity_multiplier": multiplier,
        "fees": float(header.get("fees", 0.0)),
        "underlying_spot": float(header.get("underlying_spot")) if header.get("underlying_spot") is not None else np.nan,
        "thesis": header.get("thesis", ""),
        "tags": header.get("tags", ""),
        "notes": header.get("notes", ""),
        "created_at": now,
        "updated_at": now,
    }

    leg_rows = []
    next_leg_id = next_id(legs, "leg_id")
    for idx, leg in enumerate(legs_payload, start=1):
        leg_rows.append(
            {
                "leg_id": next_leg_id,
                "trade_id": trade_id,
                "leg_no": idx,
                "side": str(leg["side"]).upper(),
                "opt_type": str(leg["opt_type"]).upper(),
                "expiry": pd.Timestamp(leg["expiry"]),
                "strike": float(leg["strike"]),
                "option_symbol": leg["option_symbol"],
                "contracts": int(leg["contracts"]),
                "entry_price": float(leg["entry_price"]),
                "exit_price": np.nan,
                "ref_last_price": float(leg["ref_last_price"]),
                "created_at": now,
                "updated_at": now,
            }
        )
        next_leg_id += 1

    upsert_parquet(TRADES_PATH, pd.DataFrame([trade_row]), ["trade_id"], ["trade_id"])
    upsert_parquet(LEGS_PATH, pd.DataFrame(leg_rows), ["leg_id"], ["trade_id", "leg_no"])


def close_trade(trade_id: int, exit_prices: dict[str, float]):
    trades, legs = load_trade_tables()
    now = utcnow_ts()
    leg_mask = pd.to_numeric(legs["trade_id"], errors="coerce") == int(trade_id)
    for idx in legs[leg_mask].index:
        symbol = legs.at[idx, "option_symbol"]
        if symbol in exit_prices:
            legs.at[idx, "exit_price"] = float(exit_prices[symbol])
            legs.at[idx, "updated_at"] = now
    trade_idx = trades.index[pd.to_numeric(trades["trade_id"], errors="coerce") == int(trade_id)]
    if len(trade_idx):
        trades.at[trade_idx[0], "status"] = "CLOSED"
        trades.at[trade_idx[0], "closed_date"] = now
        trades.at[trade_idx[0], "updated_at"] = now
    upsert_parquet(TRADES_PATH, trades, ["trade_id"], ["trade_id"])
    upsert_parquet(LEGS_PATH, legs, ["leg_id"], ["trade_id", "leg_no"])


def update_trade(trade_id: int, header_updates: dict, edited_legs: pd.DataFrame):
    trades, legs = load_trade_tables()
    now = utcnow_ts()
    trade_id = int(trade_id)

    trade_mask = _normalize_id_series(trades["trade_id"]) == trade_id
    if not trade_mask.any():
        raise ValueError(f"Trade {trade_id} não encontrado.")

    trade_idx = trades.index[trade_mask][0]
    multiplier = int(header_updates.get("quantity_multiplier", trades.at[trade_idx, "quantity_multiplier"] or 100))

    for key in ["trade_date", "asof_trade_date", "strategy", "ticker", "status", "fees", "underlying_spot", "thesis", "tags", "notes"]:
        if key not in header_updates:
            continue
        value = header_updates[key]
        if key in ("trade_date", "asof_trade_date") and value not in (None, "", pd.NaT):
            value = pd.Timestamp(value)
        trades.at[trade_idx, key] = value

    if "asset_id" in header_updates:
        trades.at[trade_idx, "asset_id"] = int(header_updates["asset_id"])

    edited = edited_legs.copy()
    if edited.empty:
        raise ValueError("A operação precisa ter pelo menos um leg.")

    for col in ["strike", "contracts", "entry_price", "exit_price", "ref_last_price", "leg_id", "leg_no"]:
        if col in edited.columns:
            edited[col] = pd.to_numeric(edited[col], errors="coerce")
    if "expiry" in edited.columns:
        edited["expiry"] = pd.to_datetime(edited["expiry"], errors="coerce")

    edited = edited.dropna(subset=["side", "opt_type", "expiry", "strike", "option_symbol", "contracts", "entry_price"]).copy()
    if edited.empty:
        raise ValueError("Todos os legs ficaram inválidos após a edição.")

    current_trade_legs = legs.loc[_normalize_id_series(legs["trade_id"]) == trade_id].copy()
    next_leg_id = next_id(legs, "leg_id")
    leg_rows = []
    direction_payload = []

    for idx, row in enumerate(edited.itertuples(index=False), start=1):
        leg_id = getattr(row, "leg_id", np.nan)
        if pd.isna(leg_id):
            leg_id = next_leg_id
            next_leg_id += 1

        existing = current_trade_legs.loc[_normalize_id_series(current_trade_legs["leg_id"]) == int(leg_id)].head(1)
        created_at = existing.iloc[0]["created_at"] if not existing.empty and "created_at" in existing.columns else now
        exit_price = getattr(row, "exit_price", np.nan)
        exit_price = np.nan if pd.isna(exit_price) else float(exit_price)
        ref_last_price = getattr(row, "ref_last_price", np.nan)
        ref_last_price = np.nan if pd.isna(ref_last_price) else float(ref_last_price)

        leg_payload = {
            "leg_id": int(leg_id),
            "trade_id": trade_id,
            "leg_no": idx,
            "side": str(row.side).upper(),
            "opt_type": str(row.opt_type).upper(),
            "expiry": pd.Timestamp(row.expiry),
            "strike": float(row.strike),
            "option_symbol": str(row.option_symbol),
            "contracts": int(row.contracts),
            "entry_price": float(row.entry_price),
            "exit_price": exit_price,
            "ref_last_price": ref_last_price,
            "created_at": created_at,
            "updated_at": now,
        }
        leg_rows.append(leg_payload)
        direction_payload.append(leg_payload)

    trades.at[trade_idx, "direction"] = direction_from_legs(direction_payload, multiplier)
    trades.at[trade_idx, "quantity_multiplier"] = multiplier
    trades.at[trade_idx, "updated_at"] = now

    legs = legs.loc[_normalize_id_series(legs["trade_id"]) != trade_id].copy()
    legs = pd.concat([legs, pd.DataFrame(leg_rows)], ignore_index=True, sort=False)
    trades = trades.copy()

    write_parquet(TRADES_PATH, trades.sort_values("trade_id").reset_index(drop=True))
    write_parquet(LEGS_PATH, legs.sort_values(["trade_id", "leg_no"]).reset_index(drop=True))


def delete_trade(trade_id: int):
    trades, legs = load_trade_tables()
    trade_id = int(trade_id)
    trades = trades.loc[_normalize_id_series(trades["trade_id"]) != trade_id].copy()
    legs = legs.loc[_normalize_id_series(legs["trade_id"]) != trade_id].copy()
    write_parquet(TRADES_PATH, trades.reset_index(drop=True))
    write_parquet(LEGS_PATH, legs.reset_index(drop=True))


st.title("Histórico Estruturado de Operações")

assets = load_assets()
if assets.empty:
    st.warning("Nenhum ativo ativo encontrado em `assets.parquet`.")
    st.stop()

tab_new, tab_history = st.tabs(["Nova Estrutura", "Histórico"])

with tab_new:
    st.subheader("Registrar operação estruturada")
    if "draft_legs" not in st.session_state:
        st.session_state.draft_legs = []

    ticker = st.selectbox("Ativo", assets["ticker"].tolist())
    asset_id = int(assets.loc[assets["ticker"] == ticker, "id"].iloc[0])
    asof_trade_date, spot = latest_spot(asset_id)
    quotes = load_quotes(asset_id)

    st.caption(f"Pregão base: {asof_trade_date} | Spot: {spot if spot is not None else 'N/A'}")

    strategy = st.selectbox(
        "Estratégia",
        [
            "Covered Call",
            "Bull Put Credit Spread",
            "Bear Call Credit Spread",
            "Bull Call Spread",
            "Bear Put Spread",
            "Long Straddle",
            "Custom",
        ],
    )

    col1, col2, col3, col4 = st.columns(4)
    opt_type = col2.selectbox("Tipo", ["CALL", "PUT"])
    default_expiry, default_subset, default_symbol = infer_defaults(quotes, spot, opt_type)
    expiries = sorted([d for d in quotes["expiry_date"].dropna().unique().tolist()])
    expiry_index = expiries.index(default_expiry) if default_expiry in expiries else 0
    expiry = col1.selectbox("Vencimento", expiries, index=expiry_index) if expiries else None
    subset = quotes[(quotes["expiry_date"] == expiry) & (quotes["option_type"] == opt_type)].copy()
    subset["strike_dist"] = (pd.to_numeric(subset["strike"], errors="coerce") - float(spot or 0.0)).abs()
    subset = subset.sort_values(["strike_dist", "trades", "volume", "strike", "option_symbol"], ascending=[True, False, False, True, True])
    symbols = subset["option_symbol"].tolist()
    default_symbol = default_symbol if default_symbol in symbols else (symbols[0] if symbols else None)
    symbol_index = symbols.index(default_symbol) if default_symbol in symbols else 0
    symbol = col3.selectbox("Opção", symbols, index=symbol_index) if symbols else None
    side = col4.selectbox("Lado", ["BUY", "SELL"])

    row = subset.loc[subset["option_symbol"] == symbol].head(1) if symbol else pd.DataFrame()
    default_strike = float(row.iloc[0]["strike"]) if not row.empty else 0.0
    default_last = float(row.iloc[0]["last_price"]) if not row.empty else 0.0

    col5, col6, col7 = st.columns(3)
    contracts = col5.number_input("Contratos", min_value=1, value=1, step=1)
    entry_price = col6.number_input("Preço de entrada", min_value=0.0, value=float(default_last), step=0.01)
    ref_last = col7.number_input("Snapshot last_price", min_value=0.0, value=float(default_last), step=0.01)

    if st.button("Adicionar leg"):
        if not symbol:
            st.warning("Selecione uma opção com liquidez.")
        else:
            st.session_state.draft_legs.append(
                {
                    "side": side,
                    "opt_type": opt_type,
                    "expiry": expiry,
                    "strike": default_strike,
                    "option_symbol": symbol,
                    "contracts": int(contracts),
                    "entry_price": float(entry_price),
                    "ref_last_price": float(ref_last),
                }
            )

    draft_df = pd.DataFrame(st.session_state.draft_legs)
    st.dataframe(draft_df, use_container_width=True, hide_index=True)

    trade_date = st.date_input("Data da operação", value=date.today())
    fees = st.number_input("Custos totais", min_value=0.0, value=0.0, step=0.01)
    thesis = st.text_input("Tese")
    tags = st.text_input("Tags")
    notes = st.text_area("Notas")

    if st.button("Salvar operação estruturada", type="primary"):
        if draft_df.empty:
            st.warning("Adicione ao menos um leg.")
        else:
            save_trade(
                {
                    "trade_date": trade_date,
                    "asof_trade_date": asof_trade_date,
                    "strategy": strategy,
                    "asset_id": asset_id,
                    "ticker": ticker,
                    "quantity_multiplier": 100,
                    "fees": fees,
                    "underlying_spot": spot,
                    "thesis": thesis,
                    "tags": tags,
                    "notes": notes,
                },
                st.session_state.draft_legs,
            )
            st.session_state.draft_legs = []
            st.success("Operação salva em parquet.")

with tab_history:
    trades, legs = load_trade_tables()
    if trades.empty:
        st.info("Ainda não há operações registradas.")
        st.stop()

    pnl_rows = []
    for trade in trades.itertuples(index=False):
        trade_id_value = _normalize_id_scalar(trade.trade_id)
        trade_legs = legs[_normalize_id_series(legs["trade_id"]) == trade_id_value].copy()
        fee = float(getattr(trade, "fees", 0.0) or 0.0)
        pnl = 0.0
        closed_count = 0
        for leg in trade_legs.itertuples(index=False):
            qty_mult = _normalize_id_scalar(getattr(trade, "quantity_multiplier", 100) or 100)
            leg_pnl = calc_leg_pnl(pd.Series(leg._asdict()), int(qty_mult))
            if leg_pnl is not None:
                pnl += leg_pnl
                closed_count += 1
        pnl_rows.append(
            {
                "trade_id": int(trade_id_value),
                "ticker": getattr(trade, "ticker", ""),
                "strategy": getattr(trade, "strategy", ""),
                "status": getattr(trade, "status", ""),
                "trade_date": getattr(trade, "trade_date", pd.NaT),
                "legs": len(trade_legs),
                "legs_fechados": closed_count,
                "pnl_total": pnl - fee,
                "fees": fee,
                "tags": getattr(trade, "tags", ""),
            }
        )

    pnl_df = pd.DataFrame(pnl_rows).sort_values(["status", "trade_date"], ascending=[True, False])
    st.dataframe(pnl_df, use_container_width=True, hide_index=True)

    trade_id = st.selectbox("Selecionar trade", pnl_df["trade_id"].tolist())
    selected_trade = trades.loc[_normalize_id_series(trades["trade_id"]) == int(trade_id)].iloc[0]
    selected_legs = legs.loc[_normalize_id_series(legs["trade_id"]) == int(trade_id)].copy()

    st.write("**Cabeçalho**")
    st.json(
        {
            "ticker": selected_trade.get("ticker"),
            "strategy": selected_trade.get("strategy"),
            "direction": selected_trade.get("direction"),
            "status": selected_trade.get("status"),
            "trade_date": str(selected_trade.get("trade_date")),
            "thesis": selected_trade.get("thesis"),
            "tags": selected_trade.get("tags"),
            "notes": selected_trade.get("notes"),
        }
    )

    st.write("**Legs**")
    st.dataframe(selected_legs, use_container_width=True, hide_index=True)

    st.write("**Editar operação**")
    edit_col1, edit_col2, edit_col3 = st.columns(3)
    editable_trade_date = pd.to_datetime(selected_trade.get("trade_date"), errors="coerce")
    editable_asof = pd.to_datetime(selected_trade.get("asof_trade_date"), errors="coerce")

    edit_trade_date = edit_col1.date_input(
        "Data da operação",
        value=(editable_trade_date.date() if pd.notna(editable_trade_date) else date.today()),
        key=f"edit_trade_date_{trade_id}",
    )
    edit_asof_date = edit_col2.date_input(
        "Pregão de referência",
        value=(editable_asof.date() if pd.notna(editable_asof) else date.today()),
        key=f"edit_asof_{trade_id}",
    )
    edit_status = edit_col3.selectbox(
        "Status",
        ["OPEN", "CLOSED"],
        index=(0 if str(selected_trade.get("status", "OPEN")).upper() == "OPEN" else 1),
        key=f"edit_status_{trade_id}",
    )

    edit_col4, edit_col5, edit_col6 = st.columns(3)
    edit_strategy = edit_col4.text_input("Estratégia", value=str(selected_trade.get("strategy", "")), key=f"edit_strategy_{trade_id}")
    edit_fees = edit_col5.number_input("Custos", min_value=0.0, value=float(selected_trade.get("fees", 0.0) or 0.0), step=0.01, key=f"edit_fees_{trade_id}")
    edit_spot = edit_col6.number_input(
        "Spot de referência",
        min_value=0.0,
        value=float(selected_trade.get("underlying_spot", 0.0) or 0.0),
        step=0.01,
        key=f"edit_spot_{trade_id}",
    )

    edit_col7, edit_col8 = st.columns(2)
    edit_thesis = edit_col7.text_input("Tese", value=str(selected_trade.get("thesis", "") or ""), key=f"edit_thesis_{trade_id}")
    edit_tags = edit_col8.text_input("Tags", value=str(selected_trade.get("tags", "") or ""), key=f"edit_tags_{trade_id}")
    edit_notes = st.text_area("Notas", value=str(selected_trade.get("notes", "") or ""), key=f"edit_notes_{trade_id}")

    editable_legs = selected_legs.copy()
    if not editable_legs.empty:
        editable_legs["expiry"] = pd.to_datetime(editable_legs["expiry"], errors="coerce").dt.date
    editable_cols = [
        "leg_id",
        "leg_no",
        "side",
        "opt_type",
        "expiry",
        "strike",
        "option_symbol",
        "contracts",
        "entry_price",
        "exit_price",
        "ref_last_price",
    ]
    editable_cols = [c for c in editable_cols if c in editable_legs.columns]
    edited_legs = st.data_editor(
        editable_legs[editable_cols],
        key=f"edit_legs_{trade_id}",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "leg_id": st.column_config.NumberColumn("leg_id", disabled=True),
            "leg_no": st.column_config.NumberColumn("leg_no", disabled=True),
            "side": st.column_config.SelectboxColumn("side", options=["BUY", "SELL"], required=True),
            "opt_type": st.column_config.SelectboxColumn("opt_type", options=["CALL", "PUT"], required=True),
            "expiry": st.column_config.DateColumn("expiry", format="YYYY-MM-DD"),
            "strike": st.column_config.NumberColumn("strike", format="%.4f"),
            "contracts": st.column_config.NumberColumn("contracts", format="%d"),
            "entry_price": st.column_config.NumberColumn("entry_price", format="%.4f"),
            "exit_price": st.column_config.NumberColumn("exit_price", format="%.4f"),
            "ref_last_price": st.column_config.NumberColumn("ref_last_price", format="%.4f"),
        },
    )

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("Salvar edições da operação", key=f"save_edit_{trade_id}", use_container_width=True):
        try:
            update_trade(
                int(trade_id),
                {
                    "trade_date": edit_trade_date,
                    "asof_trade_date": edit_asof_date,
                    "strategy": edit_strategy,
                    "asset_id": int(selected_trade.get("asset_id")),
                    "ticker": str(selected_trade.get("ticker")),
                    "status": edit_status,
                    "fees": edit_fees,
                    "underlying_spot": edit_spot,
                    "thesis": edit_thesis,
                    "tags": edit_tags,
                    "notes": edit_notes,
                    "quantity_multiplier": int(selected_trade.get("quantity_multiplier", 100) or 100),
                },
                edited_legs,
            )
            st.success("Operação atualizada.")
            st.rerun()
        except Exception as exc:
            st.error(f"Falha ao atualizar a operação: {exc}")

    confirm_delete = action_col2.checkbox("Confirmar exclusão permanente", key=f"confirm_delete_{trade_id}")
    if action_col2.button("Remover operação estruturada", key=f"delete_trade_{trade_id}", use_container_width=True, disabled=not confirm_delete):
        try:
            delete_trade(int(trade_id))
            st.success("Operação removida.")
            st.rerun()
        except Exception as exc:
            st.error(f"Falha ao remover a operação: {exc}")

    open_legs = selected_legs[selected_legs["exit_price"].isna()].copy()
    if not open_legs.empty:
        st.write("**Fechamento da estrutura**")
        exit_prices = {}
        cols = st.columns(max(1, len(open_legs)))
        for col, leg in zip(cols, open_legs.itertuples(index=False)):
            exit_prices[str(leg.option_symbol)] = col.number_input(
                f"{leg.option_symbol} exit",
                min_value=0.0,
                value=float(leg.ref_last_price) if pd.notna(leg.ref_last_price) else 0.0,
                step=0.01,
                key=f"exit_{trade_id}_{leg.option_symbol}",
            )
        if st.button("Fechar trade selecionado"):
            close_trade(int(trade_id), exit_prices)
            st.success("Trade fechado e legs atualizados.")
