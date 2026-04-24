import os
from datetime import date

import numpy as np
import pandas as pd
import requests
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
                "quantity",
                "underlying_entry_price",
                "notes",
                "status",
                "created_at",
                "updated_at",
            ]
        )
    for col in ["position_id", "asset_id", "quantity", "contracts", "quantity_multiplier"]:
        if col in df.columns:
            df[col] = _normalize_id_series(df[col])
    # Backward compatibility: synthesize quantity from legacy contracts * multiplier.
    if "quantity" not in df.columns:
        contracts = pd.to_numeric(df.get("contracts"), errors="coerce").fillna(0)
        multiplier = pd.to_numeric(df.get("quantity_multiplier"), errors="coerce").fillna(0)
        qty = (contracts * multiplier).round()
        df["quantity"] = qty.where(qty > 0, contracts.where(contracts > 0, 0)).astype("Int64")
    else:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).round().astype("Int64")
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


def historical_spot_on_or_before(asset_id: int, target_date: date | None) -> float | None:
    if target_date is None:
        return None
    daily = load_daily_bars_all()
    if daily.empty:
        return None
    asset_daily = daily.loc[pd.to_numeric(daily["asset_id"], errors="coerce") == int(asset_id)].copy()
    if asset_daily.empty:
        return None
    asset_daily = asset_daily.loc[pd.to_datetime(asset_daily["trade_date"], errors="coerce").dt.date <= target_date].copy()
    if asset_daily.empty:
        return None
    asset_daily = asset_daily.sort_values("trade_date")
    row = asset_daily.iloc[-1]
    # Prefer adjusted close for historical entry reference (closer to corporate-action adjusted price paid).
    return _safe_float(row.get("adj_close"), _safe_float(row.get("close")))


def infer_ticker_from_option_symbol(option_symbol: str, assets: pd.DataFrame) -> str | None:
    symbol = str(option_symbol or "").strip().upper()
    if not symbol or assets is None or assets.empty:
        return None

    # 1) Prefer exact mapping from local quote parquet (most reliable).
    quotes = load_option_quote_all()
    if not quotes.empty and "option_symbol" in quotes.columns:
        scope = quotes.loc[quotes["option_symbol"].astype(str).str.upper() == symbol].copy()
        if not scope.empty:
            if "trade_date" in scope.columns:
                scope["trade_date"] = pd.to_datetime(scope["trade_date"], errors="coerce")
                scope = scope.sort_values("trade_date")
            aid = _normalize_id_scalar(scope.iloc[-1].get("asset_id"))
            if not pd.isna(aid):
                row = assets.loc[_normalize_id_series(assets["id"]) == int(aid)]
                if not row.empty:
                    return str(row.iloc[0]["ticker"])

    # 2) Fallback by root prefix (e.g. ITUBL383 -> ITUB4.SA).
    root = "".join([c for c in symbol[:4] if c.isalpha()]).upper()
    if len(root) == 4:
        cands = assets.loc[assets["ticker"].astype(str).str.upper().str.startswith(root), "ticker"].astype(str).tolist()
        if len(cands) == 1:
            return cands[0]
    return None


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
    merged = asset_quotes.merge(
        asset_models,
        how="left",
        on=["asset_id", "trade_date", "option_symbol"],
        suffixes=("", "_m"),
    )
    # Prioriza dados vindos de option_quote (site) e usa option_model como fallback.
    for col in ["spot", "delta", "iv", "theta", "t_years", "mispricing_pct"]:
        mcol = f"{col}_m"
        if col in merged.columns and mcol in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").where(
                pd.to_numeric(merged[col], errors="coerce").notna(),
                pd.to_numeric(merged[mcol], errors="coerce"),
            )
        elif col not in merged.columns and mcol in merged.columns:
            merged[col] = pd.to_numeric(merged[mcol], errors="coerce")
    for col in ["spot", "delta", "iv", "theta", "t_years", "mispricing_pct"]:
        if col not in merged.columns:
            merged[col] = np.nan
    merged["DTE_days"] = (pd.to_datetime(merged["expiry_date"]) - pd.to_datetime(merged["trade_date"])).dt.days
    merged["liq"] = merged["trades"].fillna(0) + np.log1p(merged["volume"].fillna(0))
    return merged.sort_values(["expiry_date", "strike", "option_symbol"]).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_open_interest_live(underlying_symbol: str) -> pd.DataFrame:
    symbol = str(underlying_symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()
    params = {
        "z": str(int(pd.Timestamp.now().timestamp() / 10)),
        "r0t": "OpenInterest",
        "r0p.descending": "true",
        "r0p.sort_expression": "uncovered",
        "r0p.limit": "6000",
        "r0p.underlying_assets_ids": symbol,
    }
    try:
        response = requests.get("https://opcoes.net.br/api/v1", params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        reqs = payload.get("requests", [])
        if not reqs:
            return pd.DataFrame()
        result = reqs[0].get("results", {})
        fields = result.get("data_fields", [])
        rows = result.get("data_rows", [])
        if not fields or not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows, columns=fields)
        if "ticker" not in out.columns:
            return pd.DataFrame()
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        if "expiration_date" in out.columns:
            out["expiration_date"] = _to_date(out["expiration_date"])
        if "strike" in out.columns:
            out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
        return out
    except Exception:
        return pd.DataFrame()


def lookup_option_snapshot(
    asset_id: int,
    option_symbol: str,
    target_trade_date: date | None = None,
    underlying_ticker: str | None = None,
) -> pd.Series | None:
    symbol = str(option_symbol or "").strip().upper()
    if not symbol:
        return None
    base = str((underlying_ticker or "")).replace(".SA", "").replace(".sa", "").strip().upper()

    def from_open_interest() -> pd.Series | None:
        if not base:
            return None
        oi = load_open_interest_live(base)
        if oi.empty:
            return None
        oi_row = oi.loc[oi["ticker"] == symbol].head(1)
        if oi_row.empty:
            return None
        row = oi_row.iloc[0]
        return pd.Series(
            {
                "option_symbol": symbol,
                "strike": _safe_float(row.get("strike")),
                "expiry_date": row.get("expiration_date"),
                "last_price": np.nan,
                "trade_date": target_trade_date,
            }
        )

    quotes = load_option_quote_all()
    if quotes.empty:
        return from_open_interest()
    scope = quotes.loc[
        (pd.to_numeric(quotes["asset_id"], errors="coerce") == int(asset_id))
        & (quotes["option_symbol"].astype(str).str.upper() == symbol)
    ].copy()
    if scope.empty:
        return from_open_interest()
    scope["trade_date"] = _to_date(scope["trade_date"])
    scope["expiry_date"] = _to_date(scope["expiry_date"])
    if target_trade_date is not None:
        by_date = scope.loc[scope["trade_date"] == target_trade_date].copy()
        if by_date.empty:
            by_date = scope.loc[scope["trade_date"] <= target_trade_date].copy()
        if not by_date.empty:
            by_date = by_date.sort_values("trade_date")
            return by_date.iloc[-1]
    scope = scope.sort_values("trade_date")
    local = scope.iloc[-1].copy()

    # Fallback live: enrich strike/expiry from OpenInterest when symbol exists but local row is incomplete.
    if base:
        oi = load_open_interest_live(base)
        if not oi.empty:
            oi_row = oi.loc[oi["ticker"] == symbol].head(1)
            if not oi_row.empty:
                row = oi_row.iloc[0]
                if pd.isna(local.get("strike")) and pd.notna(row.get("strike")):
                    local["strike"] = row.get("strike")
                if pd.isna(local.get("expiry_date")) and pd.notna(row.get("expiration_date")):
                    local["expiry_date"] = row.get("expiration_date")
    return local


def lookup_option_last_price(asset_id: int, option_symbol: str, ref_trade_date: date | None = None) -> float | None:
    snap = lookup_option_snapshot(asset_id, option_symbol, ref_trade_date)
    if snap is None:
        return None
    return _safe_float(snap.get("last_price"))


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
        "quantity": int(payload["quantity"]),
        "contracts": int(payload["quantity"]),
        "quantity_multiplier": 1,
        "underlying_entry_price": float(payload["underlying_entry_price"]),
        "notes": payload.get("notes", ""),
        "status": str(payload.get("status", "OPEN")).upper(),
        "created_at": now,
        "updated_at": now,
    }
    upsert_parquet(POSITIONS_PATH, pd.DataFrame([row]), ["position_id"], ["position_id"])
    load_positions.clear()


def update_position(position_id: int, payload: dict):
    positions = load_positions()
    if positions.empty:
        return
    mask = _normalize_id_series(positions["position_id"]) == int(position_id)
    if not mask.any():
        return
    now = utcnow_ts()
    updates = {
        "ticker": payload["ticker"],
        "asset_id": int(payload["asset_id"]),
        "trade_date": pd.Timestamp(payload["trade_date"]),
        "asof_trade_date": pd.Timestamp(payload["asof_trade_date"]) if payload.get("asof_trade_date") else pd.NaT,
        "option_symbol": str(payload["option_symbol"]).strip().upper(),
        "option_expiry": pd.Timestamp(payload["option_expiry"]),
        "option_strike": float(payload["option_strike"]),
        "sale_price": float(payload["sale_price"]),
        "quantity": int(payload["quantity"]),
        "contracts": int(payload["quantity"]),
        "quantity_multiplier": 1,
        "underlying_entry_price": float(payload["underlying_entry_price"]),
        "notes": payload.get("notes", ""),
        "status": str(payload.get("status", "OPEN")).upper(),
        "updated_at": now,
    }
    for key, value in updates.items():
        positions.loc[mask, key] = value
    upsert_parquet(POSITIONS_PATH, positions.loc[mask].copy(), ["position_id"], ["position_id"])
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
    original_sale_price = _safe_float(position.get("sale_price"), 0.0)
    current_mark = _safe_float(position.get("sale_price"), 0.0)
    current_mark_source = "preco_registrado"
    current_dte = None
    current_delta = None
    if current_row is not None:
        marked = _safe_float(current_row.get("last_price"))
        if marked is not None:
            current_mark = marked
            current_mark_source = "mercado_atual"
        current_dte = _safe_float(current_row.get("DTE_days"))
        current_delta = _safe_float(current_row.get("delta"))

    spot_ref = _safe_float(spot_ref, _safe_float(position.get("underlying_entry_price"), 0.0))
    entry_spot = _safe_float(position.get("underlying_entry_price"), spot_ref or 0.0)
    current_strike = _safe_float(position.get("option_strike"), 0.0)
    quantity = int(_safe_float(position.get("quantity"), None) or 0)
    if quantity <= 0:
        contracts = int(_safe_float(position.get("contracts"), 1) or 1)
        multiplier = int(_safe_float(position.get("quantity_multiplier"), 100) or 100)
        quantity = max(1, contracts * multiplier)

    candidates = calls.copy()
    candidates = candidates[
        (candidates["DTE_days"] >= MIN_DTE_DAYS)
        & (candidates["DTE_days"] <= MAX_DTE_DAYS)
        & (candidates["expiry_date"] > pd.to_datetime(position["option_expiry"]).date())
    ].copy()
    for col in ["delta", "iv", "theta", "last_price", "strike", "trades", "volume", "DTE_days", "coberto", "travado", "descoberto"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    for required in ["delta", "iv"]:
        if required not in candidates.columns:
            candidates[required] = np.nan
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
            "removed_min_filters": 0,
        }

    candidates["spot_ref"] = float(spot_ref or 0.0)
    candidates["entry_spot"] = float(entry_spot or 0.0)
    candidates["open_interest_total"] = (
        candidates.get("coberto", pd.Series(0, index=candidates.index)).fillna(0)
        + candidates.get("travado", pd.Series(0, index=candidates.index)).fillna(0)
        + candidates.get("descoberto", pd.Series(0, index=candidates.index)).fillna(0)
    )
    candidates["credito_rolagem_hoje"] = candidates["last_price"] - float(current_mark or 0.0)
    candidates["credito_liquido"] = candidates["credito_rolagem_hoje"] + float(original_sale_price or 0.0)
    candidates["net_credit"] = candidates["credito_liquido"]
    candidates["roll_credit_now"] = candidates["credito_rolagem_hoje"]
    candidates["result_since_entry"] = candidates["credito_liquido"]
    candidates["cash_in_roll"] = candidates["credito_rolagem_hoje"] * quantity
    candidates["strike_gain"] = candidates["strike"] - float(current_strike or 0.0)
    candidates["strike_gain_positive"] = candidates["strike_gain"].clip(lower=0.0)
    candidates["retorno_adicional"] = candidates["credito_liquido"] + candidates["strike_gain_positive"]
    base_dte = float(current_dte or 0.0)
    candidates["dias_adicionais"] = (pd.to_numeric(candidates["DTE_days"], errors="coerce") - base_dte).clip(lower=0.0)
    candidates["retorno_por_dia"] = np.where(
        candidates["dias_adicionais"] > 0,
        candidates["retorno_adicional"] / candidates["dias_adicionais"],
        np.nan,
    )
    candidates["strike_gain_pct"] = np.where(
        candidates["spot_ref"] > 0,
        candidates["strike_gain"] / candidates["spot_ref"],
        np.nan,
    )
    candidates["yield_on_spot"] = np.where(
        candidates["spot_ref"] > 0,
        candidates["credito_rolagem_hoje"] / candidates["spot_ref"],
        np.nan,
    )
    candidates["yield_on_entry_spot"] = np.where(
        candidates["entry_spot"] > 0,
        candidates["credito_rolagem_hoje"] / candidates["entry_spot"],
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

    # Risco (Q): delta atual vs novo, strike atual vs novo, prob. de exercício (proxy ~ delta).
    cur_delta = float(current_delta) if current_delta is not None and np.isfinite(current_delta) else np.nan
    candidates["delta_atual"] = cur_delta
    candidates["delta_novo"] = candidates["delta"]
    candidates["delta_diff"] = candidates["delta_novo"] - candidates["delta_atual"]
    candidates["prob_exercicio"] = candidates["delta_novo"].clip(lower=0.0, upper=1.0)
    candidates["q_delta"] = np.where(
        np.isfinite(candidates["delta_atual"]),
        np.where(candidates["delta_novo"] <= candidates["delta_atual"], 1.0, (1.0 - (candidates["delta_novo"] - candidates["delta_atual"])).clip(lower=0.0, upper=1.0)),
        (1.0 - candidates["delta_novo"]).clip(lower=0.0, upper=1.0),
    )
    candidates["q_strike"] = ((candidates["strike"] - float(current_strike or 0.0)) / (candidates["spot_ref"].replace(0, np.nan))).clip(lower=-0.3, upper=0.3)
    candidates["q_strike"] = ((candidates["q_strike"] + 0.3) / 0.6).clip(lower=0.0, upper=1.0)
    candidates["q_prob"] = (1.0 - candidates["prob_exercicio"]).clip(lower=0.0, upper=1.0)
    candidates["Q"] = 0.40 * candidates["q_delta"] + 0.35 * candidates["q_strike"] + 0.25 * candidates["q_prob"]

    # Liquidez (L): spread (se disponível), volume e open interest.
    spread_cols = [c for c in ["spread", "spread_pct", "bid_ask_spread"] if c in candidates.columns]
    if spread_cols:
        spread_series = pd.to_numeric(candidates[spread_cols[0]], errors="coerce")
        candidates["spread_score"] = 1.0 - spread_series.rank(pct=True, method="average")
    else:
        candidates["spread_score"] = 0.5
    candidates["volume_score"] = pd.to_numeric(candidates["volume"], errors="coerce").rank(pct=True, method="average")
    candidates["oi_score"] = pd.to_numeric(candidates["open_interest_total"], errors="coerce").rank(pct=True, method="average")
    candidates["L"] = (
        0.40 * candidates["spread_score"].fillna(0.5)
        + 0.35 * candidates["volume_score"].fillna(0.0)
        + 0.25 * candidates["oi_score"].fillna(0.0)
    )

    # Retorno (R), crédito (C) e score final.
    candidates["R"] = pd.to_numeric(candidates["retorno_por_dia"], errors="coerce").rank(pct=True, method="average")
    candidates["C"] = pd.to_numeric(candidates["credito_liquido"], errors="coerce").rank(pct=True, method="average")
    candidates["score"] = (
        0.35 * candidates["R"].fillna(0.0)
        + 0.30 * candidates["Q"].fillna(0.0)
        + 0.20 * candidates["L"].fillna(0.0)
        + 0.15 * candidates["C"].fillna(0.0)
    )

    min_mask = (
        (candidates["dias_adicionais"] > 0)
        & (candidates["volume"].fillna(0) > 0)
        & (candidates["trades"].fillna(0) > 0)
        & (candidates["delta_novo"].between(0.01, 0.99))
    )
    if "open_interest_total" in candidates.columns:
        min_mask = min_mask & (candidates["open_interest_total"].fillna(0) >= 1)
    removed_min_filters = int((~min_mask).sum())
    candidates = candidates.loc[min_mask].copy()
    candidates = candidates.sort_values(["score", "retorno_por_dia", "credito_liquido"], ascending=[False, False, False]).reset_index(drop=True)

    context = {
        "trade_date": latest_trade_date,
        "spot_ref": spot_ref,
        "hist_vol": hist_vol,
        "current_mark": current_mark,
        "current_mark_source": current_mark_source,
        "current_dte": current_dte,
        "removed_missing_model": removed_missing_model,
        "removed_min_filters": removed_min_filters,
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

    if col1.button("Atualizar cotações do ativo", width="stretch"):
        with st.spinner("Baixando histórico do Yahoo Finance e recalculando indicadores..."):
            summary = run_market_update(update_quotes=True, update_options=False, selected_tickers=selected)
        clear_caches()
        st.success("Cotações e indicadores atualizados.")
        st.dataframe(pd.DataFrame(summary["tickers"]), width="stretch", hide_index=True)

    if col2.button("Atualizar opções e gregas", width="stretch"):
        with st.spinner("Coletando opções e recalculando gregas..."):
            summary = run_market_update(update_quotes=True, update_options=True, selected_tickers=selected)
        clear_caches()
        st.success("Cotações, opções e gregas atualizadas.")
        st.dataframe(pd.DataFrame(summary["tickers"]), width="stretch", hide_index=True)

    if col3.button("Recarregar arquivos parquet", width="stretch"):
        clear_caches()
        st.info("Caches locais do Streamlit foram limpos.")

    st.markdown("**Arquivos usados**")
    st.write(PATHS)

with tab_register:
    assets = load_assets()
    if assets.empty:
        st.warning("Nenhum ativo ativo encontrado em `assets.parquet`.")
        st.stop()

    # Keep underlying ticker coherent with option symbol when symbol is provided.
    typed_symbol = str(st.session_state.get("cc_option_symbol", "") or "").strip()
    inferred_ticker = infer_ticker_from_option_symbol(typed_symbol, assets) if typed_symbol else None
    if inferred_ticker and inferred_ticker in assets["ticker"].tolist():
        if st.session_state.get("register_ticker") != inferred_ticker:
            st.session_state["register_ticker"] = inferred_ticker

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

    default_trade_date = st.session_state.get("cc_trade_date", date.today())
    option_symbol = st.text_input(
        "Option symbol",
        placeholder="Ex.: ITUBD429W4",
        help="Digite o símbolo da opção. Se existir no parquet, Strike/Preço/Vencimento podem ser preenchidos automaticamente.",
        key="cc_option_symbol",
    )
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    trade_date = row1_col1.date_input("Data da venda", value=default_trade_date, key="cc_trade_date")
    quantity = row1_col2.number_input("Quantidade", min_value=1, value=int(st.session_state.get("cc_quantity", 100)), step=1, key="cc_quantity")
    auto_fill_option = row1_col3.checkbox("Auto preencher opção (strike/preço/venc.)", value=True, key="cc_auto_fill_option")

    option_symbol_clean = str(option_symbol).strip().upper()
    option_snapshot = (
        lookup_option_snapshot(asset_id, option_symbol_clean, trade_date, ticker)
        if option_symbol_clean
        else None
    )

    auto_strike = _safe_float(option_snapshot.get("strike")) if option_snapshot is not None else None
    auto_sale = _safe_float(option_snapshot.get("last_price")) if option_snapshot is not None else None
    auto_expiry = option_snapshot.get("expiry_date") if option_snapshot is not None else None
    auto_expiry_date = auto_expiry if isinstance(auto_expiry, date) else (asof_trade_date or date.today())

    # Keep widget state in sync for reactive auto-fill.
    if auto_fill_option:
        if auto_expiry is not None:
            st.session_state["cc_option_expiry"] = auto_expiry
        if auto_strike is not None:
            st.session_state["cc_option_strike"] = float(auto_strike)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    option_expiry = row2_col1.date_input(
        "Vencimento da opção",
        value=(auto_expiry_date if auto_fill_option else st.session_state.get("cc_option_expiry", auto_expiry_date)),
        key="cc_option_expiry",
    )
    option_strike = row2_col2.number_input(
        "Strike",
        min_value=0.0,
        value=float(auto_strike if (auto_fill_option and auto_strike is not None) else st.session_state.get("cc_option_strike", auto_strike or 0.0)),
        step=0.01,
        key="cc_option_strike",
    )
    sale_price = row2_col3.number_input(
        "Preço de venda da opção",
        min_value=0.0,
        value=float(st.session_state.get("cc_sale_price", auto_sale or 0.0)),
        step=0.01,
        key="cc_sale_price",
    )

    auto_entry = historical_spot_on_or_before(asset_id, trade_date)
    auto_fill_entry = st.checkbox("Auto preencher preço do ativo pela data da venda", value=True, key="cc_auto_fill_entry")
    if auto_fill_entry and auto_entry is not None:
        st.session_state["cc_underlying_entry"] = float(auto_entry)
    underlying_entry_price = st.number_input(
        "Preço do ativo subjacente",
        min_value=0.0,
        value=float(auto_entry if (auto_fill_entry and auto_entry is not None) else st.session_state.get("cc_underlying_entry", auto_entry or latest_spot or 0.0)),
        step=0.01,
        key="cc_underlying_entry",
    )
    if auto_entry is None:
        st.caption("Não encontrei fechamento histórico até a data da venda para este ativo; mantendo valor manual.")
    notes = st.text_area(
        "Notas",
        value=str(st.session_state.get("cc_notes", "")),
        placeholder="Ex.: venda coberta para gerar renda e aceitar rolagem para vencimentos longos.",
        key="cc_notes",
    )
    if st.button("Salvar venda coberta", type="primary", width="stretch"):
        if not option_symbol_clean:
            st.warning("Informe o option symbol da CALL vendida.")
        else:
            final_entry_price = float(auto_entry) if (auto_fill_entry and auto_entry is not None) else float(underlying_entry_price)
            final_sale_price = float(sale_price)
            save_position(
                {
                    "ticker": ticker,
                    "asset_id": asset_id,
                    "trade_date": trade_date,
                    "asof_trade_date": asof_trade_date,
                    "option_symbol": option_symbol_clean,
                    "option_expiry": option_expiry,
                    "option_strike": float(option_strike),
                    "sale_price": final_sale_price,
                    "quantity": int(quantity),
                    "underlying_entry_price": final_entry_price,
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
                    "quantity",
                    "underlying_entry_price",
                    "status",
                    "notes",
                ]
            ],
            width="stretch",
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

        if manage_col2.button(f"Marcar como {next_status}", width="stretch"):
            update_position_status(int(selected_position_id), next_status)
            st.success("Status atualizado.")
            st.rerun()

        confirm_delete = manage_col3.checkbox("Confirmar exclusão", key="confirm_delete_position")
        if st.button("Excluir posição selecionada", disabled=not confirm_delete, width="stretch"):
            delete_position(int(selected_position_id))
            st.success("Posição removida do parquet.")
            st.rerun()

        with st.expander("Editar posição selecionada", expanded=False):
            with st.form("covered_call_edit_form", clear_on_submit=False):
                edit_trade_date = st.date_input(
                    "Data da venda (edição)",
                    value=selected_position.get("trade_date") or date.today(),
                    key="edit_trade_date",
                )
                edit_option_expiry = st.date_input(
                    "Vencimento (edição)",
                    value=selected_position.get("option_expiry") or date.today(),
                    key="edit_option_expiry",
                )
                ec1, ec2, ec3 = st.columns(3)
                edit_option_symbol = ec1.text_input(
                    "Option symbol (edição)",
                    value=str(selected_position.get("option_symbol") or ""),
                    key="edit_option_symbol",
                )
                edit_option_strike = ec2.number_input(
                    "Strike (edição)",
                    min_value=0.0,
                    value=float(_safe_float(selected_position.get("option_strike"), 0.0) or 0.0),
                    step=0.01,
                    key="edit_option_strike",
                )
                edit_sale_price = ec3.number_input(
                    "Preço de venda (edição)",
                    min_value=0.0,
                    value=float(_safe_float(selected_position.get("sale_price"), 0.0) or 0.0),
                    step=0.01,
                    key="edit_sale_price",
                )

                ec4, ec5, ec6 = st.columns(3)
                edit_quantity = ec4.number_input(
                    "Quantidade (edição)",
                    min_value=1,
                    value=int(
                        _safe_float(
                            selected_position.get("quantity"),
                            (_safe_float(selected_position.get("contracts"), 1) or 1)
                            * (_safe_float(selected_position.get("quantity_multiplier"), 100) or 100),
                        )
                        or 1
                    ),
                    step=1,
                    key="edit_quantity",
                )
                ec5.empty()
                edit_entry_price = ec6.number_input(
                    "Entry point (edição)",
                    min_value=0.0,
                    value=float(_safe_float(selected_position.get("underlying_entry_price"), 0.0) or 0.0),
                    step=0.01,
                    key="edit_entry_price",
                )
                use_auto_entry = st.checkbox("Usar fechamento histórico automático para entry point", value=True)
                edit_notes = st.text_area(
                    "Notas (edição)",
                    value=str(selected_position.get("notes") or ""),
                    key="edit_notes",
                )
                save_edit = st.form_submit_button("Salvar edição da posição", type="primary", width="stretch")
                if save_edit:
                    edit_symbol_clean = str(edit_option_symbol).strip().upper()
                    if not edit_symbol_clean:
                        st.warning("Informe o option symbol na edição.")
                    else:
                        new_entry = historical_spot_on_or_before(asset_id, edit_trade_date) if use_auto_entry else None
                        final_entry = float(new_entry) if new_entry is not None else float(edit_entry_price)
                        final_sale = float(edit_sale_price)
                        update_position(
                            int(selected_position_id),
                            {
                                "ticker": ticker,
                                "asset_id": asset_id,
                                "trade_date": edit_trade_date,
                                "asof_trade_date": asof_trade_date,
                                "option_symbol": edit_symbol_clean,
                                "option_expiry": edit_option_expiry,
                                "option_strike": float(edit_option_strike),
                                "sale_price": final_sale,
                                "quantity": int(edit_quantity),
                                "underlying_entry_price": final_entry,
                                "notes": edit_notes,
                                "status": str(selected_position.get("status") or "OPEN"),
                            },
                        )
                        st.success("Posição atualizada.")
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
        f"Filtros mínimos: vencimento entre {MIN_DTE_DAYS}-{MAX_DTE_DAYS} dias, dias adicionais > 0, delta válido, volume/trades > 0 e open interest > 0."
    )
    if context.get("removed_missing_model"):
        st.info(f"{int(context['removed_missing_model'])} opções foram removidas por falta de Delta ou IV.")
    if context.get("removed_min_filters"):
        st.info(f"{int(context['removed_min_filters'])} opções foram removidas pelos filtros mínimos.")
    st.caption(
        "Score final: 0,35R + 0,30Q + 0,20L + 0,15C. "
        "R=retorno por dia, Q=risco (delta/strike/prob. exercício), L=liquidez (spread/volume/OI), C=crédito líquido."
    )

    if candidates.empty:
        st.warning("Não encontrei candidatos longos de rolagem para essa posição com o parquet atual.")
    else:
        display = candidates[
            [
                "option_symbol",
                "expiry_date",
                "DTE_days",
                "dias_adicionais",
                "strike",
                "last_price",
                "credito_rolagem_hoje",
                "credito_liquido",
                "retorno_adicional",
                "retorno_por_dia",
                "cash_in_roll",
                "delta_atual",
                "delta_novo",
                "delta_diff",
                "prob_exercicio",
                "strike_gain",
                "open_interest_total",
                "volume",
                "trades",
                "R",
                "Q",
                "L",
                "C",
                "score",
            ]
        ].copy()

        pct_cols = [
            "prob_exercicio",
            "R",
            "Q",
            "L",
            "C",
        ]
        for col in pct_cols:
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce")
        if "prob_exercicio" in display.columns:
            display["prob_exercicio"] = display["prob_exercicio"] * 100.0

        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "expiry_date": st.column_config.DateColumn("Vencimento", format="YYYY-MM-DD"),
                "DTE_days": st.column_config.NumberColumn("Dias até venc.", format="%d"),
                "dias_adicionais": st.column_config.NumberColumn("Dias adicionais", format="%d"),
                "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "last_price": st.column_config.NumberColumn("Prêmio atual", format="%.2f"),
                "credito_rolagem_hoje": st.column_config.NumberColumn("Crédito rolagem (hoje)", format="%.2f"),
                "credito_liquido": st.column_config.NumberColumn("Crédito líquido (acumulado)", format="%.2f"),
                "retorno_adicional": st.column_config.NumberColumn("Retorno adicional", format="%.2f"),
                "retorno_por_dia": st.column_config.NumberColumn("Retorno/dia", format="%.4f"),
                "cash_in_roll": st.column_config.NumberColumn("Caixa na rolagem", format="%.2f"),
                "delta_atual": st.column_config.NumberColumn("Delta atual", format="%.3f"),
                "delta_novo": st.column_config.NumberColumn("Delta novo", format="%.3f"),
                "delta_diff": st.column_config.NumberColumn("Δ Delta", format="%.3f"),
                "prob_exercicio": st.column_config.NumberColumn("Prob. exercício", format="%.2f%%"),
                "strike_gain": st.column_config.NumberColumn("Ganho de strike", format="%.2f"),
                "open_interest_total": st.column_config.NumberColumn("Open interest", format="%d"),
                "volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                "trades": st.column_config.NumberColumn("Negócios", format="%.0f"),
                "R": st.column_config.NumberColumn("R", format="%.2f"),
                "Q": st.column_config.NumberColumn("Q", format="%.2f"),
                "L": st.column_config.NumberColumn("L", format="%.2f"),
                "C": st.column_config.NumberColumn("C", format="%.2f"),
                "score": st.column_config.NumberColumn("Score final", format="%.3f"),
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
                "credito_rolagem_hoje": round(float(best["credito_rolagem_hoje"]), 2),
                "credito_liquido_acumulado": round(float(best["credito_liquido"]), 2),
                "retorno_adicional": round(float(best["retorno_adicional"]), 2),
                "retorno_por_dia": round(float(best["retorno_por_dia"]), 5) if pd.notna(best["retorno_por_dia"]) else None,
                "R": round(float(best["R"]), 3) if pd.notna(best["R"]) else None,
                "Q": round(float(best["Q"]), 3) if pd.notna(best["Q"]) else None,
                "L": round(float(best["L"]), 3) if pd.notna(best["L"]) else None,
                "C": round(float(best["C"]), 3) if pd.notna(best["C"]) else None,
                "score_final": round(float(best["score"]), 4),
            }
        )
