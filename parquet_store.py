import os
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TABLE_FILES = {
    "assets": "assets.parquet",
    "daily_bars": "daily_bars.parquet",
    "option_quote": "option_quote.parquet",
    "option_model": "option_model.parquet",
    "yield_curve": "yield_curve.parquet",
    "covered_call_positions": "covered_call_positions.parquet",
    "structured_trades": "structured_trades.parquet",
    "structured_trade_legs": "structured_trade_legs.parquet",
}


def base_dir_from(path_hint: Optional[str] = None) -> Path:
    if path_hint:
        return Path(path_hint).resolve()
    return Path(__file__).resolve().parent


def env_paths(base_dir: Optional[str] = None) -> dict[str, str]:
    root = base_dir_from(base_dir)
    data_dir = Path(os.getenv("DATA_DIR", str(root / "data"))).resolve()
    return {
        "DATA_DIR": str(data_dir),
        "PATH_ASSETS": os.getenv("PATH_ASSETS", str(data_dir / TABLE_FILES["assets"])),
        "PATH_DAILY": os.getenv("PATH_DAILY", str(data_dir / TABLE_FILES["daily_bars"])),
        "PATH_QUOTE": os.getenv("PATH_QUOTE", str(data_dir / TABLE_FILES["option_quote"])),
        "PATH_MODEL": os.getenv("PATH_MODEL", str(data_dir / TABLE_FILES["option_model"])),
        "PATH_YIELD_CURVE": os.getenv("PATH_YIELD_CURVE", str(data_dir / TABLE_FILES["yield_curve"])),
        "PATH_COVERED_CALL_POSITIONS": os.getenv(
            "PATH_COVERED_CALL_POSITIONS", str(data_dir / TABLE_FILES["covered_call_positions"])
        ),
        "PATH_STRUCTURED_TRADES": os.getenv(
            "PATH_STRUCTURED_TRADES", str(data_dir / TABLE_FILES["structured_trades"])
        ),
        "PATH_STRUCTURED_TRADE_LEGS": os.getenv(
            "PATH_STRUCTURED_TRADE_LEGS", str(data_dir / TABLE_FILES["structured_trade_legs"])
        ),
    }


def ensure_data_dir(base_dir: Optional[str] = None) -> dict[str, str]:
    paths = env_paths(base_dir)
    Path(paths["DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    return paths


def table_path(table_name: str, base_dir: Optional[str] = None) -> Path:
    paths = ensure_data_dir(base_dir)
    mapping = {
        "assets": "PATH_ASSETS",
        "daily_bars": "PATH_DAILY",
        "option_quote": "PATH_QUOTE",
        "option_model": "PATH_MODEL",
        "yield_curve": "PATH_YIELD_CURVE",
        "covered_call_positions": "PATH_COVERED_CALL_POSITIONS",
        "structured_trades": "PATH_STRUCTURED_TRADES",
        "structured_trade_legs": "PATH_STRUCTURED_TRADE_LEGS",
    }
    return Path(paths[mapping[table_name]])


def read_parquet_safe(path: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    df = _normalize_frame_types(pd.read_parquet(path))
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
    return df


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_frame_types(df).to_parquet(path, index=False)


def _normalize_int_series(series: pd.Series) -> pd.Series:
    raw = series
    if pd.api.types.is_datetime64_any_dtype(raw):
        raw = raw.astype("int64", copy=False)
    num = pd.to_numeric(raw, errors="coerce")
    if num.isna().all():
        return series
    return num.round().astype("Int64")


def _normalize_datetime_series(series: pd.Series, date_only: bool) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(out):
        out = out.dt.tz_convert(None)
    if date_only:
        out = out.dt.normalize()
    return out


def _looks_like_temporal_object(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.head(50)
    return sample.map(lambda v: isinstance(v, (date, datetime, pd.Timestamp, np.datetime64))).all()


def _normalize_frame_types(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    int_cols = {"id", "asset_id", "vertex_bd", "is_active"}
    date_cols = {"trade_date", "expiry_date"}
    datetime_cols = {"created_at", "updated_at", "collected_at"}

    for col in out.columns:
        if col in int_cols:
            out[col] = _normalize_int_series(out[col])
            continue
        if col in date_cols:
            out[col] = _normalize_datetime_series(out[col], date_only=True)
            continue
        if col in datetime_cols:
            out[col] = _normalize_datetime_series(out[col], date_only=False)
            continue
        if out[col].dtype == "object" and _looks_like_temporal_object(out[col]):
            out[col] = _normalize_datetime_series(out[col], date_only=False)

    return out


def _safe_sort_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def normalize(value):
        if pd.isna(value):
            return (2, "")
        if isinstance(value, pd.Timestamp):
            return (0, int(value.value))
        if isinstance(value, np.datetime64):
            return (0, int(pd.Timestamp(value).value))
        try:
            return (0, float(value))
        except Exception:
            return (1, str(value))

    return series.map(normalize)


def _concat_compat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    if len(valid) == 1:
        return valid[0].reset_index(drop=True)

    all_cols: list[str] = []
    for frame in valid:
        for col in frame.columns:
            if col not in all_cols:
                all_cols.append(col)

    prepared = []
    for frame in valid:
        keep_cols = [col for col in frame.columns if not frame[col].isna().all()]
        prepared.append(frame[keep_cols] if keep_cols else frame.iloc[:, 0:0])

    merged = pd.concat(prepared, ignore_index=True, sort=False)
    for col in all_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[all_cols]


def upsert_parquet(
    path: Path,
    new_df: pd.DataFrame,
    key_cols: list[str],
    sort_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return read_parquet_safe(path)

    current = read_parquet_safe(path)
    incoming = _normalize_frame_types(new_df)
    merged = _concat_compat([current, incoming])
    if merged.empty:
        write_parquet(path, merged)
        return merged

    merged = merged.drop_duplicates(subset=key_cols, keep="last")

    if sort_cols:
        sortable = [col for col in sort_cols if col in merged.columns]
        if sortable:
            sort_frame = merged.copy()
            for col in sortable:
                sort_frame[f"__sort__{col}"] = _safe_sort_series(sort_frame[col])
            merged = (
                sort_frame.sort_values([f"__sort__{col}" for col in sortable])
                .drop(columns=[f"__sort__{col}" for col in sortable])
                .reset_index(drop=True)
            )

    write_parquet(path, merged)
    return merged


def utcnow_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.utcnow()).tz_localize(None)


def load_assets(base_dir: Optional[str] = None) -> pd.DataFrame:
    path = table_path("assets", base_dir)
    df = read_parquet_safe(path)
    if df.empty:
        return pd.DataFrame(columns=["id", "ticker", "is_active", "created_at", "updated_at"])
    if "is_active" not in df.columns:
        df["is_active"] = 1
    return df


def ensure_assets(tickers: list[str], base_dir: Optional[str] = None) -> pd.DataFrame:
    path = table_path("assets", base_dir)
    current = load_assets(base_dir)
    if current.empty:
        current = pd.DataFrame(columns=["id", "ticker", "is_active", "created_at", "updated_at"])

    existing = set(current.get("ticker", pd.Series(dtype=str)).astype(str))
    next_id = int(pd.to_numeric(current.get("id", pd.Series(dtype=float)), errors="coerce").max()) + 1 if not current.empty else 1

    rows = []
    now = utcnow_ts()
    for ticker in tickers:
        if ticker in existing:
            continue
        rows.append(
            {
                "id": next_id,
                "ticker": ticker,
                "is_active": 1,
                "created_at": now,
                "updated_at": pd.NaT,
            }
        )
        next_id += 1

    if rows:
        current = pd.concat([current, pd.DataFrame(rows)], ignore_index=True, sort=False)

    current["is_active"] = pd.to_numeric(current["is_active"], errors="coerce").fillna(1).astype(int)
    current = current.sort_values(["ticker", "id"]).reset_index(drop=True)
    write_parquet(path, current)
    return current


def latest_trade_date(table_name: str, asset_id: int, base_dir: Optional[str] = None):
    path = table_path(table_name, base_dir)
    df = read_parquet_safe(path)
    if df.empty or "asset_id" not in df.columns or "trade_date" not in df.columns:
        return None
    s = pd.to_datetime(df.loc[df["asset_id"].astype(int) == int(asset_id), "trade_date"], errors="coerce").dropna()
    if s.empty:
        return None
    return s.max().date()


def load_table(table_name: str, base_dir: Optional[str] = None) -> pd.DataFrame:
    return read_parquet_safe(table_path(table_name, base_dir))
