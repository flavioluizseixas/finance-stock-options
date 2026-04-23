import os
from datetime import datetime
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
    df = pd.read_parquet(path)
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
    return df


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


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


def upsert_parquet(
    path: Path,
    new_df: pd.DataFrame,
    key_cols: list[str],
    sort_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return read_parquet_safe(path)

    current = read_parquet_safe(path)
    merged = pd.concat([current, new_df], ignore_index=True, sort=False)
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
