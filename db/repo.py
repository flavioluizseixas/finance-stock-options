import os
from datetime import date
import pandas as pd
import streamlit as st

from config import env_paths

def _require_file(path: str):
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        st.stop()

def _coerce_dates(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df

def _paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return env_paths(base_dir)

@st.cache_data(ttl=300, show_spinner=False)
def load_assets() -> pd.DataFrame:
    paths = _paths()
    _require_file(paths["PATH_ASSETS"])
    df = pd.read_parquet(paths["PATH_ASSETS"])
    if "is_active" in df.columns:
        df = df[df["is_active"].astype(int) == 1]
    df = df[["id", "ticker"]].dropna().sort_values("ticker")
    df["id"] = df["id"].astype(int)
    return df.reset_index(drop=True)

@st.cache_data(ttl=300, show_spinner=False)
def _load_option_quote_all() -> pd.DataFrame:
    paths = _paths()
    _require_file(paths["PATH_QUOTE"])
    df = pd.read_parquet(paths["PATH_QUOTE"])
    df = _coerce_dates(df, ["trade_date", "expiry_date"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def _load_option_model_all() -> pd.DataFrame:
    paths = _paths()
    _require_file(paths["PATH_MODEL"])
    df = pd.read_parquet(paths["PATH_MODEL"])
    df = _coerce_dates(df, ["trade_date"])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def _load_daily_bars_all() -> pd.DataFrame:
    paths = _paths()
    _require_file(paths["PATH_DAILY"])
    df = pd.read_parquet(paths["PATH_DAILY"])
    df = _coerce_dates(df, ["trade_date"])
    return df

@st.cache_data(ttl=60, show_spinner=False)
def load_latest_trade_date(asset_id: int):
    oq = _load_option_quote_all()
    d = oq.loc[oq["asset_id"].astype(int) == int(asset_id), "trade_date"].dropna()
    return max(d) if len(d) else None

@st.cache_data(ttl=60, show_spinner=False)
def load_daily_indicators(asset_id: int, trade_date: date):
    db = _load_daily_bars_all()
    m = (db["asset_id"].astype(int) == int(asset_id)) & (db["trade_date"] == trade_date)
    d = db.loc[m].copy()
    if d.empty:
        return None
    return d.iloc[0].to_dict()

@st.cache_data(ttl=60, show_spinner=False)
def load_chain(asset_id: int, trade_date: date) -> pd.DataFrame:
    oq = _load_option_quote_all()
    om = _load_option_model_all()

    m = (oq["asset_id"].astype(int) == int(asset_id)) & (oq["trade_date"] == trade_date)
    q = oq.loc[m].copy()
    if q.empty:
        return pd.DataFrame()

    for c in ["trades", "last_price"]:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q[(q["trades"] > 0) & (q["last_price"] > 0)].copy()
    if q.empty:
        return pd.DataFrame()

    m2 = (om["asset_id"].astype(int) == int(asset_id)) & (om["trade_date"] == trade_date)
    mm = om.loc[m2].copy()

    if "collected_at" in q.columns:
        q = q.rename(columns={"collected_at": "quote_collected_at"})
    if "collected_at" in mm.columns:
        mm = mm.rename(columns={"collected_at": "model_collected_at"})

    out = q.merge(
        mm,
        how="left",
        on=["asset_id", "trade_date", "option_symbol"],
        suffixes=("", "_m"),
    )

    if set(["expiry_date","option_type","strike"]).issubset(out.columns):
        out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
        out = out.sort_values(["expiry_date", "option_type", "strike"], ascending=[True, True, True])

    return out.reset_index(drop=True)
