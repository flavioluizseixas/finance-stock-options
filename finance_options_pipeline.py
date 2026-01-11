#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv


# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
ENV_BSM_PATH = os.path.join(BASE_DIR, ".env.bsm")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except Exception:
        return default


@dataclass(frozen=True)
class CFG:
    db_host: str
    db_port: int
    db_user: str
    db_pass: str
    db_name: str

    tickers: List[str]
    hist_start: str

    di_curve_mode: str   # 'auto' | 'b3_file'
    di_curve_file: str

    risk_free_rate: float   # fallback flat (decimal a.a)
    dividend_yield: float   # decimal a.a

    batch_size: int
    iv_max_iter: int
    iv_tol: float
    iv_bounds_eps: float

    sharpe_win: int = 60
    sma_fast: int = 20
    sma_med: int = 50
    sma_slow: int = 200
    ema_fast: int = 12
    ema_slow: int = 26
    macd_sig: int = 9
    rsi_n: int = 14
    atr_n: int = 14


# =========================
# BCB SGS API (JSON)
# =========================
# O portal de dados abertos do BCB documenta o endpoint JSON do SGS. :contentReference[oaicite:1]{index=1}

def fetch_bcb_sgs_last_value(series_code: int, timeout: int = 20) -> Optional[float]:
    today = pd.Timestamp.today().date()
    start = (pd.Timestamp(today) - pd.Timedelta(days=30)).strftime("%d/%m/%Y")
    end = pd.Timestamp(today).strftime("%d/%m/%Y")
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"
    params = {"formato": "json", "dataInicial": start, "dataFinal": end}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        val = float(str(data[-1]["valor"]).replace(",", "."))
        return val
    except Exception:
        return None


def resolve_risk_free_rate_from_env() -> float:
    raw = os.getenv("RISK_FREE_RATE", "0.11").strip().lower()
    if raw in ("auto", "bcb", "selic", "sgs432"):
        # Meta Selic (SGS 432) – proxy para taxa curta
        selic_pct = fetch_bcb_sgs_last_value(432)
        if selic_pct is None:
            raise RuntimeError("Falha ao obter RISK_FREE_RATE automático (SGS 432). Defina valor fixo.")
        return selic_pct / 100.0
    return _env_float("RISK_FREE_RATE", 0.11)


def load_cfg() -> CFG:
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    load_dotenv(dotenv_path=ENV_BSM_PATH, override=False)

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = _env_int("DB_PORT", 3306)
    db_user = os.getenv("DB_USER", "root")
    db_pass = os.getenv("DB_PASS", os.getenv("DB_PASSWORD", ""))
    db_name = os.getenv("DB_NAME", "finance_options")

    raw = os.getenv("UNDERLYING_TICKER", "").strip()
    tickers = [t.strip() for t in raw.split(",") if t.strip()]
    if not tickers:
        raise RuntimeError("Defina UNDERLYING_TICKER no .env.bsm (ex: PETR4.SA,VALE3.SA).")

    hist_start = os.getenv("HIST_START", "2018-01-01").strip()

    di_curve_mode = os.getenv("DI_CURVE_MODE", "auto").strip().lower()
    di_curve_file = os.getenv("DI_CURVE_FILE", "").strip()

    cfg = CFG(
        db_host=db_host,
        db_port=db_port,
        db_user=db_user,
        db_pass=db_pass,
        db_name=db_name,
        tickers=tickers,
        hist_start=hist_start,
        di_curve_mode=di_curve_mode,
        di_curve_file=di_curve_file,
        risk_free_rate=resolve_risk_free_rate_from_env(),
        dividend_yield=_env_float("DIVIDEND_YIELD", 0.0),
        batch_size=_env_int("BATCH_SIZE", 2000),
        iv_max_iter=_env_int("IV_MAX_ITER", 60),
        iv_tol=_env_float("IV_TOL", 1e-6),
        iv_bounds_eps=_env_float("IV_BOUNDS_EPS", 0.05),
    )
    if not cfg.db_pass:
        raise RuntimeError("DB_PASSWORD/DB_PASS não definido no .env.")
    if cfg.di_curve_mode == "b3_file" and not cfg.di_curve_file:
        raise RuntimeError("DI_CURVE_MODE=b3_file mas DI_CURVE_FILE não foi definido.")
    return cfg


# =========================
# DB helpers
# =========================

def get_conn(cfg: CFG):
    return pymysql.connect(
        host=cfg.db_host,
        port=cfg.db_port,
        user=cfg.db_user,
        password=cfg.db_pass,
        database=cfg.db_name,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


def chunked(lst: List[Tuple], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def nan_to_none(x):
    if x is None:
        return None
    try:
        if pd.isna(x) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def int_or_none(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(round(f))
    except Exception:
        return None


def get_or_create_asset_id(cur, ticker: str) -> int:
    cur.execute("SELECT id FROM assets WHERE ticker=%s", (ticker,))
    row = cur.fetchone()
    if row:
        return int(row["id"])
    cur.execute("INSERT INTO assets (ticker, is_active) VALUES (%s, 1)", (ticker,))
    return int(cur.lastrowid)


def get_last_trade_date(cur, asset_id: int) -> Optional[date]:
    cur.execute("SELECT MAX(trade_date) AS d FROM daily_bars WHERE asset_id=%s", (asset_id,))
    row = cur.fetchone()
    if row and row.get("d"):
        d = row["d"]
        return d if isinstance(d, date) else pd.to_datetime(d).date()
    return None


def get_latest_spot_and_histvol(cur, asset_id: int, trade_date: date) -> Tuple[Optional[float], Optional[float]]:
    cur.execute(
        "SELECT close, vol_annual FROM daily_bars WHERE asset_id=%s AND trade_date=%s",
        (asset_id, trade_date),
    )
    row = cur.fetchone()
    if not row:
        return None, None
    return row.get("close"), row.get("vol_annual")


# =========================
# YF history + MultiIndex fix
# =========================

def normalize_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance pode devolver MultiIndex (Price, Ticker)
        try:
            df = df.xs(ticker, axis=1, level="Ticker", drop_level=True)
        except Exception:
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    return df


def fetch_yf_history(ticker: str, start: str, last_dt: Optional[date]) -> pd.DataFrame:
    dl_start = (pd.Timestamp(last_dt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last_dt else start
    df = yf.download(ticker, start=dl_start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_yf_columns(df, ticker)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df.index = pd.to_datetime(df.index.date)
    return df.sort_index()


# =========================
# indicators
# =========================

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(n).mean()
    down = -delta.clip(upper=0.0).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_indicators_one(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    d = df.copy()
    close = pd.to_numeric(d["Close"], errors="coerce")
    high = pd.to_numeric(d["High"], errors="coerce")
    low = pd.to_numeric(d["Low"], errors="coerce")

    d["log_ret"] = _log_returns(close)
    vol_daily = d["log_ret"].rolling(cfg.sharpe_win).std()
    d["vol_annual"] = vol_daily * np.sqrt(252)

    d["sma_20"] = close.rolling(cfg.sma_fast).mean()
    d["sma_50"] = close.rolling(cfg.sma_med).mean()
    d["sma_200"] = close.rolling(cfg.sma_slow).mean()

    e12 = _ema(close, cfg.ema_fast)
    e26 = _ema(close, cfg.ema_slow)
    macd = e12 - e26
    sig = _ema(macd, cfg.macd_sig)

    d["ema_12"] = e12
    d["ema_26"] = e26
    d["macd"] = macd
    d["macd_signal"] = sig
    d["macd_hist"] = macd - sig

    d["rsi_14"] = _rsi(close, cfg.rsi_n)
    d["atr_14"] = _atr(high, low, close, cfg.atr_n)
    return d


def upsert_daily_bars(conn, asset_id: int, df: pd.DataFrame, cfg: CFG) -> int:
    if df is None or df.empty:
        return 0

    sql = """
    INSERT INTO daily_bars (
      asset_id, trade_date,
      open, high, low, close, adj_close, volume,
      log_ret, vol_annual,
      sma_20, sma_50, sma_200,
      ema_12, ema_26, macd, macd_signal, macd_hist,
      rsi_14, atr_14
    ) VALUES (
      %s, %s,
      %s, %s, %s, %s, %s, %s,
      %s, %s,
      %s, %s, %s,
      %s, %s, %s, %s, %s,
      %s, %s
    )
    ON DUPLICATE KEY UPDATE
      open=VALUES(open),
      high=VALUES(high),
      low=VALUES(low),
      close=VALUES(close),
      adj_close=VALUES(adj_close),
      volume=VALUES(volume),
      log_ret=VALUES(log_ret),
      vol_annual=VALUES(vol_annual),
      sma_20=VALUES(sma_20),
      sma_50=VALUES(sma_50),
      sma_200=VALUES(sma_200),
      ema_12=VALUES(ema_12),
      ema_26=VALUES(ema_26),
      macd=VALUES(macd),
      macd_signal=VALUES(macd_signal),
      macd_hist=VALUES(macd_hist),
      rsi_14=VALUES(rsi_14),
      atr_14=VALUES(atr_14);
    """

    rows: List[Tuple] = []
    for dt, r in df.iterrows():
        td = pd.Timestamp(dt).date()
        rows.append((
            asset_id, td,
            nan_to_none(r.get("Open")),
            nan_to_none(r.get("High")),
            nan_to_none(r.get("Low")),
            nan_to_none(r.get("Close")),
            nan_to_none(r.get("AdjClose")),
            int_or_none(r.get("Volume")),
            nan_to_none(r.get("log_ret")),
            nan_to_none(r.get("vol_annual")),
            nan_to_none(r.get("sma_20")),
            nan_to_none(r.get("sma_50")),
            nan_to_none(r.get("sma_200")),
            nan_to_none(r.get("ema_12")),
            nan_to_none(r.get("ema_26")),
            nan_to_none(r.get("macd")),
            nan_to_none(r.get("macd_signal")),
            nan_to_none(r.get("macd_hist")),
            nan_to_none(r.get("rsi_14")),
            nan_to_none(r.get("atr_14")),
        ))

    aff = 0
    with conn.cursor() as cur:
        for batch in chunked(rows, cfg.batch_size):
            cur.executemany(sql, batch)
            aff += cur.rowcount
    return aff


# =========================
# DI Curve
# =========================

def busdays_between(d0: date, d1: date) -> int:
    return int(np.busday_count(d0, d1))


def year_fraction_bd252(trade_date: date, expiry_date: date) -> float:
    bd = busdays_between(trade_date, expiry_date)
    return max(0.0, float(bd) / 252.0)


def load_curve_from_b3_file(path: str) -> pd.DataFrame:
    """
    Espera um arquivo com colunas minimamente:
      - vertex_bd  (dias úteis)
      - rate       (taxa a.a em decimal ou %)
    Se vier em %, converte.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df = df.copy()
    # normaliza nomes
    df.columns = [c.strip().lower() for c in df.columns]

    # tente inferir
    if "vertex_bd" not in df.columns:
        # aceita "du" ou "dias_uteis" etc.
        for alt in ("du", "dias_uteis", "diasuteis", "business_days", "vertex"):
            if alt in df.columns:
                df = df.rename(columns={alt: "vertex_bd"})
                break

    if "rate" not in df.columns:
        for alt in ("taxa", "rate_aa", "taxa_aa", "juros"):
            if alt in df.columns:
                df = df.rename(columns={alt: "rate"})
                break

    df["vertex_bd"] = pd.to_numeric(df["vertex_bd"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["vertex_bd", "rate"]).copy()
    df["vertex_bd"] = df["vertex_bd"].astype(int)

    # heurística: se a maioria for > 1.5, assume que veio em %
    if (df["rate"] > 1.5).mean() > 0.7:
        df["rate"] = df["rate"] / 100.0

    df = df.sort_values("vertex_bd").drop_duplicates("vertex_bd")
    return df[["vertex_bd", "rate"]]


def build_curve(cfg: CFG, trade_date: date) -> Tuple[pd.DataFrame, str]:
    """
    Retorna (df_curve, source)
    df_curve: colunas vertex_bd, rate (decimal a.a)
    """
    if cfg.di_curve_mode == "b3_file":
        df = load_curve_from_b3_file(cfg.di_curve_file)
        return df, "B3_FILE"

    # auto (BCB proxy): curva plana com r = cfg.risk_free_rate
    df = pd.DataFrame({"vertex_bd": [1, 21, 63, 126, 252, 504], "rate": [cfg.risk_free_rate]*6})
    return df, "BCB"


def curve_rate(df_curve: pd.DataFrame, vertex_bd: int) -> float:
    """
    Interpolação linear simples em vértices (dias úteis).
    """
    v = int(vertex_bd)
    x = df_curve["vertex_bd"].values.astype(float)
    y = df_curve["rate"].values.astype(float)

    if v <= x.min():
        return float(y[x.argmin()])
    if v >= x.max():
        return float(y[x.argmax()])

    return float(np.interp(v, x, y))


def upsert_yield_curve(conn, trade_date: date, df_curve: pd.DataFrame, source: str, cfg: CFG) -> int:
    sql = """
    INSERT INTO yield_curve (trade_date, vertex_bd, rate, source)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      rate=VALUES(rate),
      source=VALUES(source);
    """
    rows = [(trade_date, int(r.vertex_bd), float(r.rate), source) for r in df_curve.itertuples(index=False)]
    aff = 0
    with conn.cursor() as cur:
        for batch in chunked(rows, cfg.batch_size):
            cur.executemany(sql, batch)
            aff += cur.rowcount
    return aff


# =========================
# opcoes.net wrappers
# =========================

def optionchaindate(subjacente: str, vencimento: str) -> pd.DataFrame:
    url = (
        "https://opcoes.net.br/listaopcoes/completa"
        f"?idAcao={subjacente}&listarVencimentos=false&cotacoes=true&vencimentos={vencimento}"
    )
    r = requests.get(url, timeout=30).json()

    rows = []
    cotacoes = r.get("data", {}).get("cotacoesOpcoes", [])
    for i in cotacoes:
        ativo = str(i[0]).split("_")[0]
        rows.append([subjacente, vencimento, ativo, i[2], i[3], i[5], i[8], i[9], i[10]])

    return pd.DataFrame(
        rows,
        columns=["subjacente","vencimento","ativo","tipo","modelo","strike","preco","negocios","volume"]
    )


def optionchain(subjacente: str) -> pd.DataFrame:
    url = (
        "https://opcoes.net.br/listaopcoes/completa"
        f"?idLista=ML&idAcao={subjacente}&listarVencimentos=true&cotacoes=true"
    )
    r = requests.get(url, timeout=30).json()
    vencimentos = [i["value"] for i in r.get("data", {}).get("vencimentos", [])]
    if not vencimentos:
        return pd.DataFrame(columns=["subjacente","vencimento","ativo","tipo","modelo","strike","preco","negocios","volume"])
    dfs = [optionchaindate(subjacente, v) for v in vencimentos]
    return pd.concat(dfs, ignore_index=True)


def upsert_option_chain_and_quotes(conn, asset_id: int, trade_date: date, df: pd.DataFrame, cfg: CFG) -> int:
    if df is None or df.empty:
        return 0

    d = df.copy()
    d["expiry_date"] = pd.to_datetime(d["vencimento"], errors="coerce").dt.date
    d["option_symbol"] = d["ativo"].astype(str)
    d["option_type"] = d["tipo"].astype(str).str.upper()
    d["model_code"] = d["modelo"].astype(str).str[:8]
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d["last_price"] = pd.to_numeric(d["preco"], errors="coerce")
    d["trades"] = pd.to_numeric(d["negocios"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
    d = d.dropna(subset=["expiry_date","option_symbol","option_type","strike"]).copy()

    chain_sql = """
    INSERT INTO option_chain (asset_id, trade_date, expiry_date)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE expiry_date=VALUES(expiry_date);
    """

    quote_sql = """
    INSERT INTO option_quote (
      asset_id, trade_date, expiry_date, option_symbol,
      option_type, model_code, strike,
      last_price, trades, volume
    ) VALUES (
      %s, %s, %s, %s,
      %s, %s, %s,
      %s, %s, %s
    )
    ON DUPLICATE KEY UPDATE
      expiry_date=VALUES(expiry_date),
      option_type=VALUES(option_type),
      model_code=VALUES(model_code),
      strike=VALUES(strike),
      last_price=VALUES(last_price),
      trades=VALUES(trades),
      volume=VALUES(volume);
    """

    chain_rows = []
    quote_rows = []

    for r in d.itertuples(index=False):
        chain_rows.append((asset_id, trade_date, r.expiry_date))
        opt_type = "CALL" if "C" in str(r.option_type) else "PUT"
        quote_rows.append((
            asset_id, trade_date, r.expiry_date, str(r.option_symbol),
            opt_type,
            None if pd.isna(r.model_code) else str(r.model_code),
            float(r.strike),
            nan_to_none(r.last_price),
            int_or_none(r.trades),
            int_or_none(r.volume),
        ))

    aff = 0
    with conn.cursor() as cur:
        chain_rows = list({x for x in chain_rows})
        for batch in chunked(chain_rows, cfg.batch_size):
            cur.executemany(chain_sql, batch)
            aff += cur.rowcount
        for batch in chunked(quote_rows, cfg.batch_size):
            cur.executemany(quote_sql, batch)
            aff += cur.rowcount
    return aff


def fetch_quotes_for_trade_date(cur, asset_id: int, trade_date: date) -> pd.DataFrame:
    cur.execute(
        """
        SELECT expiry_date, option_symbol, option_type, strike, last_price, trades, volume
        FROM option_quote
        WHERE asset_id=%s AND trade_date=%s
        """,
        (asset_id, trade_date),
    )
    rows = cur.fetchall() or []
    return pd.DataFrame(rows)


# =========================
# BSM + IV + greeks
# =========================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bsm_price_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if is_call:
        price = S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
        delta = disc_q * _norm_cdf(d1)
        rho = K * T * disc_r * _norm_cdf(d2)
        theta = (-(S * disc_q * _norm_pdf(d1) * sigma) / (2 * sqrtT)
                 - r * K * disc_r * _norm_cdf(d2)
                 + q * S * disc_q * _norm_cdf(d1))
    else:
        price = K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
        delta = -disc_q * _norm_cdf(-d1)
        rho = -K * T * disc_r * _norm_cdf(-d2)
        theta = (-(S * disc_q * _norm_pdf(d1) * sigma) / (2 * sqrtT)
                 + r * K * disc_r * _norm_cdf(-d2)
                 - q * S * disc_q * _norm_cdf(-d1))

    gamma = (disc_q * _norm_pdf(d1)) / (S * sigma * sqrtT)
    vega = S * disc_q * _norm_pdf(d1) * sqrtT
    return (price, delta, gamma, vega, theta, rho)


def implied_vol_newton(market_price: float, S: float, K: float, T: float, r: float, q: float,
                       is_call: bool, max_iter: int, tol: float, bounds_eps: float) -> Optional[float]:
    if market_price is None or not np.isfinite(market_price) or market_price <= 0:
        return None
    if S <= 0 or K <= 0 or T <= 0:
        return None

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if is_call:
        lower = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r

    if market_price < lower - bounds_eps or market_price > upper + bounds_eps:
        return None
    market_price = min(max(market_price, lower), upper)

    sigma = 0.30
    for _ in range(max_iter):
        price, _, _, vega, _, _ = bsm_price_greeks(S, K, T, r, q, sigma, is_call)
        if not np.isfinite(price) or not np.isfinite(vega) or vega <= 1e-12:
            return None
        diff = price - market_price
        if abs(diff) < tol:
            return float(max(1e-6, min(sigma, 10.0)))
        sigma = float(max(1e-6, min(sigma - diff / vega, 10.0)))
    return None


def upsert_option_model(conn, asset_id: int, trade_date: date, quotes: pd.DataFrame,
                        spot: float, hist_vol_annual: Optional[float],
                        df_curve: pd.DataFrame, cfg: CFG) -> int:
    if quotes is None or quotes.empty or spot is None or not np.isfinite(spot) or spot <= 0:
        return 0

    sql = """
    INSERT INTO option_model (
      asset_id, trade_date, option_symbol,
      spot, rate_r, dividend_q, t_years,
      iv, bsm_price, bsm_price_histvol, mispricing, mispricing_pct,
      delta, gamma, vega, theta, rho,
      hist_vol_annual
    ) VALUES (
      %s, %s, %s,
      %s, %s, %s, %s,
      %s, %s, %s, %s, %s,
      %s, %s, %s, %s, %s,
      %s
    )
    ON DUPLICATE KEY UPDATE
      spot=VALUES(spot),
      rate_r=VALUES(rate_r),
      dividend_q=VALUES(dividend_q),
      t_years=VALUES(t_years),
      iv=VALUES(iv),
      bsm_price=VALUES(bsm_price),
      bsm_price_histvol=VALUES(bsm_price_histvol),
      mispricing=VALUES(mispricing),
      mispricing_pct=VALUES(mispricing_pct),
      delta=VALUES(delta),
      gamma=VALUES(gamma),
      vega=VALUES(vega),
      theta=VALUES(theta),
      rho=VALUES(rho),
      hist_vol_annual=VALUES(hist_vol_annual);
    """

    rows: List[Tuple] = []
    q = float(cfg.dividend_yield)
    hv = float(hist_vol_annual) if (hist_vol_annual is not None and np.isfinite(hist_vol_annual) and hist_vol_annual > 0) else None

    for opt in quotes.itertuples(index=False):
        option_symbol = str(opt.option_symbol)
        expiry = opt.expiry_date
        if not isinstance(expiry, date):
            continue

        bd = busdays_between(trade_date, expiry)
        T = max(0.0, float(bd) / 252.0)
        if T <= 0:
            continue

        r = curve_rate(df_curve, bd)

        K = float(opt.strike)
        mkt = opt.last_price if (opt.last_price is not None and np.isfinite(opt.last_price)) else None
        is_call = (str(opt.option_type).upper() == "CALL")

        iv = implied_vol_newton(
            market_price=float(mkt) if mkt is not None else np.nan,
            S=float(spot), K=K, T=T, r=r, q=q,
            is_call=is_call,
            max_iter=cfg.iv_max_iter,
            tol=cfg.iv_tol,
            bounds_eps=cfg.iv_bounds_eps
        )

        if iv is not None:
            bsm_p, delta, gamma, vega, theta, rho = bsm_price_greeks(float(spot), K, T, r, q, float(iv), is_call)
            bsm_p = float(bsm_p) if np.isfinite(bsm_p) else None
            delta = float(delta) if np.isfinite(delta) else None
            gamma = float(gamma) if np.isfinite(gamma) else None
            vega = float(vega) if np.isfinite(vega) else None
            theta = float(theta) if np.isfinite(theta) else None   # theta ANUAL no DB
            rho = float(rho) if np.isfinite(rho) else None
        else:
            bsm_p = delta = gamma = vega = theta = rho = None

        if hv is not None:
            bsm_hist, *_ = bsm_price_greeks(float(spot), K, T, r, q, hv, is_call)
            bsm_hist = float(bsm_hist) if np.isfinite(bsm_hist) else None
        else:
            bsm_hist = None

        if bsm_hist is not None and mkt is not None and np.isfinite(mkt):
            mispricing = float(mkt) - float(bsm_hist)
            mispricing_pct = (float(mkt) / float(bsm_hist) - 1.0) if bsm_hist > 0 else None
        else:
            mispricing = None
            mispricing_pct = None

        rows.append((
            asset_id, trade_date, option_symbol,
            float(spot), float(r), float(q), float(T),
            float(iv) if iv is not None else None,
            bsm_p,
            bsm_hist,
            mispricing,
            mispricing_pct,
            delta, gamma, vega, theta, rho,
            hv
        ))

    if not rows:
        return 0

    aff = 0
    with conn.cursor() as cur:
        for batch in chunked(rows, cfg.batch_size):
            cur.executemany(sql, batch)
            aff += cur.rowcount
    return aff


# =========================
# MAIN
# =========================

def main():
    cfg = load_cfg()
    print(f"[ENV] DB={cfg.db_name} tickers={cfg.tickers}")
    print(f"[CURVE] mode={cfg.di_curve_mode} file={cfg.di_curve_file}")
    print(f"[BSM] fallback_r={cfg.risk_free_rate:.6f} q={cfg.dividend_yield:.6f} start={cfg.hist_start}")

    conn = get_conn(cfg)
    try:
        with conn.cursor() as cur:
            ticker_to_asset = {t: get_or_create_asset_id(cur, t) for t in cfg.tickers}
        conn.commit()

        for t in cfg.tickers:
            asset_id = ticker_to_asset[t]
            with conn.cursor() as cur:
                last_dt = get_last_trade_date(cur, asset_id)

            hist = fetch_yf_history(t, cfg.hist_start, last_dt)
            if hist.empty:
                print(f"[WARN] Sem histórico no Yahoo: {t}")
                continue

            hist_ind = add_indicators_one(hist, cfg)
            aff_daily = upsert_daily_bars(conn, asset_id, hist_ind, cfg)
            conn.commit()

            trade_date = pd.Timestamp(hist_ind.index.max()).date()
            with conn.cursor() as cur:
                spot, hist_vol_annual = get_latest_spot_and_histvol(cur, asset_id, trade_date)

            # monta curva e grava no DB
            df_curve, source = build_curve(cfg, trade_date)
            aff_curve = upsert_yield_curve(conn, trade_date, df_curve, source, cfg)
            conn.commit()

            print(f"\n=== {t} | asset_id={asset_id} ===")
            print(f"daily_bars upsert rowcount: {aff_daily}")
            print(f"trade_date={trade_date} | spot={spot} | hist_vol_annual={hist_vol_annual}")
            print(f"yield_curve upsert rowcount: {aff_curve} | source={source} | vertices={len(df_curve)}")

            subj = t.replace(".SA", "").replace(".sa", "")
            df_opt = optionchain(subj)
            aff_opt = upsert_option_chain_and_quotes(conn, asset_id, trade_date, df_opt, cfg)
            conn.commit()
            print(f"option_chain/option_quote upsert rowcount: {aff_opt} | linhas coletadas={len(df_opt)}")

            with conn.cursor() as cur:
                quotes = fetch_quotes_for_trade_date(cur, asset_id, trade_date)

            aff_model = upsert_option_model(conn, asset_id, trade_date, quotes, spot, hist_vol_annual, df_curve, cfg)
            conn.commit()
            print(f"option_model upsert rowcount: {aff_model}")

        print("\n[OK] Pipeline concluído.")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
