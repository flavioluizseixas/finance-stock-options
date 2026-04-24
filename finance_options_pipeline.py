#!/usr/bin/env python3

import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from parquet_store import ensure_assets, table_path, upsert_parquet, utcnow_ts


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
ENV_BSM_PATH = BASE_DIR / ".env.bsm"


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
    tickers: list[str]
    hist_start: str
    di_curve_mode: str
    di_curve_file: str
    risk_free_rate: float
    dividend_yield: float
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


def fetch_bcb_sgs_last_value(series_code: int, timeout: int = 20) -> Optional[float]:
    today = pd.Timestamp.today().date()
    start = (pd.Timestamp(today) - pd.Timedelta(days=30)).strftime("%d/%m/%Y")
    end = pd.Timestamp(today).strftime("%d/%m/%Y")
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"
    params = {"formato": "json", "dataInicial": start, "dataFinal": end}
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None
        return float(str(data[-1]["valor"]).replace(",", "."))
    except Exception:
        return None


def resolve_risk_free_rate_from_env() -> float:
    raw = os.getenv("RISK_FREE_RATE", "0.11").strip().lower()
    if raw in ("auto", "bcb", "selic", "sgs432"):
        selic_pct = fetch_bcb_sgs_last_value(432)
        if selic_pct is not None:
            return selic_pct / 100.0
        return 0.11
    return _env_float("RISK_FREE_RATE", 0.11)


def load_cfg() -> CFG:
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    load_dotenv(dotenv_path=ENV_BSM_PATH, override=False)

    tickers_raw = os.getenv("UNDERLYING_TICKER", "").strip()
    tickers = [ticker.strip() for ticker in tickers_raw.split(",") if ticker.strip()]
    if not tickers:
        raise RuntimeError("Defina UNDERLYING_TICKER no .env.bsm.")

    di_curve_mode = os.getenv("DI_CURVE_MODE", "auto").strip().lower()
    di_curve_file = os.getenv("DI_CURVE_FILE", "").strip()

    return CFG(
        tickers=tickers,
        hist_start=os.getenv("HIST_START", "2018-01-01").strip(),
        di_curve_mode=di_curve_mode,
        di_curve_file=di_curve_file,
        risk_free_rate=resolve_risk_free_rate_from_env(),
        dividend_yield=_env_float("DIVIDEND_YIELD", 0.0),
        iv_max_iter=_env_int("IV_MAX_ITER", 60),
        iv_tol=_env_float("IV_TOL", 1e-6),
        iv_bounds_eps=_env_float("IV_BOUNDS_EPS", 0.05),
    )


def normalize_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level="Ticker", drop_level=True)
        except Exception:
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
    return df.rename(columns={"Adj Close": "AdjClose"})


def fetch_yf_history(ticker: str, start: str, last_dt: Optional[date]) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError("Dependência ausente: instale `yfinance` para atualizar as cotações.") from exc

    today = pd.Timestamp.now().normalize()
    if last_dt:
        next_day = pd.Timestamp(last_dt).normalize() + pd.Timedelta(days=1)
        if next_day > today:
            return pd.DataFrame()
        dl_start = next_day.strftime("%Y-%m-%d")
    else:
        start_ts = pd.Timestamp(start).normalize()
        dl_start = min(start_ts, today).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=dl_start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = normalize_yf_columns(df, ticker)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df.index = pd.to_datetime(df.index.date)
    return df.sort_index()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


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
    d["vol_annual"] = d["log_ret"].rolling(cfg.sharpe_win).std() * np.sqrt(252)
    d["sma_20"] = close.rolling(cfg.sma_fast).mean()
    d["sma_50"] = close.rolling(cfg.sma_med).mean()
    d["sma_200"] = close.rolling(cfg.sma_slow).mean()
    d["ema_12"] = _ema(close, cfg.ema_fast)
    d["ema_26"] = _ema(close, cfg.ema_slow)
    d["macd"] = d["ema_12"] - d["ema_26"]
    d["macd_signal"] = _ema(d["macd"], cfg.macd_sig)
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["rsi_14"] = _rsi(close, cfg.rsi_n)
    d["atr_14"] = _atr(high, low, close, cfg.atr_n)
    return d


def _concat_compat_local(frames: list[pd.DataFrame]) -> pd.DataFrame:
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


def load_curve_from_b3_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [col.strip().lower() for col in df.columns]
    if "vertex_bd" not in df.columns:
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
    if (df["rate"] > 1.5).mean() > 0.7:
        df["rate"] = df["rate"] / 100.0
    return df.sort_values("vertex_bd").drop_duplicates("vertex_bd")[["vertex_bd", "rate"]]


def build_curve(cfg: CFG, trade_date: date) -> tuple[pd.DataFrame, str]:
    if cfg.di_curve_mode == "b3_file" and cfg.di_curve_file:
        return load_curve_from_b3_file(cfg.di_curve_file), "B3_FILE"
    return (
        pd.DataFrame(
            {
                "trade_date": [trade_date] * 6,
                "vertex_bd": [1, 21, 63, 126, 252, 504],
                "rate": [cfg.risk_free_rate] * 6,
                "source": ["BCB_PROXY"] * 6,
            }
        ),
        "BCB_PROXY",
    )


def busdays_between(d0: date, d1: date) -> int:
    return int(np.busday_count(d0, d1))


def curve_rate(df_curve: pd.DataFrame, vertex_bd: int) -> float:
    x = df_curve["vertex_bd"].values.astype(float)
    y = df_curve["rate"].values.astype(float)
    vertex = float(vertex_bd)
    if vertex <= x.min():
        return float(y[x.argmin()])
    if vertex >= x.max():
        return float(y[x.argmax()])
    return float(np.interp(vertex, x, y))


def optionchaindate(subjacente: str, vencimento: str) -> pd.DataFrame:
    url = (
        "https://opcoes.net.br/listaopcoes/completa"
        f"?idAcao={subjacente}&listarVencimentos=false&cotacoes=true&vencimentos={vencimento}"
    )
    data = requests.get(url, timeout=30).json()
    rows = []

    def _safe_item(values: list, idx: int):
        return values[idx] if idx < len(values) else None

    for item in data.get("data", {}).get("cotacoesOpcoes", []):
        ativo = str(_safe_item(item, 0)).split("_")[0]
        rows.append(
            [
                subjacente,
                vencimento,
                ativo,
                _safe_item(item, 2),   # tipo
                _safe_item(item, 3),   # mod.
                _safe_item(item, 5),   # strike
                _safe_item(item, 8),   # ultimo
                _safe_item(item, 9),   # negocios
                _safe_item(item, 10),  # volume
                _safe_item(item, 11),  # data/hora
                _safe_item(item, 12),  # vol implicita
                _safe_item(item, 13),  # delta
                _safe_item(item, 14),  # gamma
                _safe_item(item, 15),  # theta
                _safe_item(item, 16),  # vega
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "subjacente",
            "vencimento",
            "ativo",
            "tipo",
            "modelo",
            "strike",
            "preco",
            "negocios",
            "volume",
            "data_hora",
            "iv_site_raw",
            "delta_site_raw",
            "gamma_site_raw",
            "theta_site_raw",
            "vega_site_raw",
        ],
    )


def optionchain(subjacente: str) -> pd.DataFrame:
    url = (
        "https://opcoes.net.br/listaopcoes/completa"
        f"?idLista=ML&idAcao={subjacente}&listarVencimentos=true&cotacoes=true"
    )
    data = requests.get(url, timeout=30).json()
    expiries = [item["value"] for item in data.get("data", {}).get("vencimentos", [])]
    if not expiries:
        return pd.DataFrame()
    frames = [optionchaindate(subjacente, expiry) for expiry in expiries]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "subjacente",
                "vencimento",
                "ativo",
                "tipo",
                "modelo",
                "strike",
                "preco",
                "negocios",
                "volume",
                "data_hora",
                "iv_site_raw",
                "delta_site_raw",
                "gamma_site_raw",
                "theta_site_raw",
                "vega_site_raw",
            ]
        )
    return _concat_compat_local(frames).reset_index(drop=True)


def _parse_site_numeric(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or "<img" in text.lower():
        return np.nan
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("%", "").replace(" ", "").replace("\xa0", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except Exception:
        return np.nan


def fetch_open_interest(subjacente: str, limit: int = 6000) -> pd.DataFrame:
    params = {
        "z": str(int(pd.Timestamp.now().timestamp() / 10)),
        "r0t": "OpenInterest",
        "r0p.descending": "true",
        "r0p.sort_expression": "uncovered",
        "r0p.limit": str(limit),
        "r0p.underlying_assets_ids": subjacente,
    }
    try:
        response = requests.get("https://opcoes.net.br/api/v1", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        reqs = payload.get("requests", [])
        if not reqs:
            return pd.DataFrame()
        results = reqs[0].get("results", {})
        fields = results.get("data_fields", [])
        rows = results.get("data_rows", [])
        if not fields or not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows, columns=fields)
        if "ticker" not in out.columns:
            return pd.DataFrame()
        keep = {
            "ticker": "option_symbol",
            "covered": "coberto",
            "blocked": "travado",
            "uncovered": "descoberto",
            "buyers": "titulares",
            "sellers": "lancadores",
            "notional_uncovered": "notional_descoberto",
            "iq": "iq_oi",
        }
        for src in keep:
            if src not in out.columns:
                out[src] = np.nan
        out = out[list(keep.keys())].rename(columns=keep)
        out["option_symbol"] = out["option_symbol"].astype(str).str.strip()
        for col in ["coberto", "travado", "descoberto", "titulares", "lancadores", "notional_descoberto", "iq_oi"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.sort_values(["option_symbol"]).drop_duplicates(subset=["option_symbol"], keep="last")
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bsm_price_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if is_call:
        price = S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
        delta = disc_q * _norm_cdf(d1)
        rho = K * T * disc_r * _norm_cdf(d2)
        theta = (
            -(S * disc_q * _norm_pdf(d1) * sigma) / (2 * sqrt_t)
            - r * K * disc_r * _norm_cdf(d2)
            + q * S * disc_q * _norm_cdf(d1)
        )
    else:
        price = K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
        delta = -disc_q * _norm_cdf(-d1)
        rho = -K * T * disc_r * _norm_cdf(-d2)
        theta = (
            -(S * disc_q * _norm_pdf(d1) * sigma) / (2 * sqrt_t)
            + r * K * disc_r * _norm_cdf(-d2)
            - q * S * disc_q * _norm_cdf(-d1)
        )

    gamma = (disc_q * _norm_pdf(d1)) / (S * sigma * sqrt_t)
    vega = S * disc_q * _norm_pdf(d1) * sqrt_t
    return (price, delta, gamma, vega, theta, rho)


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    max_iter: int,
    tol: float,
    bounds_eps: float,
) -> Optional[float]:
    if market_price is None or not np.isfinite(market_price) or market_price <= 0:
        return None
    if S <= 0 or K <= 0 or T <= 0:
        return None

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    lower = max(0.0, S * disc_q - K * disc_r) if is_call else max(0.0, K * disc_r - S * disc_q)
    upper = S * disc_q if is_call else K * disc_r
    if market_price < lower - bounds_eps or market_price > upper + bounds_eps:
        return None

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


def build_daily_bars_frame(asset_id: int, hist_ind: pd.DataFrame) -> pd.DataFrame:
    now = utcnow_ts()
    frame = hist_ind.reset_index().rename(columns={"index": "trade_date", "AdjClose": "adj_close"})
    frame.columns = [str(col).lower() for col in frame.columns]
    frame["asset_id"] = int(asset_id)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["collected_at"] = now
    keep = [
        "asset_id",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "log_ret",
        "vol_annual",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "atr_14",
        "collected_at",
    ]
    return frame[keep]


def build_option_quote_frame(asset_id: int, trade_date: date, df_opt: pd.DataFrame) -> pd.DataFrame:
    if df_opt is None or df_opt.empty:
        return pd.DataFrame()
    now = utcnow_ts()
    out = df_opt.copy()
    out["asset_id"] = int(asset_id)
    out["trade_date"] = trade_date
    out["expiry_date"] = pd.to_datetime(out["vencimento"], errors="coerce").dt.date
    out["option_symbol"] = out["ativo"].astype(str)
    out["option_type"] = out["tipo"].astype(str).str.upper().map(lambda x: "CALL" if "C" in x else "PUT")
    out["model_code"] = out["modelo"].astype(str).str[:8]
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["last_price"] = pd.to_numeric(out["preco"], errors="coerce")
    out["trades"] = pd.to_numeric(out["negocios"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["iv"] = out["iv_site_raw"].map(_parse_site_numeric)
    out["delta"] = out["delta_site_raw"].map(_parse_site_numeric)
    out["gamma"] = out["gamma_site_raw"].map(_parse_site_numeric)
    out["theta"] = out["theta_site_raw"].map(_parse_site_numeric)
    out["vega"] = out["vega_site_raw"].map(_parse_site_numeric)
    out["iv"] = np.where(out["iv"] > 3, out["iv"] / 100.0, out["iv"])
    for col in ["coberto", "travado", "descoberto", "titulares", "lancadores", "notional_descoberto", "iq_oi"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["collected_at"] = now
    out = out.dropna(subset=["expiry_date", "option_symbol", "option_type", "strike"]).copy()
    return out[
        [
            "asset_id",
            "trade_date",
            "expiry_date",
            "option_symbol",
            "option_type",
            "model_code",
            "strike",
            "last_price",
            "trades",
            "volume",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "coberto",
            "travado",
            "descoberto",
            "titulares",
            "lancadores",
            "notional_descoberto",
            "iq_oi",
            "collected_at",
        ]
    ]


def build_option_model_frame(
    asset_id: int,
    trade_date: date,
    quotes: pd.DataFrame,
    spot: float,
    hist_vol_annual: Optional[float],
    df_curve: pd.DataFrame,
    cfg: CFG,
) -> pd.DataFrame:
    if quotes is None or quotes.empty or spot is None or not np.isfinite(spot) or spot <= 0:
        return pd.DataFrame()

    rows = []
    q = float(cfg.dividend_yield)
    hv = (
        float(hist_vol_annual)
        if hist_vol_annual is not None and np.isfinite(hist_vol_annual) and hist_vol_annual > 0
        else None
    )
    now = utcnow_ts()

    for opt in quotes.itertuples(index=False):
        expiry = getattr(opt, "expiry_date", None)
        if not isinstance(expiry, date):
            continue
        bd = busdays_between(trade_date, expiry)
        t_years = max(0.0, bd / 252.0)
        if t_years <= 0:
            continue

        strike = float(opt.strike)
        market_price = float(opt.last_price) if pd.notna(opt.last_price) else np.nan
        is_call = str(opt.option_type).upper() == "CALL"
        rate_r = curve_rate(df_curve, bd)
        iv_site = _parse_site_numeric(getattr(opt, "iv", np.nan))
        iv_site = (iv_site / 100.0) if np.isfinite(iv_site) and iv_site > 3 else iv_site
        delta_site = _parse_site_numeric(getattr(opt, "delta", np.nan))
        gamma_site = _parse_site_numeric(getattr(opt, "gamma", np.nan))
        theta_site = _parse_site_numeric(getattr(opt, "theta", np.nan))
        vega_site = _parse_site_numeric(getattr(opt, "vega", np.nan))
        iv = implied_vol_newton(
            market_price=market_price,
            S=float(spot),
            K=strike,
            T=t_years,
            r=rate_r,
            q=q,
            is_call=is_call,
            max_iter=cfg.iv_max_iter,
            tol=cfg.iv_tol,
            bounds_eps=cfg.iv_bounds_eps,
        )

        if iv is not None:
            bsm_price, delta_calc, gamma_calc, vega_calc, theta_calc, rho = bsm_price_greeks(
                float(spot), strike, t_years, rate_r, q, float(iv), is_call
            )
        else:
            bsm_price = delta_calc = gamma_calc = vega_calc = theta_calc = rho = np.nan

        if hv is not None:
            bsm_price_histvol, *_ = bsm_price_greeks(float(spot), strike, t_years, rate_r, q, hv, is_call)
        else:
            bsm_price_histvol = np.nan

        mispricing = market_price - bsm_price_histvol if np.isfinite(bsm_price_histvol) else np.nan
        mispricing_pct = (market_price / bsm_price_histvol - 1.0) if np.isfinite(bsm_price_histvol) and bsm_price_histvol > 0 else np.nan

        rows.append(
            {
                "asset_id": int(asset_id),
                "trade_date": trade_date,
                "option_symbol": str(opt.option_symbol),
                "spot": float(spot),
                "rate_r": float(rate_r),
                "dividend_q": float(q),
                "t_years": float(t_years),
                "iv": (iv_site if np.isfinite(iv_site) and iv_site > 0 else iv),
                "bsm_price": bsm_price,
                "bsm_price_histvol": bsm_price_histvol,
                "mispricing": mispricing,
                "mispricing_pct": mispricing_pct,
                "delta": (delta_site if np.isfinite(delta_site) else delta_calc),
                "gamma": (gamma_site if np.isfinite(gamma_site) else gamma_calc),
                "vega": (vega_site if np.isfinite(vega_site) else vega_calc),
                "theta": (theta_site if np.isfinite(theta_site) else theta_calc),
                "rho": rho,
                "hist_vol_annual": hv,
                "collected_at": now,
            }
        )

    return pd.DataFrame(rows)


def _last_daily_snapshot(asset_id: int) -> Optional[date]:
    path = table_path("daily_bars", BASE_DIR)
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["asset_id", "trade_date"])
    mask = pd.to_numeric(df["asset_id"], errors="coerce") == int(asset_id)
    s = pd.to_datetime(df.loc[mask, "trade_date"], errors="coerce").dropna()
    return None if s.empty else s.max().date()


def _latest_spot_and_histvol(asset_id: int, trade_date: date) -> tuple[Optional[float], Optional[float]]:
    path = table_path("daily_bars", BASE_DIR)
    if not path.exists():
        return None, None
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df = df[pd.to_numeric(df["asset_id"], errors="coerce") == int(asset_id)].copy()
    row = df[df["trade_date"] == trade_date].tail(1)
    if row.empty:
        return None, None
    spot = pd.to_numeric(row["close"], errors="coerce").dropna()
    spot_val = None if spot.empty else float(spot.iloc[-1])

    vol_today = pd.to_numeric(row["vol_annual"], errors="coerce").dropna()
    if not vol_today.empty and float(vol_today.iloc[-1]) > 0:
        return spot_val, float(vol_today.iloc[-1])

    vol_hist = pd.to_numeric(df.loc[df["trade_date"] <= trade_date, "vol_annual"], errors="coerce").dropna()
    vol_hist = vol_hist[vol_hist > 0]
    vol_val = None if vol_hist.empty else float(vol_hist.iloc[-1])
    return spot_val, vol_val


def _load_daily_price_window(asset_id: int, end_dt: Optional[date], rows: int) -> pd.DataFrame:
    path = table_path("daily_bars", BASE_DIR)
    if not path.exists() or end_dt is None:
        return pd.DataFrame()

    cols = ["asset_id", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    mask = (pd.to_numeric(df["asset_id"], errors="coerce") == int(asset_id)) & (df["trade_date"] <= pd.Timestamp(end_dt))
    hist = df.loc[mask, cols].copy().sort_values("trade_date").tail(max(1, int(rows)))
    if hist.empty:
        return pd.DataFrame()

    hist = hist.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "AdjClose",
            "volume": "Volume",
        }
    )
    hist["trade_date"] = pd.to_datetime(hist["trade_date"], errors="coerce")
    hist = hist.set_index("trade_date")
    keep = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for col in keep:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist[keep].dropna(how="all")


def update_underlying_history(cfg: CFG, ticker: str, asset_id: int) -> dict:
    last_dt = _last_daily_snapshot(asset_id)
    hist = fetch_yf_history(ticker, cfg.hist_start, last_dt)
    if hist.empty:
        return {"ticker": ticker, "daily_rows": 0, "trade_date": last_dt}

    lookback_rows = max(cfg.sma_slow, cfg.sharpe_win, cfg.atr_n, cfg.rsi_n, cfg.ema_slow, cfg.ema_fast, cfg.macd_sig) + 5
    prior = _load_daily_price_window(asset_id, last_dt, lookback_rows)
    if prior.empty:
        calc_base = hist.copy()
    else:
        calc_base = pd.concat([prior, hist], axis=0)
        calc_base = calc_base[~calc_base.index.duplicated(keep="last")].sort_index()

    hist_ind = add_indicators_one(calc_base, cfg)
    if last_dt is not None:
        hist_ind = hist_ind[hist_ind.index > pd.Timestamp(last_dt)]

    if hist_ind.empty:
        return {"ticker": ticker, "daily_rows": 0, "trade_date": last_dt}

    daily_frame = build_daily_bars_frame(asset_id, hist_ind)
    upsert_parquet(
        table_path("daily_bars", BASE_DIR),
        daily_frame,
        key_cols=["asset_id", "trade_date"],
        sort_cols=["asset_id", "trade_date"],
    )
    trade_date = pd.Timestamp(hist_ind.index.max()).date()
    return {"ticker": ticker, "daily_rows": len(daily_frame), "trade_date": trade_date}


def update_option_chain(cfg: CFG, ticker: str, asset_id: int, trade_date: date) -> dict:
    subj = ticker.replace(".SA", "").replace(".sa", "")
    raw_chain = optionchain(subj)
    oi = fetch_open_interest(subj)
    if not oi.empty and "ativo" in raw_chain.columns:
        raw_chain = raw_chain.merge(oi, how="left", left_on="ativo", right_on="option_symbol")
        raw_chain = raw_chain.drop(columns=["option_symbol"], errors="ignore")
    quote_frame = build_option_quote_frame(asset_id, trade_date, raw_chain)
    upsert_parquet(
        table_path("option_quote", BASE_DIR),
        quote_frame,
        key_cols=["asset_id", "trade_date", "option_symbol"],
        sort_cols=["asset_id", "trade_date", "expiry_date", "option_symbol"],
    )
    return {"ticker": ticker, "option_rows": len(quote_frame)}


def update_option_models(cfg: CFG, ticker: str, asset_id: int, trade_date: date) -> dict:
    quote_path = table_path("option_quote", BASE_DIR)
    if not quote_path.exists():
        return {"ticker": ticker, "model_rows": 0}
    quotes = pd.read_parquet(quote_path)
    quotes["trade_date"] = pd.to_datetime(quotes["trade_date"], errors="coerce").dt.date
    quotes["expiry_date"] = pd.to_datetime(quotes["expiry_date"], errors="coerce").dt.date
    quotes = quotes[
        (pd.to_numeric(quotes["asset_id"], errors="coerce") == int(asset_id)) & (quotes["trade_date"] == trade_date)
    ].copy()
    if quotes.empty:
        return {"ticker": ticker, "model_rows": 0}

    spot, hist_vol_annual = _latest_spot_and_histvol(asset_id, trade_date)
    if spot is None:
        return {"ticker": ticker, "model_rows": 0}

    df_curve, source = build_curve(cfg, trade_date)
    if "trade_date" not in df_curve.columns:
        df_curve = df_curve.copy()
        df_curve["trade_date"] = trade_date
        df_curve["source"] = source
    upsert_parquet(
        table_path("yield_curve", BASE_DIR),
        df_curve[["trade_date", "vertex_bd", "rate", "source"]],
        key_cols=["trade_date", "vertex_bd"],
        sort_cols=["trade_date", "vertex_bd"],
    )

    model_frame = build_option_model_frame(asset_id, trade_date, quotes, spot, hist_vol_annual, df_curve, cfg)
    upsert_parquet(
        table_path("option_model", BASE_DIR),
        model_frame,
        key_cols=["asset_id", "trade_date", "option_symbol"],
        sort_cols=["asset_id", "trade_date", "option_symbol"],
    )
    return {"ticker": ticker, "model_rows": len(model_frame), "spot": spot, "hist_vol_annual": hist_vol_annual}


def run_market_update(
    update_quotes: bool = True,
    update_options: bool = True,
    selected_tickers: Optional[list[str]] = None,
) -> dict:
    cfg = load_cfg()
    tickers = selected_tickers or cfg.tickers
    assets = ensure_assets(tickers, BASE_DIR)
    ticker_to_asset = {str(row["ticker"]): int(row["id"]) for _, row in assets.iterrows()}

    summary = {
        "config": asdict(cfg),
        "tickers": [],
        "updated_at": utcnow_ts(),
    }

    for ticker in tickers:
        asset_id = ticker_to_asset[ticker]
        item = {"ticker": ticker, "asset_id": asset_id}

        if update_quotes:
            item.update(update_underlying_history(cfg, ticker, asset_id))

        trade_date = item.get("trade_date")
        if update_options and not trade_date:
            trade_date = _last_daily_snapshot(asset_id)
            if trade_date:
                item["trade_date"] = trade_date

        if update_options and trade_date:
            item.update(update_option_chain(cfg, ticker, asset_id, trade_date))
            item.update(update_option_models(cfg, ticker, asset_id, trade_date))
        elif update_options:
            item.update({"option_rows": 0, "model_rows": 0})

        summary["tickers"].append(item)

    return summary


def main():
    summary = run_market_update(update_quotes=True, update_options=True)
    print("[OK] Pipeline parquet concluído.")
    for item in summary["tickers"]:
        print(item)


if __name__ == "__main__":
    main()
