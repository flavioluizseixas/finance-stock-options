# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# === BASE DIR (pasta onde está este arquivo) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CFG:
    # Arquivos/pastas com PATH COMPLETO
    TICKERS_CSV   = os.path.join(BASE_DIR, "tickers.csv")
    OUT_DIR       = os.path.join(BASE_DIR, "MS_ASCII")
    OUT_FILE      = "ms7_all.txt"           # nome do arquivo (será combinado com OUT_DIR)
    CACHE_DIR     = os.path.join(BASE_DIR, "cache_yf")

    # Período (primeiro uso baixa tudo; próximos usos só o “rabo” faltante)
    START         = "2010-01-01"
    END           = None                    # None => até hoje (yfinance usa end exclusivo)

    # Polidez c/ servidor
    SLEEP_BETWEEN = 0.10                    # segundos entre requisições


# Garante pastas
os.makedirs(CFG.OUT_DIR, exist_ok=True)
os.makedirs(CFG.CACHE_DIR, exist_ok=True)


def _hash(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def cache_path(kind: str, key: str, ext="parquet") -> str:
    fn = f"{kind}__{_hash(key)}.{ext}"
    return os.path.join(CFG.CACHE_DIR, fn)


def load_parquet(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def save_parquet(df: pd.DataFrame, path: str):
    if df is not None and not df.empty:
        df.to_parquet(path, index=True)


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """Garante DatetimeIndex normalizado para 'data' (sem hora/tz)."""
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()]
    d.index = pd.to_datetime(d.index.date)
    return d


def fetch_history_cached(ticker: str,
                         start: str = CFG.START,
                         end: str | None = CFG.END) -> pd.DataFrame:
    """
    Carrega histórico do cache. Se já existir, baixa apenas o trecho faltante
    (última_data + 1 dia .. end) e faz merge. Salva cache atualizado.
    """
    p = cache_path("hist", ticker)
    df_cached = load_parquet(p)

    def _download(_start: str, _end: str | None) -> pd.DataFrame:
        df = yf.download(ticker, start=_start, end=_end,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Adj Close": "AdjClose"})
        df = _ensure_dtindex(df).sort_index()
        return df

    if df_cached is None or df_cached.empty:
        # Primeira vez: baixa tudo
        df = _download(start, end)
        if df.empty:
            return pd.DataFrame()
        save_parquet(df, p)
        time.sleep(CFG.SLEEP_BETWEEN)
        return df

    # Incremental
    df_cached = _ensure_dtindex(df_cached).sort_index()
    last_dt = df_cached.index.max()
    need_start_dt = last_dt + timedelta(days=1)
    today = pd.Timestamp.today().normalize()

    if need_start_dt <= today:
        tail = _download(need_start_dt.strftime("%Y-%m-%d"), end)
        if not tail.empty:
            df = pd.concat([df_cached, tail]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            save_parquet(df, p)
            time.sleep(CFG.SLEEP_BETWEEN)
            return df

    # Nada novo
    return df_cached


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas simples (Open_xxx etc.) quando houver MultiIndex."""
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = ['_'.join([str(c) for c in tup if c]) for tup in d.columns]
    else:
        d.columns = [str(c) for c in d.columns]
    return d


def _pick_ohlcv_cols(df: pd.DataFrame):
    """Encontra colunas Open/High/Low/Close/Volume mesmo quando vêm sufixadas."""
    cols = [str(c) for c in df.columns]
    low_cols = [c.lower() for c in cols]

    def find(prefix: str):
        for c, lc in zip(cols, low_cols):
            if lc.startswith(prefix):
                return c
        return None

    open_col  = find("open")
    high_col  = find("high")
    low_col   = find("low")
    close_col = find("close")
    vol_col   = find("vol")
    return open_col, high_col, low_col, close_col, vol_col


def _symbol_from_ticker(ticker: str) -> str:
    """MetaStock não gosta de '.', troca por '_' e usa maiúsculas."""
    return ticker.replace(".", "_").upper()


def hist_to_ms7_lines(ticker: str,
                      df_hist: pd.DataFrame,
                      min_date_exclusive: pd.Timestamp | None = None) -> list[str]:
    """
    Converte um histórico em linhas MS7 (SYMBOL,YYMMDD,Open,High,Low,Close,Volume).
    Se min_date_exclusive for passado, exporta apenas datas > min_date_exclusive.
    """
    if df_hist is None or df_hist.empty:
        return []

    df = _flatten_cols(df_hist)
    open_col, high_col, low_col, close_col, vol_col = _pick_ohlcv_cols(df)
    if not all([open_col, high_col, low_col, close_col]):
        return []

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            return []
    df = df[~df.index.isna()].sort_index()

    out = pd.DataFrame({
        "Open":   pd.to_numeric(df[open_col], errors="coerce"),
        "High":   pd.to_numeric(df[high_col], errors="coerce"),
        "Low":    pd.to_numeric(df[low_col], errors="coerce"),
        "Close":  pd.to_numeric(df[close_col], errors="coerce"),
        "Volume": pd.to_numeric(df[vol_col], errors="coerce") if (vol_col and (vol_col in df.columns)) else 0,
    }, index=df.index)

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
    out = out[out["Close"] > 0]

    if min_date_exclusive is not None:
        out = out[out.index > min_date_exclusive]

    if out.empty:
        return []

    dates = out.index.strftime("%y%m%d")
    symbol = _symbol_from_ticker(ticker)

    lines = [
        f"{symbol},{d},{o:.4f},{h:.4f},{l:.4f},{c:.4f},{int(v) if not pd.isna(v) else 0}"
        for d, o, h, l, c, v in zip(dates, out["Open"], out["High"], out["Low"], out["Close"], out["Volume"])
    ]
    return lines


def load_existing_last_dates(ms7_path: str) -> dict[str, pd.Timestamp]:
    """
    Lê o MS7 existente e retorna {SYMBOL: última_data (Timestamp)}.
    Data no arquivo está como YYMMDD.
    """
    last: dict[str, pd.Timestamp] = {}
    if not os.path.exists(ms7_path):
        return last

    with open(ms7_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            sym, yymmdd = parts[0], parts[1]
            try:
                dt = pd.to_datetime(datetime.strptime(yymmdd, "%y%m%d").date())
            except Exception:
                continue
            if (sym not in last) or (dt > last[sym]):
                last[sym] = dt
    return last


def _read_tickers(csv_path: str) -> list[str]:
    """
    Lê a coluna 'ticker' como string, limpa espaços e deduplica corretamente.
    Evita o bug de transformar strings em sets de caracteres.
    """
    tk_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    col = tk_df.columns[0] if "ticker" not in tk_df.columns else "ticker"
    raw = (tk_df[col] if col == "ticker" else tk_df.iloc[:, 0])

    tickers = {str(x).strip() for x in raw if str(x).strip()}
    tickers = [t for t in tickers if t not in {"-", "_", "."}]
    return sorted(tickers)


def main():
    # Lê a lista de tickers (corrigido)
    tickers = _read_tickers(CFG.TICKERS_CSV)
    print(f"Tickers: {len(tickers)}")

    out_path = os.path.join(CFG.OUT_DIR, CFG.OUT_FILE)  # path COMPLETO
    last_dates = load_existing_last_dates(out_path)      # {SYMBOL: last_dt}

    new_lines_all: list[str] = []

    for tk in tqdm(tickers, desc="Atualizando cache e gerando MS7 (novas datas)"):
        try:
            hist = fetch_history_cached(tk, CFG.START, CFG.END)
            sym = _symbol_from_ticker(tk)
            min_dt = last_dates.get(sym, None)  # Timestamp ou None
            lines = hist_to_ms7_lines(tk, hist, min_date_exclusive=min_dt)
            if lines:
                new_lines_all.extend(lines)
        except Exception as e:
            print(f"[erro] {tk}: {e}")

    # Se o arquivo não existe e não houve novas linhas (ex.: primeiro run),
    # então exporta tudo do histórico para todos os tickers.
    if (not os.path.exists(out_path)) and (len(new_lines_all) == 0):
        print("[info] Arquivo ainda não existe; exportando todo o histórico inicial.")
        for tk in tqdm(tickers, desc="Exportando histórico completo (primeiro run)"):
            try:
                hist = fetch_history_cached(tk, CFG.START, CFG.END)
                lines = hist_to_ms7_lines(tk, hist, min_date_exclusive=None)
                if lines:
                    new_lines_all.extend(lines)
            except Exception as e:
                print(f"[erro] {tk}: {e}")

    # Gravação
    if new_lines_all:
        mode = "a" if os.path.exists(out_path) else "w"
        with open(out_path, mode, encoding="utf-8", newline="\n") as f:
            for ln in new_lines_all:
                f.write(ln + "\n")
        print(f"\nNovas linhas MS7 adicionadas: {len(new_lines_all)}")
    else:
        print("\nNada novo para adicionar. Arquivo já está atualizado.")

    print(f"Arquivo: {out_path}")
    print("\nNo MetaStock Downloader:")
    print("  Source: ASCII Text")
    print("  Fields: Symbol,Date,Open,High,Low,Close,Volume")
    print("  Date format: YYMMDD | Delimiter: , | Header lines: 0")


if __name__ == "__main__":
    main()
