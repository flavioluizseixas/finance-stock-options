from __future__ import annotations

from io import StringIO
from pathlib import Path
import pandas as pd
import requests

# =========================
# URLs oficiais (Tesouro Transparente)
# =========================
URL_VENDAS = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "f0468ecc-ae97-4287-89c2-6d8139fb4343/resource/"
    "e5f90e3a-8f8d-4895-9c56-4bb2f7877920/download/vendastesourodireto.csv"
)

URL_PRECOTAXA = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PARQUET_VENDAS = DATA_DIR / "vendastesouro.parquet"
PARQUET_PRECOTAXA = DATA_DIR / "precotaxa.parquet"


# =========================
# I/O helpers
# =========================
def fetch_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    txt = r.content.decode("latin1", errors="replace")
    df = pd.read_csv(StringIO(txt), sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def _to_num(s: pd.Series) -> pd.Series:
    # preserva numérico se já vier parseado pelo read_csv(decimal=",")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def update_parquet_incremental(path: Path, new_df: pd.DataFrame, dedup_cols: list[str]) -> dict:
    old = load_parquet(path)

    # evita warning/edge de concat com vazio
    if old.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([old, new_df], ignore_index=True)

    merged = merged.drop_duplicates(subset=dedup_cols, keep="last")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)

    return {
        "path": str(path),
        "linhas_antes": int(len(old)),
        "linhas_novas": int(len(new_df)),
        "linhas_total": int(len(merged)),
        "linhas_inseridas_liquidas": int(len(merged) - len(old)),
    }


# =========================
# Normalização: VENDAS
# =========================
def normalize_vendas(df: pd.DataFrame) -> pd.DataFrame:
    # Esperado (vendas):
    # Tipo Titulo | Vencimento do Titulo | Data Venda | PU | Quantidade | Valor
    out = pd.DataFrame()
    out["data_venda"] = pd.to_datetime(df["Data Venda"], dayfirst=True, errors="coerce")
    out["tipo_titulo"] = df["Tipo Titulo"].astype(str).str.strip()
    out["vencimento"] = pd.to_datetime(df["Vencimento do Titulo"], dayfirst=True, errors="coerce")
    out["ano_vencimento"] = out["vencimento"].dt.year.astype("Int64")

    out["pu_venda"] = _to_num(df["PU"])
    out["quantidade"] = _to_num(df["Quantidade"])

    out = out.dropna(subset=["data_venda", "tipo_titulo", "vencimento", "ano_vencimento", "pu_venda", "quantidade"])
    out["ano_vencimento"] = out["ano_vencimento"].astype(int)

    out = out.sort_values(["tipo_titulo", "ano_vencimento", "data_venda"]).reset_index(drop=True)
    return out


def update_vendas() -> dict:
    raw = fetch_csv(URL_VENDAS)
    df = normalize_vendas(raw)

    stats = update_parquet_incremental(
        PARQUET_VENDAS,
        df,
        dedup_cols=["data_venda", "tipo_titulo", "vencimento", "pu_venda", "quantidade"],
    )
    stats["ultima_data_vendas"] = str(pd.to_datetime(df["data_venda"]).max().date()) if len(df) else None
    return stats


# =========================
# Normalização: PREÇO/TAXA
# =========================
def normalize_precotaxa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Layout confirmado por você:
    ['Tipo Titulo', 'Data Vencimento', 'Data Base',
     'Taxa Compra Manha', 'Taxa Venda Manha',
     'PU Compra Manha', 'PU Venda Manha', 'PU Base Manha']
    """
    required = [
        "Tipo Titulo",
        "Data Vencimento",
        "Data Base",
        "Taxa Compra Manha",
        "Taxa Venda Manha",
        "PU Compra Manha",
        "PU Venda Manha",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Não encontrei colunas esperadas em precotaxatesourodireto.csv. "
            f"Faltando: {missing}. Colunas: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["data_base"] = pd.to_datetime(df["Data Base"], dayfirst=True, errors="coerce")
    out["tipo_titulo"] = df["Tipo Titulo"].astype(str).str.strip()
    out["vencimento"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True, errors="coerce")
    out["ano_vencimento"] = out["vencimento"].dt.year.astype("Int64")

    out["taxa_compra"] = _to_num(df["Taxa Compra Manha"])
    out["taxa_venda"] = _to_num(df["Taxa Venda Manha"])
    out["pu_compra"] = _to_num(df["PU Compra Manha"])
    out["pu_venda"] = _to_num(df["PU Venda Manha"])

    out = out.dropna(
        subset=[
            "data_base",
            "tipo_titulo",
            "vencimento",
            "ano_vencimento",
            "taxa_compra",
            "taxa_venda",
            "pu_compra",
            "pu_venda",
        ]
    )
    out["ano_vencimento"] = out["ano_vencimento"].astype(int)

    out = out.sort_values(["tipo_titulo", "ano_vencimento", "data_base"]).reset_index(drop=True)
    return out


def update_precotaxa() -> dict:
    raw = fetch_csv(URL_PRECOTAXA)
    df = normalize_precotaxa(raw)

    stats = update_parquet_incremental(
        PARQUET_PRECOTAXA,
        df,
        dedup_cols=[
            "data_base",
            "tipo_titulo",
            "vencimento",
            "taxa_compra",
            "taxa_venda",
            "pu_compra",
            "pu_venda",
        ],
    )
    stats["ultima_data_precotaxa"] = str(pd.to_datetime(df["data_base"]).max().date()) if len(df) else None
    return stats


def update_all() -> dict:
    return {
        "vendas": update_vendas(),
        "precotaxa": update_precotaxa(),
    }


if __name__ == "__main__":
    s = update_all()
    print("Atualização concluída:")
    print(s)
