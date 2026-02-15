from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.tesourotransparente.gov.br/ckan/dataset/f0468ecc-ae97-4287-89c2-6d8139fb4343/resource/e5f90e3a-8f8d-4895-9c56-4bb2f7877920/download/vendastesourodireto.csv"

# ✅ Paths sempre relativos ao diretório deste arquivo
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PARQUET_PATH = DATA_DIR / "vendastesouro.parquet"


def fetch_csv(timeout: int = 60) -> pd.DataFrame:
    """
    Baixa o CSV (vendastesourodireto.csv).
    Nota: usamos decimal="," pois o CSV vem em pt-BR.
    """
    r = requests.get(URL, timeout=timeout)
    r.raise_for_status()
    txt = r.content.decode("latin1", errors="replace")
    df = pd.read_csv(StringIO(txt), sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def _to_num(s: pd.Series) -> pd.Series:
    """
    Converte para numérico sem destruir floats já parseados pelo read_csv(decimal=",").
    BUG anterior: ao converter float -> str e remover ".", o ponto decimal era perdido.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s = s.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colunas confirmadas no CSV:
      - Tipo Titulo
      - Vencimento do Titulo
      - Data Venda
      - PU
      - Quantidade
      - Valor (não usado)
    """
    out = pd.DataFrame()
    out["data_venda"] = pd.to_datetime(df["Data Venda"], dayfirst=True, errors="coerce")
    out["tipo_titulo"] = df["Tipo Titulo"].astype(str).str.strip()
    out["vencimento"] = pd.to_datetime(df["Vencimento do Titulo"], dayfirst=True, errors="coerce")
    out["ano_vencimento"] = out["vencimento"].dt.year.astype("Int64")

    out["pu"] = _to_num(df["PU"])
    out["quantidade"] = _to_num(df["Quantidade"])

    out = out.dropna(subset=["data_venda", "tipo_titulo", "vencimento", "ano_vencimento", "pu", "quantidade"])
    out["ano_vencimento"] = out["ano_vencimento"].astype(int)

    # Ordenação por Data Venda (exigência)
    out = out.sort_values(["tipo_titulo", "ano_vencimento", "data_venda"]).reset_index(drop=True)
    return out


def load_parquet() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
        df["vencimento"] = pd.to_datetime(df["vencimento"], errors="coerce")
        return df
    return pd.DataFrame(columns=["data_venda", "tipo_titulo", "vencimento", "ano_vencimento", "pu", "quantidade"])


def update_parquet(new_df: pd.DataFrame) -> dict:
    """
    Atualização incremental:
      - concat old + new
      - dedup
      - salva parquet
    """
    old = load_parquet()

    merged = pd.concat([old, new_df], ignore_index=True)

    merged = merged.drop_duplicates(
        subset=["data_venda", "tipo_titulo", "vencimento", "pu", "quantidade"],
        keep="last",
    )

    merged = merged.sort_values(["tipo_titulo", "ano_vencimento", "data_venda"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(PARQUET_PATH, index=False)

    ultima_data_global = pd.to_datetime(merged["data_venda"]).max() if len(merged) else None

    return {
        "parquet_path": str(PARQUET_PATH),
        "linhas_antes": int(len(old)),
        "linhas_novas_baixadas": int(len(new_df)),
        "linhas_total_depois": int(len(merged)),
        "linhas_inseridas_liquidas": int(len(merged) - len(old)),
        "ultima_data_global": str(ultima_data_global.date()) if ultima_data_global is not None else None,
    }


if __name__ == "__main__":
    print("Baixando CSV do Tesouro Direto...")
    raw = fetch_csv()

    print("Normalizando dados...")
    df = normalize(raw)
    print(f"Linhas normalizadas: {len(df)}")

    print("Atualizando Parquet incremental...")
    stats = update_parquet(df)

    print("Parquet gerado/atualizado com sucesso!")
    print(stats)
