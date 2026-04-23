import argparse
import json
from io import StringIO
import requests
import pandas as pd
import re

def _flatten_columns(columns) -> list[str]:
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if str(p).strip() and not str(p).startswith("Unnamed")]
            flat.append(" | ".join(parts) if parts else "data")
        else:
            txt = str(col).strip()
            flat.append("data" if txt.startswith("Unnamed") else txt)
    return flat


def _normalize_key(label: str) -> str:
    txt = str(label).strip().lower()
    txt = txt.replace("vol. fin.", "volume_fin")
    txt = txt.replace("vol. fin", "volume_fin")
    txt = txt.replace("negócios", "negocios")
    txt = txt.replace("cotação não ajustada", "cotacao")
    txt = txt.replace("volatilidade implícita", "vol_implicita")
    txt = txt.replace(" ", "_")
    txt = txt.replace(".", "")
    txt = txt.replace("-", "_")
    txt = txt.replace("|", "_")
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt.strip("_")


def _extract_history_table(html: str, ativo_objeto: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html), decimal=",", thousands=".")
    for table in tables:
        df = table.copy()
        df.columns = _flatten_columns(df.columns)
        first_col = df.columns[0]
        first_series = pd.to_datetime(df[first_col], errors="coerce", dayfirst=True)
        if first_series.notna().sum() == 0:
            continue

        renamed = {}
        for col in df.columns:
            key = _normalize_key(col)
            if key == "data":
                renamed[col] = "data"
            elif "cotacao" in key and "min" in key:
                renamed[col] = "opcao_min"
            elif "cotacao" in key and "pri" in key:
                renamed[col] = "opcao_pri"
            elif "cotacao" in key and "med" in key:
                renamed[col] = "opcao_med"
            elif "cotacao" in key and "ult" in key:
                renamed[col] = "opcao_ult"
            elif "cotacao" in key and "max" in key:
                renamed[col] = "opcao_max"
            elif "cotacao" in key and "negocios" in key:
                renamed[col] = "opcao_negocios"
            elif "cotacao" in key and "volume_fin" in key:
                renamed[col] = "opcao_volume_fin"
            elif "vol_implicita" in key and "min" in key:
                renamed[col] = "vi_min"
            elif "vol_implicita" in key and "pri" in key:
                renamed[col] = "vi_pri"
            elif "vol_implicita" in key and "med" in key:
                renamed[col] = "vi_med"
            elif "vol_implicita" in key and "ult" in key:
                renamed[col] = "vi_ult"
            elif "vol_implicita" in key and "max" in key:
                renamed[col] = "vi_max"
            elif ativo_objeto.lower() in key and "min" in key:
                renamed[col] = "ativo_min"
            elif ativo_objeto.lower() in key and "abe" in key:
                renamed[col] = "ativo_abe"
            elif ativo_objeto.lower() in key and "med" in key:
                renamed[col] = "ativo_med"
            elif ativo_objeto.lower() in key and "ult" in key:
                renamed[col] = "ativo_ult"
            elif ativo_objeto.lower() in key and "max" in key:
                renamed[col] = "ativo_max"

        out = df.rename(columns=renamed).copy()
        out["data"] = pd.to_datetime(out["data"], errors="coerce", dayfirst=True)
        out = out[out["data"].notna()].sort_values("data", ascending=False).reset_index(drop=True)
        if not out.empty and "opcao_ult" in out.columns:
            return out
    return pd.DataFrame()


def consultar_premio_opcao(ticker: str):
    url = f"https://opcoes.net.br/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    html = resp.text

    # Confere se a página realmente parece ser da opção
    if ticker not in html:
        raise ValueError(f"Ticker {ticker} não encontrado na página.")

    # Extrai título com strike/vencimento
    titulo_match = re.search(
        rf"{ticker}\s*-\s*(CALL|PUT)\s*de\s*([A-Z0-9]+)\s*-\s*Strike\s*R\$?\s*([\d,]+)\s*-\s*Vencimento\s*([\d/]+)",
        html,
        re.IGNORECASE
    )

    info = {}
    if titulo_match:
        info = {
            "ticker": ticker,
            "tipo": titulo_match.group(1).upper(),
            "ativo_objeto": titulo_match.group(2).upper(),
            "strike": titulo_match.group(3),
            "vencimento": titulo_match.group(4),
        }
    else:
        info = {"ticker": ticker}

    ativo_objeto = info.get("ativo_objeto", "")
    df = _extract_history_table(html, ativo_objeto)

    if df.empty:
        return {
            **info,
            "observacao": "Não encontrei tabela de cotações visível no HTML. Pode exigir login/plano ou a estrutura da página mudou."
        }

    ultima = df.iloc[0].to_dict()
    ultima["data"] = ultima["data"].strftime("%d/%m/%Y")

    historico = df.copy()
    historico["data"] = historico["data"].dt.strftime("%d/%m/%Y")

    return {
        **info,
        "data_ref": ultima["data"],
        "premio_ultimo": ultima.get("opcao_ult"),
        "premio_compra_ref": ultima.get("opcao_pri"),
        "negocios": ultima.get("opcao_negocios"),
        "volume_fin": ultima.get("opcao_volume_fin"),
        "linha_mais_recente": ultima,
        "historico": historico.to_dict(orient="records"),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consulta prêmio e histórico básico de uma opção no opcoes.net.br."
    )
    parser.add_argument(
        "ticker",
        help="Ticker da opção a consultar, por exemplo: PETRV460",
    )
    args = parser.parse_args()

    resultado = consultar_premio_opcao(args.ticker.upper())
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
