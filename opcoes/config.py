"""Configuração central: caminhos dos arquivos e parâmetros ajustáveis via `.env`."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _carregar_env(path: Path) -> None:
    """Leitor mínimo de `.env` (CHAVE=valor), sem dependência externa."""
    if not path.exists():
        return
    for linha in path.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.split(" #", 1)[0].strip().strip('"').strip("'"))


_carregar_env(ROOT / ".env")


def _f(chave: str, padrao: float) -> float:
    try:
        return float(os.getenv(chave, padrao))
    except ValueError:
        return padrao


def _i(chave: str, padrao: int) -> int:
    try:
        return int(os.getenv(chave, padrao))
    except ValueError:
        return padrao


DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()
CADEIAS_DIR = DATA_DIR / "cadeias"  # arquivo diário das cadeias (um parquet por pregão)

ARQ_UNIVERSO = DATA_DIR / "universo.parquet"  # ranking de liquidez de todos os ativos com opções
ARQ_ACOMPANHADOS = DATA_DIR / "ativos_acompanhados.json"
ARQ_PRECOS = DATA_DIR / "precos_ativos.parquet"  # histórico diário (Yahoo Finance)
ARQ_CADEIA = DATA_DIR / "cadeia_atual.parquet"  # última cadeia enriquecida (IV, gregas, liquidez)
ARQ_VOL_HIST = DATA_DIR / "volatilidade_hist.parquet"  # IV ATM e vol. histórica por pregão
ARQ_OPERACOES = DATA_DIR / "operacoes.parquet"
ARQ_PERNAS = DATA_DIR / "pernas.parquet"
ARQ_HIST_PERNAS = DATA_DIR / "historico_pernas.parquet"  # preço diário de cada opção usada em operações
ARQ_META = DATA_DIR / "meta.json"

HIST_INICIO = os.getenv("HIST_INICIO", "2018-01-01")

# Conta no opcoes.net.br (opcional): com login o site entrega ofertas de compra/venda e gregas.
OPCOESNET_CPF = os.getenv("OPCOESNET_CPF", "").strip()
OPCOESNET_SENHA = os.getenv("OPCOESNET_SENHA", "").strip()
TAXA_JUROS = os.getenv("TAXA_JUROS", "auto")  # "auto" = Selic meta do BCB; ou valor anual, ex.: 0.15
TAXA_JUROS_PADRAO = _f("TAXA_JUROS_PADRAO", 0.15)

# Custos operacionais estimados (aplicados a cada ordem)
CUSTO_PCT = _f("CUSTO_PCT", 0.035)  # % sobre o volume financeiro (emolumentos/registro B3 aprox.)
CORRETAGEM_ORDEM = _f("CORRETAGEM_ORDEM", 0.0)  # R$ por ordem
CUSTO_TICK = _f("CUSTO_TICK", 0.01)  # R$ por opção: meio spread estimado ao montar (sem book de ofertas)
PREMIO_MIN = _f("PREMIO_MIN", 0.05)  # prêmio justo mínimo por série nas estratégias (evita "moedinhas")

# Universo e coleta
ATIVOS_PADRAO = os.getenv(
    "ATIVOS_PADRAO", "PETR4,VALE3,ITUB4,BBDC4,BBAS3,BOVA11,B3SA3,ABEV3,PRIO3,ITSA4"
).split(",")
UNIVERSO_TOP_N = _i("UNIVERSO_TOP_N", 25)
VENC_MAX_DIAS = _i("VENC_MAX_DIAS", 760)  # ignora séries mais longas que isso
PAUSA_REQ = _f("PAUSA_REQ", 5.0)  # segundos entre requisições ao opcoes.net.br (o site limita por IP)
ESPERA_MAX = _f("ESPERA_MAX", 120.0)  # se o site pedir espera maior (Retry-After), interrompe a coleta
UNIVERSO_VARREDURA = _i("UNIVERSO_VARREDURA", 40)  # nº de ativos (maiores em contratos em aberto) com volume medido
MANTER_CADEIAS_DIAS = _i("MANTER_CADEIAS_DIAS", 400)

# Liquidez de cada série (último pregão)
LIQ_ALTA_NEGOCIOS = _i("LIQ_ALTA_NEGOCIOS", 20)
LIQ_ALTA_VOLUME = _f("LIQ_ALTA_VOLUME", 50_000.0)
LIQ_MEDIA_NEGOCIOS = _i("LIQ_MEDIA_NEGOCIOS", 5)
LIQ_MEDIA_DIAS = _i("LIQ_MEDIA_DIAS", 2)  # negociada há no máximo N pregões

LIQ_ROTULOS = {0: "Sem negócios", 1: "Baixa", 2: "Média", 3: "Alta"}


def garantir_pastas() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CADEIAS_DIR.mkdir(parents=True, exist_ok=True)
