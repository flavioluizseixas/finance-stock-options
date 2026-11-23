"""Gera, avalia e ranqueia estruturas com opções.

Para cada ativo e vencimento monta todas as combinações plausíveis (com séries líquidas) e calcula:

* prêmio, capital exigido, lucro/perda máximos e pontos de equilíbrio (exatos, payoff linear por partes);
* distribuição do preço no vencimento por simulação histórica filtrada (caudas reais do ativo,
  escaladas pela vol atual, média ajustada ao forward);
* probabilidade de lucro (POP), valor esperado, CVaR 5% e **vantagem esperada acima do CDI**
  (valor esperado + rendimento do caixa recebido − custo do caixa aplicado), anualizada.

A vantagem é positiva quando o prêmio de mercado paga mais que o risco realista da estrutura — o
critério principal de ranqueamento. Liquidez é filtro rígido (todas as pernas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .precificacao import DU_ANO, amostras_preco_final, vol_referencia

MAX_PERNAS = 4


@dataclass(frozen=True)
class Estrategia:
    chave: str
    nome: str
    visoes: tuple[str, ...]
    descricao: str


ESTRATEGIAS: dict[str, Estrategia] = {
    e.chave: e
    for e in [
        Estrategia("venda_coberta", "Venda coberta", ("Alta", "Neutra"),
                   "Compra a ação e vende uma call. Renda com a ação em carteira; limita o ganho acima do strike."),
        Estrategia("venda_put", "Venda de put (caixa garantido)", ("Alta", "Neutra"),
                   "Vende uma put deixando em caixa o valor do strike. Recebe prêmio; pode ser exercido e comprar a ação."),
        Estrategia("trava_alta_put", "Trava de alta com puts (crédito)", ("Alta", "Neutra"),
                   "Vende put de strike maior e compra put de strike menor. Crédito na montagem, risco limitado."),
        Estrategia("trava_baixa_call", "Trava de baixa com calls (crédito)", ("Baixa", "Neutra"),
                   "Vende call de strike menor e compra call de strike maior. Crédito na montagem, risco limitado."),
        Estrategia("trava_alta_call", "Trava de alta com calls (débito)", ("Alta",),
                   "Compra call de strike menor e vende call de strike maior. Débito pequeno, ganho limitado."),
        Estrategia("trava_baixa_put", "Trava de baixa com puts (débito)", ("Baixa",),
                   "Compra put de strike maior e vende put de strike menor. Débito pequeno, ganho limitado."),
        Estrategia("iron_condor", "Iron condor", ("Neutra",),
                   "Trava de alta com puts + trava de baixa com calls. Lucra se o ativo ficar dentro do intervalo."),
        Estrategia("collar", "Collar (ação protegida)", ("Alta", "Neutra"),
                   "Ação + put comprada (proteção) financiada por call vendida. Perda e ganho limitados."),
        Estrategia("compra_call", "Compra de call", ("Alta",),
                   "Aposta direcional na alta com perda limitada ao prêmio."),
        Estrategia("compra_put", "Compra de put", ("Baixa",),
                   "Aposta direcional na queda (ou proteção) com perda limitada ao prêmio."),
        Estrategia("straddle", "Compra de straddle/strangle", ("Volatilidade",),
                   "Compra call e put. Lucra com movimento forte para qualquer lado."),
    ]
}

VISOES = ["Todas", "Alta", "Baixa", "Neutra", "Volatilidade"]


# ---------------------------------------------------------------------------
# Representação vetorizada de um conjunto de candidatos (até 4 pernas + ação)
# ---------------------------------------------------------------------------

@dataclass
class Bloco:
    chave: str
    idx: np.ndarray  # (n, 4) posição da série em g (-1 = perna vazia)
    q: np.ndarray  # (n, 4) quantidade relativa com sinal (+ compra, − venda)
    acao: np.ndarray  # (n,) quantidade relativa de ações

    def __len__(self) -> int:
        return len(self.acao)


def _bloco(chave: str, pernas: list[tuple[np.ndarray, float]], acao: float = 0.0) -> Bloco:
    n = len(pernas[0][0]) if pernas else 0
    idx = np.full((n, MAX_PERNAS), -1, dtype=int)
    q = np.zeros((n, MAX_PERNAS))
    for j, (ix, qq) in enumerate(pernas):
        idx[:, j] = ix
        q[:, j] = qq
    return Bloco(chave, idx, q, np.full(n, acao))


def _pares(k: np.ndarray, largura_max: float) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.triu_indices(len(k), 1)
    ok = (k[b] > k[a]) & (k[b] - k[a] <= largura_max)
    return a[ok], b[ok]


def _gerar(g: pd.DataFrame, S: float, chaves: set[str]) -> list[Bloco]:
    """Todas as combinações para um ativo/vencimento. `g` já filtrado por liquidez, ordenado por strike."""
    tipo = g["tipo"].to_numpy()
    k = g["strike"].to_numpy(float)
    ic = np.flatnonzero(tipo == "CALL")
    ip = np.flatnonzero(tipo == "PUT")
    kc, kp = k[ic], k[ip]
    larg = 0.25 * S
    out: list[Bloco] = []

    def quer(c):
        return c in chaves

    if quer("venda_coberta"):
        s = ic[kc >= 0.9 * S]
        out.append(_bloco("venda_coberta", [(s, -1)], acao=1))
    if quer("venda_put"):
        s = ip[kp <= 1.05 * S]
        out.append(_bloco("venda_put", [(s, -1)]))
    if quer("compra_call"):
        s = ic[(kc >= 0.8 * S) & (kc <= 1.2 * S)]
        out.append(_bloco("compra_call", [(s, 1)]))
    if quer("compra_put"):
        s = ip[(kp >= 0.8 * S) & (kp <= 1.2 * S)]
        out.append(_bloco("compra_put", [(s, 1)]))

    a, b = _pares(kc, larg)  # calls: a < b
    if quer("trava_baixa_call") or quer("iron_condor"):
        m = kc[a] >= 0.95 * S
        out.append(_bloco("trava_baixa_call", [(ic[a[m]], -1), (ic[b[m]], 1)]))
    if quer("trava_alta_call"):
        m = (kc[a] >= 0.85 * S) & (kc[a] <= 1.10 * S)
        out.append(_bloco("trava_alta_call", [(ic[a[m]], 1), (ic[b[m]], -1)]))

    a, b = _pares(kp, larg)  # puts: a < b
    if quer("trava_alta_put") or quer("iron_condor"):
        m = kp[b] <= 1.05 * S
        out.append(_bloco("trava_alta_put", [(ip[b[m]], -1), (ip[a[m]], 1)]))
    if quer("trava_baixa_put"):
        m = (kp[b] >= 0.9 * S) & (kp[b] <= 1.15 * S)
        out.append(_bloco("trava_baixa_put", [(ip[b[m]], 1), (ip[a[m]], -1)]))

    if quer("collar"):
        cc = ic[(kc >= S) & (kc <= 1.25 * S)]
        pp = ip[(kp >= 0.75 * S) & (kp <= S)]
        if len(cc) and len(pp):
            x, y = np.meshgrid(cc, pp, indexing="ij")
            out.append(_bloco("collar", [(x.ravel(), -1), (y.ravel(), 1)], acao=1))
    if quer("straddle"):
        cc = ic[(kc >= 0.97 * S) & (kc <= 1.15 * S)]
        pp = ip[(kp >= 0.85 * S) & (kp <= 1.03 * S)]
        if len(cc) and len(pp):
            x, y = np.meshgrid(cc, pp, indexing="ij")
            x, y = x.ravel(), y.ravel()
            m = (k[y] <= k[x]) & (k[x] - k[y] <= 0.15 * S)
            out.append(_bloco("straddle", [(x[m], 1), (y[m], 1)]))
    return [bl for bl in out if len(bl)]


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def _payoff(K, tp, q, px, acao, S0, ST) -> np.ndarray:
    """P&L no vencimento (sem custos) para cada candidato (linhas) e preço final (colunas)."""
    pay = acao[:, None] * (ST[None, :] - S0)
    for j in range(K.shape[1]):
        if not np.any(tp[:, j]):
            continue
        intr = np.maximum(tp[:, j, None] * (ST[None, :] - K[:, j, None]), 0.0)
        pay += q[:, j, None] * (intr - px[:, j, None])
    return pay


COLS_CADEIA = ["simbolo", "tipo", "strike", "teorico", "ultimo", "fonte_preco", "preco_ref", "iv_modelo", "liquidez",
               "delta", "theta", "vega"]


def _avaliar(bl: Bloco, g: dict[str, np.ndarray], S0: float, du: int, r: float, amostras: np.ndarray) -> pd.DataFrame:
    """`g`: colunas da cadeia do vencimento como arrays (ver COLS_CADEIA)."""
    vazio = bl.idx < 0
    ix = np.where(vazio, 0, bl.idx)

    def col(nome, preenche=0.0):
        return np.where(vazio, preenche, g[nome][ix])

    K = col("strike").astype(float)
    tp = np.where(vazio, 0, np.where(col("tipo", "") == "CALL", 1, -1)).astype(float)
    # Preço justo do smile (consistente entre strikes). O último negócio de cada série ocorre em
    # horários diferentes e, combinado entre pernas, cria "arbitragens" falsas.
    px = col("teorico").astype(float)
    ult = np.where(col("fonte_preco", "") == "último", col("ultimo", np.nan).astype(float), np.nan)
    q, acao = bl.q, bl.acao
    T = du / DU_ANO

    custo = (
        config.CUSTO_PCT / 100 * (np.abs(q * px).sum(1) + acao * S0)
        + config.CORRETAGEM_ORDEM * ((~vazio).sum(1) + (acao != 0))
        + config.CUSTO_TICK * np.abs(q).sum(1)
    )
    premio = -(q * px).sum(1) - acao * S0  # fluxo de caixa na montagem (+ recebe)
    premio_ult = -(q * np.where(vazio, 0, ult)).sum(1) - acao * S0

    pl = _payoff(K, tp, q, px, acao, S0, amostras) - custo[:, None]
    ev = pl.mean(1)
    dp_amostra = pl.std(1)
    pop = np.clip((pl > 0).mean(1), 0.01, 0.99)  # resolução do modelo: não afirma 0% nem 100%
    n5 = max(1, int(0.05 * pl.shape[1]))
    cvar = np.partition(pl, n5, axis=1)[:, :n5].mean(1)
    vantagem = ev + premio * (np.exp(r * T) - 1)

    # extremos exatos: payoff linear por partes → avalia em 0, nos strikes e muito acima
    hi = 10 * max(S0, float(np.nanmax(K)))
    nos = np.concatenate([np.zeros((len(acao), 1)), K, np.full((len(acao), 1), hi)], axis=1)
    pay_nos = acao[:, None] * (nos - S0)
    for j in range(MAX_PERNAS):
        intr = np.maximum(tp[:, j, None] * (nos - K[:, j, None]), 0.0)
        pay_nos += q[:, j, None] * (intr - px[:, j, None])
    pay_nos -= custo[:, None]
    inclin = acao + (q * (tp == 1)).sum(1)  # inclinação para S → ∞
    max_lucro = np.where(inclin > 1e-9, np.inf, pay_nos.max(1))
    max_perda = np.where(inclin < -1e-9, -np.inf, pay_nos.min(1))
    capital = np.maximum(np.maximum(-max_perda, -premio), 1e-9)
    # Piso de risco: a amostra histórica é finita (caudas raras podem não aparecer). Considera no mínimo
    # 5% a.a. de oscilação sobre o capital — evita "risco zero" em vendas muito fora do dinheiro.
    dp = np.maximum(dp_amostra, 0.05 * capital * np.sqrt(T))

    # pontos de equilíbrio
    grade = np.linspace(0.4 * S0, 1.8 * S0, 561)
    pg = _payoff(K, tp, q, px, acao, S0, grade) - custo[:, None]
    sinal = np.sign(pg)
    troca = (sinal[:, :-1] * sinal[:, 1:]) < 0
    be1 = np.full(len(acao), np.nan)
    be2 = np.full(len(acao), np.nan)
    for i in np.flatnonzero(troca.any(1)):
        js = np.flatnonzero(troca[i])[:2]
        xs = [grade[j] - pg[i, j] * (grade[j + 1] - grade[j]) / (pg[i, j + 1] - pg[i, j]) for j in js]
        be1[i] = xs[0]
        if len(xs) > 1:
            be2[i] = xs[1]

    liq = np.where(vazio, 9, col("liquidez", 9)).min(1)
    teorico = ((col("fonte_preco", "") == "teórico") & ~vazio).any(1)
    gregas = {n: (q * col(n).astype(float)).sum(1) for n in ("delta", "theta", "vega")}
    gregas["delta"] = gregas["delta"] + acao

    est = ESTRATEGIAS[bl.chave]
    simb, tipo = col("simbolo", ""), col("tipo", "")
    ref, ivm = col("preco_ref").astype(float), col("iv_modelo").astype(float)
    pernas = {}
    for j in range(MAX_PERNAS):
        v = vazio[:, j]
        pernas.update({
            f"s{j}": np.where(v, "", simb[:, j]), f"t{j}": np.where(v, "", tipo[:, j]),
            f"k{j}": np.where(v, np.nan, K[:, j]), f"q{j}": q[:, j], f"p{j}": np.where(v, np.nan, px[:, j]),
            f"u{j}": np.where(v, np.nan, ref[:, j]), f"iv{j}": np.where(v, np.nan, ivm[:, j]),
        })
    return pd.DataFrame(
        {
            "chave": bl.chave,
            "estrategia": est.nome,
            "premio": premio,
            "premio_ultimo": premio_ult,
            "capital": capital,
            "max_lucro": max_lucro,
            "max_perda": max_perda,
            "be1": be1,
            "be2": be2,
            "pop": pop,
            "ev": ev,
            "cvar5": cvar,
            "vantagem": vantagem,
            "vantagem_pct": vantagem / capital,
            "vantagem_aa": vantagem / capital * DU_ANO / max(du, 1),
            "indice": vantagem / dp * np.sqrt(DU_ANO / max(du, 1)),
            "retorno_max": np.where(np.isfinite(max_lucro), max_lucro / capital, np.inf),
            "delta": gregas["delta"],
            "theta": gregas["theta"],
            "vega": gregas["vega"],
            "liquidez_min": liq,
            "usa_teorico": teorico,
            "acao_qtd": acao,
            "custo": custo,
            **pernas,
        }
    )


def _condor(put_cred: pd.DataFrame, call_cred: pd.DataFrame, S0: float, top: int = 15) -> pd.DataFrame | None:
    """Combina as melhores travas de crédito OTM de cada lado em iron condors."""
    pc = put_cred[put_cred["k0"] <= S0].nlargest(top, "vantagem")
    cc = call_cred[call_cred["k0"] >= S0].nlargest(top, "vantagem")
    if pc.empty or cc.empty:
        return None
    a = pc.reset_index(drop=True).add_prefix("P_")
    b = cc.reset_index(drop=True).add_prefix("C_")
    return a.merge(b, how="cross")


def ranquear(
    cadeia: pd.DataFrame,
    precos: pd.DataFrame,
    ativos: list[str] | None = None,
    du_min: int = 5,
    du_max: int = 60,
    liquidez_min: int = 2,
    chaves: list[str] | None = None,
    n_amostras: int = 800,
) -> pd.DataFrame:
    """Avalia todas as estruturas e devolve o conjunto de candidatos (sem filtros de exibição)."""
    if cadeia is None or cadeia.empty:
        return pd.DataFrame()
    chaves_set = set(chaves or ESTRATEGIAS)
    cadeia = cadeia[cadeia["sessao"] == cadeia["sessao"].max()]  # só o último pregão (spot e prazos coerentes)
    base = cadeia[
        (cadeia["du"] >= du_min) & (cadeia["du"] <= du_max) & (cadeia["liquidez"] >= liquidez_min)
        & (cadeia["teorico"] >= config.PREMIO_MIN) & cadeia["iv_modelo"].notna()
    ]
    if ativos:
        base = base[base["ativo"].isin(ativos)]
    if base.empty:
        return pd.DataFrame()

    refs = {}  # vol de referência e choques históricos por ativo (calculados uma vez)
    if precos is not None and not precos.empty:
        for a, g in precos[precos["ativo"].isin(base["ativo"].unique())].sort_values("data").groupby("ativo"):
            c = g["fech_ajustado"].fillna(g["fechamento"]).astype(float)
            refs[a] = vol_referencia(np.log(c).diff().to_numpy()[1:])

    saida = []
    for (ativo, venc), g in base.groupby(["ativo", "vencimento"], sort=False):
        g = g.sort_values(["tipo", "strike"]).reset_index(drop=True)
        S0 = float(g["spot"].iloc[0])
        du = int(g["du"].iloc[0])
        r = float(g["taxa_r"].iloc[0])
        g = g[(g["strike"] >= 0.5 * S0) & (g["strike"] <= 1.6 * S0)].reset_index(drop=True)
        if g.empty:
            continue
        iv_atm = float(g.loc[(np.log(g["strike"] / g["forward"])).abs().idxmin(), "iv_modelo"])
        amostras = amostras_preco_final(None, S0, du, r, n=n_amostras, iv_fallback=iv_atm, ref=refs.get(ativo))
        arr = {c: g[c].to_numpy() for c in COLS_CADEIA}

        partes = {}
        for bl in _gerar(g, S0, chaves_set):
            partes[bl.chave] = _avaliar(bl, arr, S0, du, r, amostras)

        if "iron_condor" in chaves_set and "trava_alta_put" in partes and "trava_baixa_call" in partes:
            cx = _condor(partes["trava_alta_put"], partes["trava_baixa_call"], S0)
            if cx is not None and not cx.empty:
                pos = {s: i for i, s in enumerate(g["simbolo"])}
                idx = np.stack([cx[c].map(pos).to_numpy() for c in ("P_s0", "P_s1", "C_s0", "C_s1")], axis=1)
                q = np.tile(np.array([-1.0, 1.0, -1.0, 1.0]), (len(cx), 1))
                partes["iron_condor"] = _avaliar(Bloco("iron_condor", idx, q, np.zeros(len(cx))), arr, S0, du, r, amostras)

        for chave in list(partes):
            if chave not in chaves_set:
                del partes[chave]
        if not partes:
            continue
        df = pd.concat(partes.values(), ignore_index=True)
        df = df[np.isfinite(df["capital"]) & (df["capital"] > 0.01)]
        # mantém os melhores de cada estrutura por critérios distintos (evita tabelas gigantes)
        manter = set()
        for criterio, n in (("indice", 60), ("vantagem_aa", 30), ("pop", 30), ("retorno_max", 30)):
            ordem = df.assign(_o=df[criterio].replace(np.inf, np.nan)).sort_values("_o", ascending=False)
            manter.update(ordem.groupby("chave").head(n).index)
        df = df.loc[sorted(manter)]
        df.insert(0, "ativo", ativo)
        df.insert(1, "vencimento", venc)
        df.insert(2, "du", du)
        df.insert(3, "spot", S0)
        df["iv_atm"] = iv_atm
        ref = refs.get(ativo)
        df["vol_modelo"] = ref[0] * np.sqrt(DU_ANO) if ref is not None and np.isfinite(ref[0]) else np.nan
        saida.append(df)
    if not saida:
        return pd.DataFrame()
    out = pd.concat(saida, ignore_index=True)
    out["visoes"] = out["chave"].map(lambda c: ", ".join(ESTRATEGIAS[c].visoes))
    out["descricao_pernas"] = descrever(out)
    return out


def descrever(df: pd.DataFrame) -> pd.Series:
    """Texto curto das pernas: 'C 100 ações | V PETRL467 44,41'."""
    partes = []
    for row in df.itertuples(index=False):
        txt = []
        if getattr(row, "acao_qtd", 0):
            txt.append("C ação")
        for j in range(MAX_PERNAS):
            s = getattr(row, f"s{j}")
            if s:
                lado = "C" if getattr(row, f"q{j}") > 0 else "V"
                txt.append(f"{lado} {s} ({getattr(row, f't{j}')[0]} {getattr(row, f'k{j}'):.2f})")
        partes.append(" | ".join(txt))
    return pd.Series(partes, index=df.index)


def pernas_da_linha(row: pd.Series, lote: int, preco: str = "mercado") -> list[dict]:
    """Converte um candidato em pernas (formato usado por `operacoes`), com quantidade = lote.

    `preco="mercado"`: último negócio do dia (ou justo, se a série não negociou) — sugestão de entrada.
    `preco="justo"`: preço do smile, a mesma base das métricas do ranking.
    """
    pernas = []
    if row.get("acao_qtd", 0):
        pernas.append(
            {"tipo": "ACAO", "simbolo": row["ativo"], "lado": "COMPRA", "quantidade": int(lote * row["acao_qtd"]),
             "strike": np.nan, "vencimento": pd.NaT, "preco": float(row["spot"]), "iv": np.nan}
        )
    for j in range(MAX_PERNAS):
        s = row.get(f"s{j}")
        if not s:
            continue
        q = float(row[f"q{j}"])
        mercado = row.get(f"u{j}") if preco == "mercado" else np.nan
        pernas.append(
            {
                "tipo": row[f"t{j}"],
                "simbolo": s,
                "lado": "COMPRA" if q > 0 else "VENDA",
                "quantidade": int(round(abs(q) * lote)),
                "strike": float(row[f"k{j}"]),
                "vencimento": pd.Timestamp(row["vencimento"]),
                # sugestão de preço de entrada: último negócio do dia (ou justo, se a série não negociou)
                "preco": round(float(mercado if pd.notna(mercado) else row[f"p{j}"]), 2),
                "iv": float(row[f"iv{j}"]),
            }
        )
    return pernas
