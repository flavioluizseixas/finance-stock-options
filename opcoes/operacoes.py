"""Registro e acompanhamento de operações com opções (REAL ou ESTUDO).

Uma operação tem N pernas (CALL, PUT ou ACAO). Cada perna guarda preço de entrada e, quando
encerrada, preço/data de saída — isso permite rolagens (encerra uma perna, adiciona outra).

Avaliação de uma operação aberta:
* **Desfazer hoje**: zera todas as pernas pelo preço atual (último negócio do pregão ou preço justo do
  smile, se a série não negociou) — mostra o caixa que entra/sai e o resultado final, já com custos.
* **No vencimento**: resultado se o ativo terminar em S (padrão: preço atual), com strikes ajustados
  por proventos e a descrição do que acontece no exercício.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import calendario, config
from .dados import gravar, ler
from .formato import brl, num
from .precificacao import DU_ANO, preco_bs

COLS_OP = [
    "op_id", "modo", "status", "ativo", "estrategia", "data_entrada", "spot_entrada", "custos",
    "proventos", "notas", "data_encerramento", "criado_em", "atualizado_em",
]
COLS_PERNA = [
    "perna_id", "op_id", "tipo", "simbolo", "lado", "quantidade", "strike", "vencimento",
    "preco_entrada", "data_entrada", "preco_saida", "data_saida",
]
MODOS = ["REAL", "ESTUDO"]


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def carregar() -> tuple[pd.DataFrame, pd.DataFrame]:
    ops = ler(config.ARQ_OPERACOES)
    pernas = ler(config.ARQ_PERNAS)
    ops = ops.reindex(columns=COLS_OP) if not ops.empty else pd.DataFrame(columns=COLS_OP)
    pernas = pernas.reindex(columns=COLS_PERNA) if not pernas.empty else pd.DataFrame(columns=COLS_PERNA)
    for c in ["data_entrada", "data_encerramento", "criado_em", "atualizado_em"]:
        ops[c] = pd.to_datetime(ops[c], errors="coerce")
    for c in ["vencimento", "data_entrada", "data_saida"]:
        pernas[c] = pd.to_datetime(pernas[c], errors="coerce")
    for c in ["custos", "proventos", "spot_entrada"]:
        ops[c] = pd.to_numeric(ops[c], errors="coerce").fillna(0.0 if c != "spot_entrada" else np.nan)
    for c in ["quantidade", "strike", "preco_entrada", "preco_saida"]:
        pernas[c] = pd.to_numeric(pernas[c], errors="coerce")
    ops["notas"] = ops["notas"].fillna("").astype(str)
    return ops, pernas


def _salvar(ops: pd.DataFrame, pernas: pd.DataFrame) -> None:
    gravar(config.ARQ_OPERACOES, ops[COLS_OP].sort_values("op_id"))
    gravar(config.ARQ_PERNAS, pernas[COLS_PERNA].sort_values(["op_id", "perna_id"]))


def _prox(df: pd.DataFrame, col: str) -> int:
    return 1 if df.empty or df[col].isna().all() else int(pd.to_numeric(df[col]).max()) + 1


def custo_estimado(pernas: list[dict] | pd.DataFrame, preco_col: str = "preco") -> float:
    """Custos de uma rodada de ordens (emolumentos % + corretagem por ordem)."""
    df = pd.DataFrame(pernas)
    if df.empty:
        return 0.0
    fin = (df["quantidade"].astype(float) * df[preco_col].astype(float)).abs().sum()
    return float(config.CUSTO_PCT / 100 * fin + config.CORRETAGEM_ORDEM * len(df))


def criar(
    modo: str,
    ativo: str,
    estrategia: str,
    data_entrada: date,
    pernas: list[dict],
    custos: float | None = None,
    spot_entrada: float | None = None,
    notas: str = "",
    status: str = "ABERTA",
) -> int:
    if modo not in MODOS:
        raise ValueError("modo deve ser REAL ou ESTUDO")
    if not pernas:
        raise ValueError("a operação precisa de pelo menos uma perna")
    ops, todas = carregar()
    op_id = _prox(ops, "op_id")
    agora = pd.Timestamp.now().floor("s")
    ops = pd.concat(
        [
            ops,
            pd.DataFrame(
                [
                    {
                        "op_id": op_id, "modo": modo, "status": status, "ativo": ativo,
                        "estrategia": estrategia, "data_entrada": pd.Timestamp(data_entrada),
                        "spot_entrada": spot_entrada,
                        "custos": custo_estimado(pernas) if custos is None else float(custos),
                        "proventos": 0.0, "notas": notas, "data_encerramento": pd.NaT,
                        "criado_em": agora, "atualizado_em": agora,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    todas = pd.concat([todas, _linhas_pernas(pernas, op_id, _prox(todas, "perna_id"), data_entrada)], ignore_index=True)
    _salvar(ops, todas)
    return op_id


def _linhas_pernas(pernas: list[dict], op_id: int, primeiro_id: int, data_entrada) -> pd.DataFrame:
    linhas = []
    for i, p in enumerate(pernas):
        tipo = str(p["tipo"]).upper()
        linhas.append(
            {
                "perna_id": primeiro_id + i,
                "op_id": op_id,
                "tipo": tipo,
                "simbolo": str(p["simbolo"]).upper().strip(),
                "lado": str(p["lado"]).upper(),
                "quantidade": int(p["quantidade"]),
                "strike": float(p["strike"]) if tipo != "ACAO" and pd.notna(p.get("strike")) else np.nan,
                "vencimento": pd.Timestamp(p["vencimento"]) if tipo != "ACAO" and pd.notna(p.get("vencimento")) else pd.NaT,
                "preco_entrada": float(p["preco"]),
                "data_entrada": pd.Timestamp(p.get("data_entrada") or data_entrada),
                "preco_saida": float(p["preco_saida"]) if pd.notna(p.get("preco_saida", np.nan)) else np.nan,
                "data_saida": pd.Timestamp(p["data_saida"]) if pd.notna(p.get("data_saida", pd.NaT)) else pd.NaT,
            }
        )
    return pd.DataFrame(linhas, columns=COLS_PERNA)


def editar(op_id: int, **campos) -> None:
    ops, pernas = carregar()
    m = ops["op_id"] == op_id
    for k, v in campos.items():
        if k in ("data_entrada", "data_encerramento"):
            v = pd.Timestamp(v) if v not in (None, "") else pd.NaT
        ops.loc[m, k] = v
    ops.loc[m, "atualizado_em"] = pd.Timestamp.now().floor("s")
    _salvar(ops, pernas)


def substituir_pernas(op_id: int, editadas: pd.DataFrame) -> None:
    """Grava a tabela de pernas editada pelo usuário (mantém ids existentes)."""
    ops, pernas = carregar()
    e = editadas.copy()
    e = e[e["simbolo"].astype(str).str.strip() != ""].dropna(subset=["tipo", "lado", "quantidade", "preco_entrada"])
    if e.empty:
        raise ValueError("a operação precisa de pelo menos uma perna")
    prox = _prox(pernas, "perna_id")
    novos = e["perna_id"].isna()
    e.loc[novos, "perna_id"] = range(prox, prox + int(novos.sum()))
    e["op_id"] = op_id
    data_op = ops.loc[ops["op_id"] == op_id, "data_entrada"].iloc[0]
    e["data_entrada"] = pd.to_datetime(e.get("data_entrada"), errors="coerce").fillna(data_op)
    for c in ["vencimento", "data_saida"]:
        e[c] = pd.to_datetime(e[c], errors="coerce")
    e["simbolo"] = e["simbolo"].astype(str).str.upper().str.strip()
    pernas = pd.concat([pernas[pernas["op_id"] != op_id], e.reindex(columns=COLS_PERNA)], ignore_index=True)
    _atualizar_status(ops, pernas, op_id)
    _salvar(ops, pernas)


def adicionar_pernas(op_id: int, novas: list[dict], data_entrada: date, custos: float | None = None) -> None:
    ops, pernas = carregar()
    pernas = pd.concat([pernas, _linhas_pernas(novas, op_id, _prox(pernas, "perna_id"), data_entrada)], ignore_index=True)
    m = ops["op_id"] == op_id
    ops.loc[m, "custos"] += custo_estimado(novas) if custos is None else float(custos)
    _atualizar_status(ops, pernas, op_id)
    _salvar(ops, pernas)


def encerrar_pernas(op_id: int, saidas: dict[int, float], data_saida: date, custos: float | None = None) -> None:
    """Registra a saída das pernas indicadas ({perna_id: preço}); encerra a operação se nada ficar aberto."""
    ops, pernas = carregar()
    fluxo = []
    for pid, preco in saidas.items():
        m = pernas["perna_id"] == pid
        pernas.loc[m, "preco_saida"] = float(preco)
        pernas.loc[m, "data_saida"] = pd.Timestamp(data_saida)
        fluxo.append({"quantidade": float(pernas.loc[m, "quantidade"].iloc[0]), "preco": float(preco)})
    mo = ops["op_id"] == op_id
    ops.loc[mo, "custos"] += custo_estimado(fluxo) if custos is None else float(custos)
    _atualizar_status(ops, pernas, op_id, data_saida)
    _salvar(ops, pernas)


def _atualizar_status(ops: pd.DataFrame, pernas: pd.DataFrame, op_id: int, data=None) -> None:
    p = pernas[pernas["op_id"] == op_id]
    m = ops["op_id"] == op_id
    if len(p) and p["data_saida"].notna().all():
        ops.loc[m, "status"] = "ENCERRADA"
        ops.loc[m, "data_encerramento"] = pd.Timestamp(data) if data is not None else p["data_saida"].max()
    else:
        ops.loc[m, "status"] = "ABERTA"
        ops.loc[m, "data_encerramento"] = pd.NaT
    ops.loc[m, "atualizado_em"] = pd.Timestamp.now().floor("s")


def excluir(op_id: int) -> None:
    ops, pernas = carregar()
    _salvar(ops[ops["op_id"] != op_id], pernas[pernas["op_id"] != op_id])


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def _sinal(lado) -> np.ndarray:
    return np.where(pd.Series(lado).astype(str).str.upper().to_numpy() == "COMPRA", 1.0, -1.0)


def avaliar_pernas(pernas: pd.DataFrame, cadeia: pd.DataFrame, precos: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Preço atual, strike ajustado, gregas e resultado de cada perna (abertas e encerradas)."""
    if pernas.empty:
        return pernas.assign(preco_atual=[], fonte=[])
    p = pernas.copy().reset_index(drop=True)
    cad = cadeia.set_index("simbolo") if cadeia is not None and not cadeia.empty else pd.DataFrame()
    ult_preco = precos.sort_values("data").groupby("ativo").last() if precos is not None and not precos.empty else pd.DataFrame()
    spot_cad = cadeia.groupby("ativo")["spot"].last() if cadeia is not None and not cadeia.empty else pd.Series(dtype=float)
    sessao = pd.Timestamp(cadeia["sessao"].max()) if cadeia is not None and not cadeia.empty else pd.Timestamp(calendario.ultimo_pregao())
    r = float(cadeia["taxa_r"].iloc[0]) if cadeia is not None and not cadeia.empty else np.log1p(config.TAXA_JUROS_PADRAO)

    def spot_de(ativo):
        if ativo in spot_cad.index:
            return float(spot_cad[ativo])
        if ativo in ult_preco.index:
            return float(ult_preco.loc[ativo, "fechamento"])
        return np.nan

    linhas = []
    for row in p.itertuples(index=False):
        info = {"preco_atual": np.nan, "fonte": "", "strike_atual": row.strike, "iv": np.nan, "delta": np.nan,
                "gama": np.nan, "theta": np.nan, "vega": np.nan, "du": 0, "liquidez": np.nan, "ativo_obj": None,
                "spot": np.nan, "ultimo_negocio": pd.NaT}
        if row.tipo == "ACAO":
            info.update(ativo_obj=row.simbolo, spot=spot_de(row.simbolo), delta=1.0, gama=0.0, theta=0.0, vega=0.0)
            info.update(preco_atual=info["spot"], fonte="ação")
        elif not cad.empty and row.simbolo in cad.index:
            c = cad.loc[row.simbolo]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[0]
            info.update(
                preco_atual=float(c["preco_ref"]), fonte=str(c["fonte_preco"]), strike_atual=float(c["strike"]),
                iv=float(c["iv_modelo"]), delta=float(c["delta"]), gama=float(c["gama"]), theta=float(c["theta"]),
                vega=float(c["vega"]), du=int(c["du"]),
                liquidez=int(c["liquidez"]), ativo_obj=c["ativo"], spot=float(c["spot"]), ultimo_negocio=c["data_ultimo"],
            )
        else:  # série fora da cadeia (vencida ou ativo não acompanhado)
            ativo = _ativo_de_simbolo(row.simbolo, cadeia)
            S = spot_de(ativo) if ativo else np.nan
            info.update(ativo_obj=ativo, spot=S)
            venc = row.vencimento
            if pd.notna(venc) and venc <= sessao:
                S_v = _fechamento_em(precos, ativo, venc)
                S_v = S if not np.isfinite(S_v) else S_v
                intr = max(S_v - row.strike, 0) if row.tipo == "CALL" else max(row.strike - S_v, 0)
                info.update(preco_atual=intr, fonte="vencida (intrínseco)", spot=S_v)
            else:
                h = hist[hist["simbolo"] == row.simbolo].sort_values("data") if hist is not None and not hist.empty else pd.DataFrame()
                if not h.empty:
                    info.update(preco_atual=float(h["fechamento"].iloc[-1]), fonte=f"histórico {h['data'].iloc[-1]:%d/%m}")
                elif np.isfinite(S) and pd.notna(venc):
                    du = int(calendario.dias_uteis(sessao, venc))
                    iv = 0.35
                    info.update(
                        preco_atual=float(preco_bs(S, row.strike, du / DU_ANO, r, iv, row.tipo == "CALL")),
                        fonte="modelo (sem cotação)", du=du, iv=iv,
                    )
        linhas.append(info)
    out = pd.concat([p, pd.DataFrame(linhas)], axis=1)
    sinal = _sinal(out["lado"])
    aberta = out["data_saida"].isna()
    q = out["quantidade"].astype(float)
    out["sinal"] = sinal
    out["aberta"] = aberta
    out["resultado"] = np.where(
        aberta, sinal * q * (out["preco_atual"] - out["preco_entrada"]), sinal * q * (out["preco_saida"] - out["preco_entrada"])
    )
    # zerar: posição comprada é vendida (entra caixa); vendida é recomprada (sai caixa)
    out["caixa_desfazer"] = np.where(aberta, sinal * q * out["preco_atual"], 0.0)
    out["custo_saida"] = np.where(
        aberta, config.CUSTO_PCT / 100 * (q * out["preco_atual"]).abs() + config.CORRETAGEM_ORDEM, 0.0
    )
    out["strike_ajustado"] = (out["tipo"] != "ACAO") & (np.abs(out["strike_atual"] - out["strike"]) > 0.005)
    for g in ("delta", "gama", "theta", "vega"):  # gregas da posição (lado × quantidade), só pernas abertas
        out[f"{g}_pos"] = np.where(aberta, sinal * q * out[g], 0.0)
    out["taxa_r"] = r
    return out


def _ativo_de_simbolo(simbolo: str, cadeia: pd.DataFrame) -> str | None:
    if cadeia is None or cadeia.empty:
        return None
    raiz = simbolo[:4]
    m = cadeia.loc[cadeia["simbolo"].str[:4] == raiz, "ativo"]
    return m.mode().iloc[0] if not m.empty else None


def _fechamento_em(precos: pd.DataFrame, ativo: str | None, d) -> float:
    if precos is None or precos.empty or not ativo:
        return np.nan
    g = precos[(precos["ativo"] == ativo) & (precos["data"] <= pd.Timestamp(d))]
    return float(g.sort_values("data")["fechamento"].iloc[-1]) if not g.empty else np.nan


def resultado_vencimento(av: pd.DataFrame, S: np.ndarray) -> np.ndarray:
    """Resultado das pernas abertas se o ativo terminar em cada S (sem custos), strikes ajustados."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    total = np.zeros_like(S)
    for row in av[av["aberta"]].itertuples(index=False):
        if row.tipo == "ACAO":
            v = S
        elif row.tipo == "CALL":
            v = np.maximum(S - row.strike_atual, 0)
        else:
            v = np.maximum(row.strike_atual - S, 0)
        total += row.sinal * row.quantidade * (v - row.preco_entrada)
    return total


def resultado_hoje_curva(av: pd.DataFrame, S: np.ndarray) -> np.ndarray:
    """Resultado das pernas abertas hoje (Black-Scholes, IV atual de cada série) para cada S."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    total = np.zeros_like(S)
    for row in av[av["aberta"]].itertuples(index=False):
        if row.tipo == "ACAO":
            v = S
        else:
            iv = row.iv if np.isfinite(row.iv) else 0.35
            v = preco_bs(S, row.strike_atual, max(row.du, 0) / DU_ANO, row.taxa_r, iv, row.tipo == "CALL")
        total += row.sinal * row.quantidade * (v - row.preco_entrada)
    return total


def resumo(op: pd.Series, av: pd.DataFrame, S_venc: float | None = None) -> dict:
    """Números-chave de uma operação."""
    abertas, fechadas = av[av["aberta"]], av[~av["aberta"]]
    realizado = float(fechadas["resultado"].sum())
    custos = float(op["custos"] or 0)
    proventos = float(op["proventos"] or 0)
    spot = float(av["spot"].dropna().iloc[0]) if av["spot"].notna().any() else np.nan
    S_venc = spot if S_venc is None else S_venc
    custo_saida = float(abertas["custo_saida"].sum())
    venc = abertas.loc[abertas["tipo"] != "ACAO", "vencimento"]
    res_venc = float(resultado_vencimento(av, [S_venc])[0]) if np.isfinite(S_venc) else np.nan

    entrada = (av["sinal"] * av["quantidade"] * av["preco_entrada"]).sum()
    grade = np.linspace(0.01, 3 * spot, 600) if np.isfinite(spot) else np.array([])
    perda_max = float(min(resultado_vencimento(av, grade).min(), 0)) if len(grade) and len(abertas) else 0.0
    base = max(abs(float(entrada)), -perda_max, 1e-9)
    return {
        "spot": spot,
        "realizado": realizado,
        "aberto": float(abertas["resultado"].sum()),
        "custos": custos,
        "proventos": proventos,
        "caixa_desfazer": float(abertas["caixa_desfazer"].sum()) - custo_saida,
        "resultado_hoje": realizado + float(abertas["resultado"].sum()) - custos - custo_saida + proventos,
        "S_venc": S_venc,
        "resultado_venc": realizado + res_venc - custos + proventos if np.isfinite(res_venc) else np.nan,
        "vencimento": venc.min() if not venc.empty else pd.NaT,
        "du": int(abertas.loc[abertas["tipo"] != "ACAO", "du"].min()) if (abertas["tipo"] != "ACAO").any() else 0,
        "delta": float(abertas["delta_pos"].sum()),
        "gama": float(abertas["gama_pos"].sum()),
        "theta": float(abertas["theta_pos"].sum()),
        "vega": float(abertas["vega_pos"].sum()),
        "base_capital": base,
        "teorico": bool(((abertas["tipo"] != "ACAO") & (abertas["fonte"] != "último")).any()),
    }


def exercicio(av: pd.DataFrame, S: float) -> list[str]:
    """O que acontece no vencimento se o ativo fechar em S (por perna aberta)."""
    msgs = []
    for row in av[av["aberta"] & (av["tipo"] != "ACAO")].itertuples(index=False):
        K = row.strike_atual
        itm = S > K if row.tipo == "CALL" else S < K
        q = int(row.quantidade)
        obj = row.ativo_obj or "ativo"
        if not itm:
            msgs.append(f"{row.simbolo} ({row.tipo} {num(K)}) vira pó — "
                        f"{'você perde o prêmio pago' if row.sinal > 0 else 'você fica com o prêmio'}.")
            continue
        if row.tipo == "CALL":
            acao = "compra" if row.sinal > 0 else "vende"
        else:
            acao = "vende" if row.sinal > 0 else "compra"
        quem = "Você exerce" if row.sinal > 0 else "Você é exercido"
        msgs.append(
            f"{row.simbolo} ({row.tipo} {num(K)}) termina dentro do dinheiro: {quem} e {acao} {num(q, 0)} {obj} "
            f"a {brl(K)} ({brl(q * K)})."
        )
    estoque = av[av["aberta"] & (av["tipo"] == "ACAO")]
    if not estoque.empty:
        liq = int((estoque["sinal"] * estoque["quantidade"]).sum())
        msgs.append(f"Posição em ações antes do exercício: {num(liq, 0)} {estoque['simbolo'].iloc[0]}.")
    return msgs


# ---------------------------------------------------------------------------
# Histórico e rolagem
# ---------------------------------------------------------------------------

def historico_resultado(op: pd.Series, av: pd.DataFrame, hist: pd.DataFrame, precos: pd.DataFrame) -> pd.DataFrame:
    """Resultado diário (marcação a mercado) desde a entrada.

    Dias sem negócio repetem o último preço observado (até 5 pregões). Dias sem cotação coletada de
    alguma perna ficam de fora — sem estimativa inventada.
    """
    ini = pd.Timestamp(op["data_entrada"]).normalize()
    fim_cands = [pd.Timestamp(date.today())]
    if pd.notna(op.get("data_encerramento")):
        fim_cands.append(pd.Timestamp(op["data_encerramento"]))
    fim = min(fim_cands)
    datas = pd.bdate_range(ini, fim)
    datas = datas[[calendario.eh_pregao(d.date()) for d in datas]]
    if len(datas) == 0:
        return pd.DataFrame()
    total = pd.Series(0.0, index=datas)
    spot = pd.Series(np.nan, index=datas)
    for row in av.itertuples(index=False):
        if row.tipo == "ACAO":
            g = precos[precos["ativo"] == row.simbolo].set_index("data")["fechamento"] if not precos.empty else pd.Series(dtype=float)
        else:
            g = hist[hist["simbolo"] == row.simbolo].set_index("data")["fechamento"] if not hist.empty else pd.Series(dtype=float)
        g = g[~g.index.duplicated()].sort_index()
        # opções: repete o último preço por no máx. 5 pregões; lacunas maiores ficam sem ponto
        serie = g.reindex(g.index.union(datas)).ffill(limit=None if row.tipo == "ACAO" else 5).reindex(datas)
        entrada = pd.Timestamp(row.data_entrada).normalize()
        serie = serie.where(serie.index > entrada, row.preco_entrada)  # até a entrada vale o preço pago
        if not row.aberta:
            serie = serie.where(serie.index < pd.Timestamp(row.data_saida), row.preco_saida)
        elif pd.notna(row.preco_atual):
            serie.iloc[-1] = row.preco_atual
        contrib = row.sinal * row.quantidade * (serie - row.preco_entrada)
        contrib = contrib.where(contrib.index >= entrada, 0.0)
        total += contrib
        if row.tipo == "ACAO" or (row.ativo_obj and spot.isna().all()):
            obj = row.simbolo if row.tipo == "ACAO" else row.ativo_obj
            sp = precos[precos["ativo"] == obj].set_index("data")["fechamento"] if not precos.empty else pd.Series(dtype=float)
            sp = sp[~sp.index.duplicated()].sort_index()
            spot = sp.reindex(sp.index.union(datas)).ffill().reindex(datas)
    out = pd.DataFrame({"data": datas, "resultado": total.to_numpy() - float(op["custos"] or 0), "spot": spot.to_numpy()})
    return out.dropna(subset=["resultado"]).reset_index(drop=True)


def sugestoes_rolagem(perna: pd.Series, cadeia: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Rolagem de uma perna vendida: recompra a atual e vende série mais longa (mesmo tipo)."""
    if cadeia is None or cadeia.empty or perna["tipo"] == "ACAO" or perna["sinal"] > 0:
        return pd.DataFrame()
    ativo, S = perna["ativo_obj"], float(perna["spot"])
    K0, recompra = float(perna["strike_atual"]), float(perna["preco_atual"])
    c = cadeia[
        (cadeia["ativo"] == ativo) & (cadeia["tipo"] == perna["tipo"]) & (cadeia["vencimento"] > perna["vencimento"])
        & (cadeia["liquidez"] >= 2) & (cadeia["teorico"] >= config.PREMIO_MIN)
    ].copy()
    if perna["tipo"] == "CALL":
        c = c[(c["strike"] >= min(K0, S) * 0.98) & (c["strike"] <= 1.35 * S)]
        ganho_strike = c["strike"] - K0
    else:
        c = c[(c["strike"] <= max(K0, S) * 1.02) & (c["strike"] >= 0.65 * S)]
        ganho_strike = K0 - c["strike"]
    if c.empty:
        return c
    c["credito"] = c["preco_ref"] - recompra
    c["credito_total"] = c["credito"] * float(perna["quantidade"])
    meses = np.maximum((c["du"] - int(perna["du"])) / 21.0, 0.25)
    c["ganho_se_exercido"] = c["credito"] + ganho_strike
    c["rend_mes_exercido"] = c["ganho_se_exercido"] / S / meses
    c["rend_mes_credito"] = c["credito"] / S / meses
    c["dist_strike"] = c["strike"] / S - 1
    c = c[c["credito"] > -0.05 * S]
    return c.sort_values(["rend_mes_exercido"], ascending=False).head(top)[
        ["simbolo", "vencimento", "du", "strike", "preco_ref", "fonte_preco", "credito", "credito_total",
         "ganho_se_exercido", "rend_mes_exercido", "rend_mes_credito", "dist_strike", "delta", "liquidez", "oi_total"]
    ]
