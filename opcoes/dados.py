"""Armazenamento local (parquet) e rotinas de atualização de mercado."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import calendario, config, fontes
from .precificacao import DU_ANO, ajustar_smile, gregas_bs, preco_bs, vol_historica, vol_implicita

Progresso = Callable[[float, str], None] | None


# ---------------------------------------------------------------------------
# I/O básico
# ---------------------------------------------------------------------------

def ler(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def gravar(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def upsert(path: Path, novo: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    if novo is None or novo.empty:
        return ler(path)
    atual = ler(path)
    if not atual.empty:
        k_novo = novo[chaves].astype(str).agg("|".join, axis=1)
        k_atual = atual[chaves].astype(str).agg("|".join, axis=1)
        atual = atual[~k_atual.isin(set(k_novo))]
        novo = pd.concat([atual, novo], ignore_index=True)
    novo = novo.sort_values(chaves).reset_index(drop=True)
    gravar(path, novo)
    return novo


def mtime(path: Path) -> float:
    return Path(path).stat().st_mtime if Path(path).exists() else 0.0


def meta() -> dict:
    if config.ARQ_META.exists():
        return json.loads(config.ARQ_META.read_text(encoding="utf-8"))
    return {}


def salvar_meta(**kw) -> None:
    m = meta()
    m.update(kw)
    config.garantir_pastas()
    config.ARQ_META.write_text(json.dumps(m, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _p(progresso: Progresso, frac: float, msg: str) -> None:
    if progresso:
        progresso(min(max(frac, 0.0), 1.0), msg)


# ---------------------------------------------------------------------------
# Universo de ativos
# ---------------------------------------------------------------------------

def ativos_acompanhados() -> list[str]:
    if config.ARQ_ACOMPANHADOS.exists():
        lst = json.loads(config.ARQ_ACOMPANHADOS.read_text(encoding="utf-8"))
        if lst:
            return lst
    uni = ler(config.ARQ_UNIVERSO)
    if not uni.empty:
        return top_liquidos(uni)
    return [a.strip() for a in config.ATIVOS_PADRAO if a.strip()]


def top_liquidos(uni: pd.DataFrame, n: int | None = None) -> list[str]:
    """Mais líquidos do ranking, sem opções de índice (strike em pontos: lote/capital não comparáveis)."""
    acoes = uni[uni["spot"].fillna(0) < 2000] if "spot" in uni.columns else uni
    return acoes.head(n or config.UNIVERSO_TOP_N)["ativo"].tolist()


def salvar_ativos_acompanhados(ativos: list[str]) -> None:
    config.garantir_pastas()
    config.ARQ_ACOMPANHADOS.write_text(json.dumps(sorted(set(ativos)), ensure_ascii=False), encoding="utf-8")


def spot_implicito(df: pd.DataFrame) -> pd.Series:
    """Preço do ativo-objeto implícito em strike/(1+distância%) — mesmo instante das cotações do site."""
    s = df["strike"] / (1.0 + df["dist_strike"])
    s = s.where(np.isfinite(s) & (s > 0))
    return s.groupby(df["ativo"]).median()


ARQ_VENCIMENTOS = config.DATA_DIR / "vencimentos.json"  # cache da lista de vencimentos por ativo


def _cache_vencimentos() -> dict:
    if ARQ_VENCIMENTOS.exists():
        return json.loads(ARQ_VENCIMENTOS.read_text(encoding="utf-8"))
    return {}


def _salvar_cache_vencimentos(cache: dict) -> None:
    config.garantir_pastas()
    ARQ_VENCIMENTOS.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _liquidez_ativo(ch: pd.DataFrame, venc_ref, sessao) -> dict:
    """Volume, negócios e séries líquidas do vencimento de referência (mensal principal) no pregão."""
    g = ch[ch["vencimento"] == pd.Timestamp(venc_ref)] if pd.notna(venc_ref) else ch
    fresco = g["data_ultimo"] == pd.Timestamp(sessao)
    return {
        "venc_referencia": pd.Timestamp(venc_ref) if pd.notna(venc_ref) else pd.NaT,
        "volume_fin": float(g.loc[fresco, "volume_fin"].sum()),
        "negocios": float(g.loc[fresco, "negocios"].sum()),
        "series_liquidas": int((fresco & (g["negocios"] >= 10)).sum()),
        "series_fm": int(g["fm"].sum()),
    }


def _classificar_universo(uni: pd.DataFrame) -> pd.DataFrame:
    medido = uni["volume_fin"].notna()
    sl = uni["series_liquidas"]
    uni["classe_liquidez"] = np.select(
        [
            ~medido,
            (uni["volume_fin"] >= 5e6) & ((sl >= 15) | sl.isna()),
            (uni["volume_fin"] >= 5e5) & ((sl >= 5) | sl.isna()),
        ],
        ["Não medida", "Alta", "Média"],
        default="Baixa",
    )
    ordem = uni.assign(_v=uni["volume_fin"].fillna(-1))
    return ordem.sort_values(["_v", "oi_financeiro"], ascending=False).drop(columns="_v").reset_index(drop=True)


FONTE_ESTUDO = "estudo"  # média diária de 10 pregões, todos os vencimentos (estudo de liquidez do site, com login)
FONTE_VARREDURA = "varredura"  # último pregão, vencimento mensal principal (sem login)


def _universo_por_estudo(uni: pd.DataFrame, progresso: Progresso) -> pd.DataFrame:
    _p(progresso, 0.3, "Estudo de liquidez do opcoes.net.br (10 pregões)…")
    fim = calendario.ultimo_pregao()
    ini = (pd.Timestamp(fim) - pd.Timedelta(days=14)).date()
    est, dias = fontes.liquidez_opcoes(ini, fim)
    dias = max(dias, 1)
    est = est.assign(volume_fin=est["volume_total"] / dias, negocios=est["negocios_total"] / dias)
    uni = uni.merge(est[["ativo", "volume_fin", "negocios"]], how="outer", on="ativo")
    anterior = ler(config.ARQ_UNIVERSO)
    if not anterior.empty:  # séries líquidas / formador vêm das cadeias coletadas
        cols = [c for c in ["ativo", "spot", "series_liquidas", "series_fm", "venc_referencia", "data_negocios"] if c in anterior.columns]
        uni = uni.merge(anterior[cols], how="left", on="ativo")
    uni["periodo_liquidez"] = f"{ini:%d/%m}–{fim:%d/%m} ({dias} pregões)"
    return uni


def atualizar_universo(progresso: Progresso = None) -> pd.DataFrame:
    """Ranking de liquidez dos ativos com opções.

    Com login no opcoes.net.br: contratos em aberto de todo o mercado + estudo de liquidez do site (média
    diária de 10 pregões, todos os vencimentos) — 4 requisições. Sem login: contratos em aberto + volume
    do último pregão medido nos `UNIVERSO_VARREDURA` maiores (1 requisição cada). Se o site bloquear,
    salva o que já foi medido.
    """
    _p(progresso, 0.0, "Contratos em aberto de todo o mercado…")
    logado = fontes.login()
    oi = fontes.contratos_em_aberto()
    if oi.empty:
        raise RuntimeError("opcoes.net.br não devolveu contratos em aberto")
    oi = oi.assign(oi_fin=oi["oi_total"] * oi["strike"])
    uni = (
        oi.groupby("ativo")
        .agg(oi_contratos=("oi_total", "sum"), oi_financeiro=("oi_fin", "sum"), series_em_aberto=("simbolo", "nunique"))
        .reset_index()
    )
    uni = uni[uni["oi_financeiro"] > 0].sort_values("oi_financeiro", ascending=False).reset_index(drop=True)
    if logado:
        try:
            uni = _universo_por_estudo(uni, progresso)
            for c in ["spot", "volume_fin", "negocios", "series_liquidas", "series_fm"]:
                uni[c] = pd.to_numeric(uni[c], errors="coerce") if c in uni.columns else np.nan
            for c in ["data_negocios", "venc_referencia"]:
                uni[c] = pd.to_datetime(uni[c], errors="coerce") if c in uni.columns else pd.NaT
            uni["avaliado_em"] = pd.Timestamp.now().floor("s")
            uni = _classificar_universo(uni)
            gravar(config.ARQ_UNIVERSO, uni)
            salvar_meta(aviso_universo=None, fonte_universo=FONTE_ESTUDO)
            _p(progresso, 1.0, "Universo atualizado (estudo de liquidez)")
            return uni
        except fontes.SiteBloqueado:
            raise
        except Exception as exc:  # estudo indisponível: segue com a varredura
            salvar_meta(aviso_universo=f"estudo de liquidez indisponível ({exc}); usando varredura")
    anterior = ler(config.ARQ_UNIVERSO)
    medidas = {}
    cache = _cache_vencimentos()
    alvo = uni["ativo"].head(config.UNIVERSO_VARREDURA).tolist()
    aviso = None
    for i, a in enumerate(alvo):
        _p(progresso, 0.05 + 0.95 * i / max(len(alvo), 1), f"Volume de opções de {a} ({i + 1}/{len(alvo)})")
        try:
            ch = fontes.cadeia(a, apenas_padrao=True, cache=cache)
        except fontes.SiteBloqueado as exc:
            aviso = str(exc)
            break
        except Exception:
            continue
        if ch.empty:
            continue
        spot = spot_implicito(ch)
        medidas[a] = {
            "spot": float(spot.iloc[0]) if len(spot) else np.nan,
            "data_negocios": ch["data_ultimo"].max(),
            **_liquidez_ativo(ch, ch["vencimento"].mode().iloc[0] if ch["vencimento"].notna().any() else pd.NaT,
                              ch["data_ultimo"].max()),
        }
    _salvar_cache_vencimentos(cache)
    med = pd.DataFrame.from_dict(medidas, orient="index").rename_axis("ativo").reset_index() if medidas else pd.DataFrame(columns=["ativo"])
    if not anterior.empty and aviso:  # bloqueado no meio: mantém medições anteriores dos não medidos agora
        prev = anterior[~anterior["ativo"].isin(med["ativo"])]
        cols = [c for c in ["ativo", "spot", "data_negocios", "venc_referencia", "volume_fin", "negocios", "series_liquidas", "series_fm"] if c in prev.columns]
        med = pd.concat([med, prev[cols]], ignore_index=True)
    uni = uni.merge(med, how="left", on="ativo")
    for c in ["spot", "volume_fin", "negocios", "series_liquidas", "series_fm"]:
        uni[c] = pd.to_numeric(uni[c], errors="coerce") if c in uni.columns else np.nan
    for c in ["data_negocios", "venc_referencia"]:
        uni[c] = pd.to_datetime(uni[c], errors="coerce") if c in uni.columns else pd.NaT
    if "data_negocios" in uni.columns and uni["data_negocios"].notna().any():
        sessao = uni["data_negocios"].max()
        velho = uni["data_negocios"] < sessao  # sem negócios no último pregão
        uni.loc[velho, ["volume_fin", "negocios", "series_liquidas"]] = 0
    uni["avaliado_em"] = pd.Timestamp.now().floor("s")
    uni["periodo_liquidez"] = "último pregão, vencimento principal"
    uni = _classificar_universo(uni)
    gravar(config.ARQ_UNIVERSO, uni)
    salvar_meta(aviso_universo=aviso, fonte_universo=FONTE_VARREDURA)
    _p(progresso, 1.0, "Universo atualizado" + (f" (parcial: {aviso})" if aviso else ""))
    return uni


def _atualizar_universo_com_cadeias(cadeia: pd.DataFrame, cache: dict) -> None:
    """Refresca no ranking a liquidez dos ativos acompanhados (sem requisições extras)."""
    uni = ler(config.ARQ_UNIVERSO)
    if uni.empty or cadeia.empty:
        return
    linhas = []
    for a, ch in cadeia.groupby("ativo"):
        padrao = next((v["vencimento"] for v in cache.get(a, []) if v.get("padrao")), None)
        if padrao is None or a not in set(uni["ativo"]):
            continue
        info = _liquidez_ativo(ch, padrao, ch["sessao"].iloc[0])
        info.update(ativo=a, spot=float(ch["spot"].iloc[0]), data_negocios=pd.Timestamp(ch["sessao"].iloc[0]))
        if meta().get("fonte_universo") == FONTE_ESTUDO:  # volume/negócios do estudo (média 10 pregões) prevalecem
            info.pop("volume_fin")
            info.pop("negocios")
        linhas.append(info)
    if not linhas:
        return
    upd = pd.DataFrame(linhas).set_index("ativo")
    uni = uni.set_index("ativo")
    for c in upd.columns:
        uni[c] = upd[c].combine_first(uni[c]) if c in uni.columns else upd[c]
    gravar(config.ARQ_UNIVERSO, _classificar_universo(uni.reset_index()))


# ---------------------------------------------------------------------------
# Preços do ativo-objeto (Yahoo)
# ---------------------------------------------------------------------------

def atualizar_precos(ativos: list[str]) -> pd.DataFrame:
    atual = ler(config.ARQ_PRECOS)
    ultimos = atual.groupby("ativo")["data"].max() if not atual.empty else pd.Series(dtype="datetime64[ns]")
    novos = [a for a in ativos if a not in ultimos.index]
    existentes = [a for a in ativos if a in ultimos.index]
    partes = []
    if novos:
        partes.append(fontes.precos_yahoo(novos, config.HIST_INICIO))
    if existentes:
        ini = (ultimos.loc[existentes].min() - pd.Timedelta(days=7)).date()
        partes.append(fontes.precos_yahoo(existentes, ini))
    novo = pd.concat([p for p in partes if p is not None and not p.empty], ignore_index=True) if partes else None
    return upsert(config.ARQ_PRECOS, novo, ["ativo", "data"])


def indicadores(precos: pd.DataFrame) -> pd.DataFrame:
    """Vol. histórica e tendência por ativo (último pregão)."""
    linhas = []
    for a, g in precos.sort_values("data").groupby("ativo"):
        c = g["fechamento"].astype(float).reset_index(drop=True)
        if len(c) < 30:
            continue
        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        ult = c.iloc[-1]
        if np.isfinite(sma200) and ult > sma50 > sma200:
            tend = "Alta"
        elif np.isfinite(sma200) and ult < sma50 < sma200:
            tend = "Baixa"
        else:
            tend = "Lateral"
        delta = c.diff()
        ganho = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        perda = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
        rsi = 100 - 100 / (1 + ganho / perda) if perda else 100.0
        linhas.append(
            {
                "ativo": a,
                "data_preco": g["data"].iloc[-1],
                "fechamento": ult,
                "var_3m": ult / c.iloc[-64] - 1 if len(c) > 64 else np.nan,
                "hv21": vol_historica(c, 21),
                "hv63": vol_historica(c, 63),
                "rsi14": rsi,
                "tendencia": tend,
            }
        )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Enriquecimento da cadeia: IV, smile, preço teórico, gregas, liquidez
# ---------------------------------------------------------------------------

def _nivel_liquidez(df: pd.DataFrame) -> np.ndarray:
    neg = df["negocios"].fillna(0)
    vol = df["volume_fin"].fillna(0)
    alta = df["fresco"] & (((neg >= config.LIQ_ALTA_NEGOCIOS) & (vol >= config.LIQ_ALTA_VOLUME)) | (df["fm"] & (neg >= 5)))
    media = (df["pregoes_sem_negocio"] <= config.LIQ_MEDIA_DIAS) & (neg >= config.LIQ_MEDIA_NEGOCIOS)
    baixa = df["data_ultimo"].notna()
    return np.select([alta, media, baixa], [3, 2, 1], default=0)


def enriquecer(bruto: pd.DataFrame, oi: pd.DataFrame, ind: pd.DataFrame, taxa_anual: float, sessao) -> pd.DataFrame:
    df = bruto.copy()
    sessao = pd.Timestamp(sessao).normalize()
    r = float(np.log1p(taxa_anual))

    spot = spot_implicito(df)
    if not ind.empty:
        fallback = ind.set_index("ativo")["fechamento"]
        spot = spot.combine_first(fallback)
    df["spot"] = df["ativo"].map(spot)
    df = df[df["spot"].notna() & df["strike"].notna() & df["vencimento"].notna()].copy()

    df["du"] = calendario.dias_uteis(sessao, df["vencimento"])
    df["T"] = df["du"] / DU_ANO
    df["forward"] = df["spot"] * np.exp(r * df["T"])
    df["fresco"] = df["data_ultimo"] == sessao
    sem = calendario.dias_uteis(df["data_ultimo"].fillna(sessao), sessao)
    df["pregoes_sem_negocio"] = np.where(df["data_ultimo"].notna(), sem, 9999)
    is_call = (df["tipo"] == "CALL").to_numpy()

    df["iv_mercado"] = np.where(
        df["fresco"] & (df["du"] > 0),
        vol_implicita(df["ultimo"], df["spot"], df["strike"], df["T"], r, is_call),
        np.nan,
    )

    hv = ind.set_index("ativo")["hv63"] if not ind.empty else pd.Series(dtype=float)
    df["iv_modelo"] = np.nan
    for ativo, g_at in df.groupby("ativo"):
        perto = g_at[(np.abs(np.log(g_at["strike"] / g_at["forward"])) < 0.05) & g_at["iv_mercado"].notna()]
        padrao = float(perto["iv_mercado"].median()) if not perto.empty else float(hv.get(ativo, np.nan))
        if not np.isfinite(padrao):
            padrao = 0.35
        for _, g in g_at.groupby("vencimento"):
            df.loc[g.index, "iv_modelo"] = ajustar_smile(g, padrao)

    df["teorico"] = preco_bs(df["spot"], df["strike"], df["T"], r, df["iv_modelo"], is_call)
    usa_ultimo = df["fresco"] & (df["ultimo"] > 0)
    df["preco_ref"] = np.where(usa_ultimo, df["ultimo"], df["teorico"])
    df["fonte_preco"] = np.where(usa_ultimo, "último", "teórico")
    g = gregas_bs(df["spot"], df["strike"], df["T"], r, df["iv_modelo"], is_call)
    for k, v in g.items():
        df[k] = v
    df["liquidez"] = _nivel_liquidez(df)

    if oi is not None and not oi.empty:
        df = df.merge(
            oi[["simbolo", "oi_coberto", "oi_travado", "oi_descoberto", "oi_total", "titulares", "lancadores"]],
            how="left",
            on="simbolo",
        )
    else:
        for c in ["oi_coberto", "oi_travado", "oi_descoberto", "oi_total", "titulares", "lancadores"]:
            df[c] = np.nan
    df["sessao"] = sessao
    df["taxa_r"] = r
    df["coletado_em"] = pd.Timestamp.now().floor("s")
    return df.sort_values(["ativo", "vencimento", "tipo", "strike"]).reset_index(drop=True)


def iv_atm(cadeia: pd.DataFrame, du_alvo: int = 21) -> pd.Series:
    """IV ATM interpolada (variância × tempo) para `du_alvo` pregões, por ativo."""
    out = {}
    for ativo, g in cadeia[cadeia["du"] > 0].groupby("ativo"):
        pts = []
        for _, gv in g.groupby("vencimento"):
            i = (np.log(gv["strike"] / gv["forward"])).abs().idxmin()
            pts.append((gv.at[i, "T"], gv.at[i, "iv_modelo"]))
        pts = sorted(p for p in pts if np.isfinite(p[1]))
        if not pts:
            continue
        T = np.array([p[0] for p in pts])
        var = np.array([p[1] ** 2 * p[0] for p in pts])
        t_alvo = du_alvo / DU_ANO
        v = np.interp(t_alvo, T, var) if len(T) > 1 else var[0] / T[0] * t_alvo
        out[ativo] = float(np.sqrt(max(v, 1e-8) / t_alvo))
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------------------
# Atualização diária
# ---------------------------------------------------------------------------

def _simbolos_abertos() -> tuple[set[str], set[str]]:
    """(ativos-objeto, códigos de opção) das pernas em aberto."""
    pernas = ler(config.ARQ_PERNAS)
    ops = ler(config.ARQ_OPERACOES)
    if pernas.empty or ops.empty:
        return set(), set()
    abertas = ops.loc[ops["status"] == "ABERTA", "op_id"]
    p = pernas[pernas["op_id"].isin(abertas) & pernas["data_saida"].isna()]
    ativos = set(ops.loc[ops["op_id"].isin(abertas), "ativo"])
    return ativos, set(p.loc[p["tipo"] != "ACAO", "simbolo"])


def atualizar_mercado(
    ativos: list[str] | None = None, progresso: Progresso = None, historico_pernas: bool = True,
    somente_operacoes: bool = False,
) -> dict:
    """Atualiza preços (Yahoo), cadeias (opcoes.net.br), contratos em aberto e o histórico das pernas.

    `somente_operacoes=True`: só os ativos com operações abertas (rápido).
    """
    config.garantir_pastas()
    ativos_ops, simbolos_ops = _simbolos_abertos()
    ativos_pedidos = list(ativos or [])
    base = set() if somente_operacoes else set(ativos or ativos_acompanhados())
    ativos = sorted(base | ativos_ops)
    if not ativos:
        return {"ativos": [], "linhas": 0, "erros": {"operacoes": "nenhuma operação aberta"}}

    _p(progresso, 0.02, "Taxa de juros (BCB) e login no opcoes.net.br…")
    taxa = fontes.taxa_juros()
    erros: dict[str, str] = {}
    try:
        logado = fontes.login()
    except Exception as exc:  # segue sem login
        logado = False
        erros["login"] = str(exc)
    if config.OPCOESNET_CPF and not logado and "login" not in erros:
        erros["login"] = "login no opcoes.net.br recusado — confira OPCOESNET_CPF/OPCOESNET_SENHA no .env"

    _p(progresso, 0.05, "Preços do ativo-objeto (Yahoo Finance)…")
    try:
        precos = atualizar_precos(ativos)
    except Exception as exc:
        erros["yahoo"] = str(exc)
        precos = ler(config.ARQ_PRECOS)
    ind = indicadores(precos[precos["ativo"].isin(ativos)]) if not precos.empty else pd.DataFrame()

    _p(progresso, 0.12, "Contratos em aberto…")
    try:
        oi = fontes.contratos_em_aberto()
    except Exception as exc:
        erros["contratos_em_aberto"] = str(exc)
        oi = pd.DataFrame()

    brutos = []
    cache = _cache_vencimentos()
    # ativos com operações abertas primeiro: se o site bloquear no meio, o essencial já veio
    ordem = sorted(ativos, key=lambda a: (a not in ativos_ops, ativos.index(a)))
    for i, a in enumerate(ordem):
        _p(progresso, 0.15 + 0.7 * i / max(len(ordem), 1), f"Cadeia de opções de {a} ({i + 1}/{len(ordem)})")
        try:
            ch = fontes.cadeia(a, cache=cache)
            if not ch.empty:
                brutos.append(ch)
        except fontes.SiteBloqueado as exc:
            faltaram = ordem[i:]
            erros["opcoes.net.br"] = f"{exc}. Sem atualizar: {', '.join(faltaram)}"
            break
        except Exception as exc:
            erros[a] = str(exc)
    _salvar_cache_vencimentos(cache)
    if not brutos:
        salvar_meta(atualizado_em=datetime.now().isoformat(timespec="seconds"), erros=erros)
        return {"ativos": ativos, "linhas": 0, "erros": erros}

    bruto = pd.concat(brutos, ignore_index=True)
    sessao = bruto["data_ultimo"].max()
    _p(progresso, 0.87, "Calculando IV, smile, gregas e liquidez…")
    nova = enriquecer(bruto, oi, ind, taxa, sessao)

    atual = ler(config.ARQ_CADEIA)
    parcial = bool(ativos_pedidos) or somente_operacoes or "opcoes.net.br" in erros
    if parcial and not atual.empty:  # atualização de um subconjunto: preserva os demais ativos
        atual = atual[~atual["ativo"].isin(nova["ativo"].unique())]
        nova = pd.concat([atual, nova], ignore_index=True)
    gravar(config.ARQ_CADEIA, nova)

    # Arquivo do pregão: só séries com alguma liquidez recente ou em operações (economiza disco).
    hoje = nova[nova["sessao"] == pd.Timestamp(sessao)]
    arquivo = hoje[(hoje["pregoes_sem_negocio"] <= 60) | (hoje["oi_total"].fillna(0) > 0) | hoje["simbolo"].isin(simbolos_ops)]
    upsert(config.CADEIAS_DIR / f"{pd.Timestamp(sessao):%Y-%m-%d}.parquet", arquivo, ["simbolo"])
    _limpar_arquivo_antigo()

    _p(progresso, 0.92, "Volatilidade histórica x implícita…")
    ivs = iv_atm(hoje)
    if not ind.empty:
        vh = ind[["ativo", "fechamento", "hv21", "hv63"]].copy()
        vh["iv_atm_21"] = vh["ativo"].map(ivs)
        vh["sessao"] = pd.Timestamp(sessao)
        upsert(config.ARQ_VOL_HIST, vh, ["ativo", "sessao"])

    _atualizar_universo_com_cadeias(hoje, cache)

    if simbolos_ops:
        _p(progresso, 0.94, "Histórico das opções em operações…")
        bloqueado = "opcoes.net.br" in erros
        atualizar_historico_pernas(simbolos_ops, nova, buscar_site=historico_pernas and not bloqueado)

    salvar_meta(
        atualizado_em=datetime.now().isoformat(timespec="seconds"),
        sessao=str(pd.Timestamp(sessao).date()),
        taxa_juros=taxa,
        login=logado,
        ativos=ativos,
        erros=erros,
    )
    _p(progresso, 1.0, "Concluído")
    return {"ativos": ativos, "linhas": len(hoje), "sessao": pd.Timestamp(sessao).date(), "erros": erros}


def _limpar_arquivo_antigo() -> None:
    limite = pd.Timestamp.today() - pd.Timedelta(days=config.MANTER_CADEIAS_DIAS)
    for f in config.CADEIAS_DIR.glob("*.parquet"):
        try:
            if pd.Timestamp(f.stem) < limite:
                f.unlink()
        except ValueError:
            continue


def atualizar_historico_pernas(simbolos: set[str], cadeia: pd.DataFrame | None = None, buscar_site: bool = True) -> pd.DataFrame:
    """Grava o fechamento diário das opções usadas em operações (cadeia + página da opção, ~5 pregões)."""
    partes = []
    if cadeia is not None and not cadeia.empty:
        c = cadeia[cadeia["simbolo"].isin(simbolos) & cadeia["data_ultimo"].notna() & (cadeia["ultimo"] > 0)]
        if not c.empty:
            partes.append(
                pd.DataFrame(
                    {
                        "simbolo": c["simbolo"],
                        "data": c["data_ultimo"],
                        "fechamento": c["ultimo"],
                        "negocios": c["negocios"],
                        "volume_fin": c["volume_fin"],
                        "iv": c["iv_mercado"],
                        "spot": np.where(c["fresco"], c["spot"], np.nan),
                    }
                )
            )
    if buscar_site:
        for s in sorted(simbolos):
            try:
                h = fontes.historico_opcao(s)
            except fontes.SiteBloqueado:
                break
            except Exception:
                continue
            if not h.empty:
                partes.append(h[["simbolo", "data", "fechamento", "negocios", "volume_fin", "iv", "spot"]])
    partes = [p for p in partes if not p.empty]
    if not partes:
        return ler(config.ARQ_HIST_PERNAS)
    novo = pd.concat(partes, ignore_index=True)
    novo = novo[novo["fechamento"].notna()].drop_duplicates(["simbolo", "data"], keep="last")
    return upsert(config.ARQ_HIST_PERNAS, novo, ["simbolo", "data"])


def preencher_historico_do_arquivo(simbolos: set[str], desde: date) -> None:
    """Recupera do arquivo diário de cadeias o fechamento de opções a partir de `desde`."""
    partes = []
    for f in sorted(config.CADEIAS_DIR.glob("*.parquet")):
        try:
            if pd.Timestamp(f.stem) < pd.Timestamp(desde):
                continue
        except ValueError:
            continue
        d = pd.read_parquet(f, columns=["simbolo", "data_ultimo", "ultimo", "negocios", "volume_fin", "iv_mercado", "spot", "fresco"])
        d = d[d["simbolo"].isin(simbolos) & d["data_ultimo"].notna()]
        if not d.empty:
            partes.append(d)
    if partes:
        atualizar_historico_pernas(simbolos, pd.concat(partes, ignore_index=True), buscar_site=False)
