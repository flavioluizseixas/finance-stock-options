"""Coleta de dados externos: opcoes.net.br (cadeias, contratos em aberto, histórico), Yahoo Finance e BCB.

Sem login, o opcoes.net.br entrega último preço, nº de negócios, volume, data do último negócio e
formador de mercado — IV/gregas vêm bloqueadas, por isso são recalculadas em `precificacao.py`.
"""

from __future__ import annotations

import math
import re
import time
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import requests

from . import config

BASE = "https://opcoes.net.br"
_SESSAO: requests.Session | None = None
_ultimo_req = 0.0
_pausa = config.PAUSA_REQ  # cadência adaptativa: cresce com HTTP 429, volta devagar ao padrão


class SiteBloqueado(RuntimeError):
    """O opcoes.net.br pediu uma espera longa (Retry-After); insistir só aumenta o bloqueio."""

    def __init__(self, segundos: float):
        self.segundos = segundos
        super().__init__(f"opcoes.net.br limitou as requisições deste IP; tente de novo em ~{math.ceil(segundos / 60)} min")


def _sessao() -> requests.Session:
    global _SESSAO
    if _SESSAO is None:
        _SESSAO = requests.Session()
        _SESSAO.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"})
    return _SESSAO


_logado: bool | None = None  # None = ainda não tentou


def login() -> bool:
    """Entra no opcoes.net.br com OPCOESNET_CPF/OPCOESNET_SENHA do `.env` (uma vez por processo).

    Devolve True se autenticou. Sem credenciais, ou se o login falhar, segue sem login.
    """
    global _logado
    if _logado is not None:
        return _logado
    _logado = False
    cpf = re.sub(r"\D", "", config.OPCOESNET_CPF)
    if len(cpf) != 11 or not config.OPCOESNET_SENHA:
        return False
    frag = _get(f"{BASE}/v1/login", params={"ctrlsOnly": "true", "returnUrl": "/opcoes/bovespa"}).text
    token = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', frag)
    if not token:
        return False
    global _ultimo_req
    espera = _pausa - (time.monotonic() - _ultimo_req)
    if espera > 0:
        time.sleep(espera)
    resp = _sessao().post(
        f"{BASE}/v1/login",
        params={"returnurl": "/opcoes/bovespa"},
        data={
            "CPF": f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}",
            "Password": config.OPCOESNET_SENHA,
            "RememberMe": "false",
            "__RequestVerificationToken": token.group(1),
        },
        allow_redirects=False,
        timeout=40,
    )
    _ultimo_req = time.monotonic()
    _logado = resp.status_code in (301, 302, 303) and any("Identity" in c.name and c.value for c in _sessao().cookies)
    return _logado


def liquidez_opcoes(data_ini: date, data_fim: date, tipo: str = "") -> tuple[pd.DataFrame, int]:
    """Estudo de liquidez do site (exige login): totais de negócios, volume e quantidade por ativo-objeto.

    Devolve (tabela, nº de pregões com negociação no período). `tipo`: "" (todas), "C" ou "P".
    """
    if not login():
        raise RuntimeError("o estudo de liquidez exige login no opcoes.net.br (OPCOESNET_CPF/OPCOESNET_SENHA)")
    url = f"{BASE}/estudos/liquidez/opcoes"
    html = _get(url).text
    tok = re.search(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', html)
    if not tok:
        raise RuntimeError("página do estudo de liquidez sem token")
    global _ultimo_req
    espera = _pausa - (time.monotonic() - _ultimo_req)
    if espera > 0:
        time.sleep(espera)
    dados = {
        "dataInicial": data_ini.strftime("%d/%m/%Y"), "dataFinal": data_fim.strftime("%d/%m/%Y"), "tipoOpcao": tipo,
        "mesesParaVencimento": "", "draw": "1", "start": "0", "length": "2000",
        "order[0][column]": "1", "order[0][dir]": "desc", "__RequestVerificationToken": tok.group(1),
    }
    resp = _sessao().post(url, data=dados, timeout=90,
                          headers={"RequestVerificationToken": tok.group(1), "X-Requested-With": "XMLHttpRequest"})
    _ultimo_req = time.monotonic()
    if resp.status_code == 429:
        raise SiteBloqueado(float(resp.headers.get("Retry-After", 60)))
    resp.raise_for_status()
    j = resp.json()
    if j.get("errorPeriodo"):
        raise RuntimeError(j["errorPeriodo"])
    df = pd.DataFrame(j.get("records") or [], columns=["ativo", "negocios_total", "volume_total", "quantidade_total"])
    for c in ["negocios_total", "volume_total", "quantidade_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, int(j.get("diasComNegociacao") or 0)


def _get(url: str, params: dict | None = None, timeout: int = 40, tentativas: int = 3) -> requests.Response:
    """GET com cadência adaptativa para o opcoes.net.br.

    O site limita requisições por IP e o `Retry-After` cresce a cada insistência. Por isso: espera
    curtas são respeitadas; esperas longas (> ESPERA_MAX) interrompem a coleta com `SiteBloqueado`.
    """
    global _ultimo_req, _pausa
    erro: Exception | None = None
    limitar = url.startswith(BASE)
    for i in range(tentativas):
        espera = _pausa - (time.monotonic() - _ultimo_req) if limitar else 0
        if espera > 0:
            time.sleep(espera)
        try:
            resp = _sessao().get(url, params=params, timeout=timeout)
            if limitar:
                _ultimo_req = time.monotonic()
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", 30))
                _pausa = min(max(_pausa * 2, 10.0), 60.0)
                if retry > config.ESPERA_MAX:
                    raise SiteBloqueado(retry)
                erro = requests.HTTPError("HTTP 429 (limite de requisições do site)")
                time.sleep(retry + 1)
                continue
            if resp.status_code in (502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            if limitar:
                _pausa = max(config.PAUSA_REQ, _pausa * 0.85)
            return resp
        except (requests.RequestException, ValueError) as exc:
            erro = exc
            time.sleep(3.0 * (i + 1))
    raise RuntimeError(f"Falha ao acessar {url}: {erro}")


# ---------------------------------------------------------------------------
# opcoes.net.br
# ---------------------------------------------------------------------------

_LETRA_CALL = "ABCDEFGHIJKL"
_LETRA_PUT = "MNOPQRSTUVWX"
_RE_TICKER = re.compile(r"^([A-Z0-9]{4})([A-X])(\d+)(W\d)?(?:_(\d{4}))?$")


def _num(v) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    txt = re.sub(r"<[^>]+>", "", str(v)).strip()
    if not txt:
        return np.nan
    txt = txt.replace("%", "").replace("\xa0", "")
    if "," in txt:
        txt = txt.replace(".", "").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return np.nan


def _parse_cotacoes(payload: dict, ativo: str, venc_map: dict) -> pd.DataFrame:
    data = payload.get("data") or {}
    rows = data.get("cotacoesOpcoes") or []
    if not rows:
        return pd.DataFrame()
    idx = {c["name"]: c["index"] for c in data.get("columns", [])}

    def col(nome: str, padrao: int):
        return idx.get(nome, padrao)

    i_tk, i_fm, i_tipo, i_mod = col("ticker", 0), col("fm", 1), col("tipo", 2), col("mod.", 3)
    i_k, i_dist, i_ult = col("strike", 5), col("distancia%dostrike", 6), col("ultimo", 8)
    i_neg, i_vol, i_dt = col("numerodenegocios", 9), col("volumenegociado", 10), col("data/hora", 11)
    i_iv = col("vol.implicita", 12)  # só com login (sem login vem borrada)

    def at(r, i):
        return r[i] if i < len(r) else None

    regs = []
    for r in rows:
        bruto = str(at(r, i_tk) or "")
        m = _RE_TICKER.match(bruto)
        simbolo = bruto.split("_")[0]
        venc = None
        if m:
            letra, semana, ano = m.group(2), m.group(4) or "", m.group(5)
            mes = (_LETRA_CALL.find(letra) if letra in _LETRA_CALL else _LETRA_PUT.find(letra)) + 1
            venc = venc_map.get((int(ano) if ano else None, mes, semana))
            if venc is None and not ano:
                venc = next((v for (a, mm, w), v in venc_map.items() if mm == mes and w == semana), None)
        regs.append(
            {
                "ativo": ativo,
                "simbolo": simbolo,
                "tipo": "CALL" if str(at(r, i_tipo)).upper().startswith("C") else "PUT",
                "estilo": str(at(r, i_mod) or "")[:1],
                "vencimento": venc,
                "strike": _num(at(r, i_k)),
                "dist_strike": _num(at(r, i_dist)),
                "ultimo": _num(at(r, i_ult)),
                "negocios": _num(at(r, i_neg)),
                "volume_fin": _num(at(r, i_vol)),
                "data_ultimo": at(r, i_dt),
                "fm": bool(at(r, i_fm)),
                "iv_site": _num(at(r, i_iv)),
            }
        )
    df = pd.DataFrame(regs)
    df["vencimento"] = pd.to_datetime(df["vencimento"], errors="coerce")
    df["data_ultimo"] = pd.to_datetime(df["data_ultimo"], errors="coerce").dt.normalize()
    return df


def _lista(ativo: str, listar: bool, vencs: list[str] | None = None) -> dict:
    params = {"idAcao": ativo, "listarVencimentos": "true" if listar else "false", "cotacoes": "true"}
    if vencs:
        params["vencimentos"] = ",".join(vencs)
    payload = _get(f"{BASE}/listaopcoes/completa", params=params).json()
    if not payload.get("success"):
        raise RuntimeError(payload.get("errorMessage", "resposta inválida"))
    return payload


def _vencimentos_do_payload(payload: dict) -> list[dict]:
    out = []
    for v in (payload.get("data") or {}).get("vencimentos") or []:
        attrs = v.get("dataAttributes") or {}
        out.append({"vencimento": str(v["value"])[:10], "semana": attrs.get("w") or "", "padrao": bool(v.get("selected"))})
    return out


def _mapa_venc(vencs: list[dict]) -> dict:
    mapa = {}
    for v in vencs:
        d = pd.Timestamp(v["vencimento"])
        mapa[(d.year, d.month, v["semana"])] = d
    return mapa


def cadeia(ativo: str, max_dias: int | None = None, apenas_padrao: bool = False, cache: dict | None = None) -> pd.DataFrame:
    """Cadeia de opções de um ativo, todos os vencimentos até `max_dias`.

    `cache` ({ativo: lista de vencimentos}) permite buscar lista + cotações em 1 requisição; sem ele são 2.
    `apenas_padrao=True` devolve só o vencimento mensal principal (1 requisição; usado no ranking).
    """
    limite = pd.Timestamp.today().normalize() + pd.Timedelta(days=max_dias or config.VENC_MAX_DIAS)

    def no_prazo(vencs):
        return [v["vencimento"] for v in vencs if pd.Timestamp(v["vencimento"]) <= limite]

    conhecidos = no_prazo(cache.get(ativo, [])) if cache is not None and not apenas_padrao else []
    payload = _lista(ativo, True, conhecidos or None)
    vencs = _vencimentos_do_payload(payload)
    if not vencs:
        return pd.DataFrame()
    if cache is not None:
        cache[ativo] = vencs
    mapa = _mapa_venc(vencs)
    df = _parse_cotacoes(payload, ativo, mapa)
    if apenas_padrao:
        return df

    recebidos = set(df["vencimento"].dropna().dt.strftime("%Y-%m-%d")) if not df.empty else set()
    padrao = {v["vencimento"] for v in vencs if v["padrao"]}
    if conhecidos and (recebidos - padrao or len(set(conhecidos)) <= 1):
        faltam = [v for v in no_prazo(vencs) if v not in set(conhecidos)]  # séries novas desde o cache
    else:  # sem cache, ou o site ignorou o filtro e devolveu só o vencimento padrão
        faltam = [v for v in no_prazo(vencs) if v not in recebidos]
    if faltam:
        extra = _parse_cotacoes(_lista(ativo, False, faltam), ativo, mapa)
        df = pd.concat([df, extra], ignore_index=True)
    if df.empty:
        return df
    df = df[df["vencimento"].notna() & (df["vencimento"] <= limite)]
    return df.drop_duplicates("simbolo", keep="last").reset_index(drop=True)


def contratos_em_aberto(ativos: list[str] | None = None) -> pd.DataFrame:
    """Posições em aberto (coberto/travado/descoberto) de todas as séries do mercado — 1 requisição."""
    params = {
        "r0t": "OpenInterest",
        "r0p.descending": "true",
        "r0p.sort_expression": "uncovered",
        "r0p.limit": "200000",
    }
    if ativos:
        params["r0p.underlying_assets_ids"] = ",".join(ativos)
    payload = _get(f"{BASE}/api/v1", params=params, timeout=90).json()
    reqs = payload.get("requests") or []
    res = (reqs[0].get("results") if reqs else None) or {}
    campos, linhas = res.get("data_fields") or [], res.get("data_rows") or []
    if not campos or not linhas:
        return pd.DataFrame()
    df = pd.DataFrame(linhas, columns=campos).rename(
        columns={
            "underlying": "ativo",
            "ticker": "simbolo",
            "expiration_date": "vencimento",
            "covered": "oi_coberto",
            "blocked": "oi_travado",
            "uncovered": "oi_descoberto",
            "buyers": "titulares",
            "sellers": "lancadores",
        }
    )
    for c in ["strike", "oi_coberto", "oi_travado", "oi_descoberto", "titulares", "lancadores"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["oi_total"] = df["oi_coberto"] + df["oi_travado"] + df["oi_descoberto"]
    df["vencimento"] = pd.to_datetime(df["vencimento"], errors="coerce")
    df["data_oi"] = pd.Timestamp(res.get("open_interest_date")) if res.get("open_interest_date") else pd.NaT
    return df[
        ["ativo", "simbolo", "vencimento", "strike", "oi_coberto", "oi_travado", "oi_descoberto",
         "oi_total", "titulares", "lancadores", "data_oi"]
    ]


def historico_opcao(simbolo: str) -> pd.DataFrame:
    """Últimos pregões de uma opção (página pública mostra ~5 dias): preço, negócios, IV e ativo-objeto."""
    html = _get(f"{BASE}/{simbolo}").text
    titulo = re.search(r"<title>(.*?)</title>", html, re.S)
    if not titulo or simbolo.upper() not in titulo.group(1).upper():
        return pd.DataFrame()
    try:
        tabelas = pd.read_html(StringIO(html), decimal=",", thousands=".")
    except ValueError:
        return pd.DataFrame()
    for t in tabelas:
        if not isinstance(t.columns, pd.MultiIndex) or t.shape[1] < 17:
            continue
        datas = pd.to_datetime(t.iloc[:, 0], dayfirst=True, errors="coerce")
        if datas.notna().sum() == 0:
            continue
        out = pd.DataFrame(
            {
                "data": datas.dt.normalize(),
                "simbolo": simbolo.upper(),
                "fechamento": pd.to_numeric(t.iloc[:, 4], errors="coerce"),
                "negocios": pd.to_numeric(t.iloc[:, 6], errors="coerce"),
                "volume_fin": pd.to_numeric(t.iloc[:, 7], errors="coerce"),
                "iv": pd.to_numeric(t.iloc[:, 11], errors="coerce") / 100.0,
                "spot": pd.to_numeric(t.iloc[:, 16], errors="coerce"),
            }
        )
        return out[out["data"].notna()].reset_index(drop=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Yahoo Finance e BCB
# ---------------------------------------------------------------------------

def yahoo_ticker(ativo: str) -> str:
    return ativo if "." in ativo or ativo.startswith("^") else f"{ativo}.SA"


def precos_yahoo(ativos: list[str], inicio: str | date) -> pd.DataFrame:
    """Histórico diário (formato longo) de vários ativos em uma chamada."""
    import yfinance as yf

    if not ativos:
        return pd.DataFrame()
    mapa = {yahoo_ticker(a): a for a in ativos}
    bruto = yf.download(
        list(mapa), start=str(inicio), progress=False, auto_adjust=False, group_by="ticker", threads=True
    )
    if bruto is None or bruto.empty:
        return pd.DataFrame()
    partes = []
    for yt, ativo in mapa.items():
        if isinstance(bruto.columns, pd.MultiIndex):
            if yt not in bruto.columns.get_level_values(0):
                continue
            d = bruto[yt].copy()
        else:
            d = bruto.copy()
        d = d.rename(columns=str.lower).rename(columns={"adj close": "fech_ajustado", "close": "fechamento"})
        d = d.dropna(subset=["fechamento"])
        if d.empty:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d.reset_index(names="data")
        d["ativo"] = ativo
        partes.append(d[["ativo", "data", "open", "high", "low", "fechamento", "fech_ajustado", "volume"]])
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes, ignore_index=True).rename(
        columns={"open": "abertura", "high": "maxima", "low": "minima"}
    )
    return out


def selic_meta() -> float | None:
    """Selic meta (% a.a.) mais recente — série SGS 432 do Banco Central."""
    hoje = date.today()
    params = {
        "formato": "json",
        "dataInicial": (pd.Timestamp(hoje) - pd.Timedelta(days=40)).strftime("%d/%m/%Y"),
        "dataFinal": hoje.strftime("%d/%m/%Y"),
    }
    try:
        dados = _get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados", params=params, timeout=20).json()
        return float(str(dados[-1]["valor"]).replace(",", ".")) / 100.0 if dados else None
    except Exception:
        return None


def taxa_juros() -> float:
    """Taxa livre de risco anual (composta). `TAXA_JUROS=auto` usa a Selic meta."""
    if str(config.TAXA_JUROS).lower() in ("auto", "selic", "bcb"):
        s = selic_meta()
        return s if s and math.isfinite(s) else config.TAXA_JUROS_PADRAO
    try:
        return float(config.TAXA_JUROS)
    except ValueError:
        return config.TAXA_JUROS_PADRAO
