"""Black-Scholes vetorizado, volatilidade implícita, smile por vencimento e distribuição de preços.

Convenções B3: tempo em dias úteis / 252; taxa contínua r = ln(1 + taxa anual composta); sem dividendos
(os strikes da B3 são ajustados por proventos).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

DU_ANO = 252.0
_PDF = 1.0 / np.sqrt(2.0 * np.pi)


def _d1d2(S, K, T, r, sig):
    sT = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / sT
    return d1, d1 - sT


def preco_bs(S, K, T, r, sig, is_call):
    """Preço Black-Scholes (arrays). Com T<=0 devolve o valor intrínseco."""
    S, K, T, sig = (np.asarray(x, dtype=float) for x in (S, K, T, sig))
    is_call = np.asarray(is_call, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        Tp = np.where(T > 0, T, 1e-9)
        sp = np.where(sig > 0, sig, 1e-9)
        d1, d2 = _d1d2(S, K, Tp, r, sp)
        df = np.exp(-r * Tp)
        call = S * ndtr(d1) - K * df * ndtr(d2)
        put = K * df * ndtr(-d2) - S * ndtr(-d1)
    preco = np.where(is_call, call, put)
    intr = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    return np.where(T > 0, preco, intr)


def gregas_bs(S, K, T, r, sig, is_call) -> dict[str, np.ndarray]:
    """Delta, gama, theta (R$ por dia útil) e vega (R$ por 1 ponto de vol)."""
    S, K, T, sig = (np.asarray(x, dtype=float) for x in (S, K, T, sig))
    is_call = np.asarray(is_call, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        Tp = np.where(T > 0, T, 1e-9)
        d1, d2 = _d1d2(S, K, Tp, r, sig)
        pdf = _PDF * np.exp(-0.5 * d1 * d1)
        df = np.exp(-r * Tp)
        delta = np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)
        gama = pdf / (S * sig * np.sqrt(Tp))
        vega = S * pdf * np.sqrt(Tp) / 100.0
        th_call = -(S * pdf * sig) / (2 * np.sqrt(Tp)) - r * K * df * ndtr(d2)
        th_put = -(S * pdf * sig) / (2 * np.sqrt(Tp)) + r * K * df * ndtr(-d2)
        theta = np.where(is_call, th_call, th_put) / DU_ANO
    return {"delta": delta, "gama": gama, "theta": theta, "vega": vega}


def vol_implicita(preco, S, K, T, r, is_call, lo=0.01, hi=5.0, iters=64) -> np.ndarray:
    """IV por bisseção vetorizada (robusta). NaN quando o preço viola os limites de arbitragem."""
    preco, S, K, T = (np.asarray(x, dtype=float) for x in (preco, S, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    ok = np.isfinite(preco) & (preco > 0) & (T > 0) & (S > 0) & (K > 0)
    df = np.exp(-r * np.where(T > 0, T, 0))
    piso = np.where(is_call, np.maximum(S - K * df, 0), np.maximum(K * df - S, 0))
    teto = np.where(is_call, S, K * df)
    ok &= (preco > piso + 1e-4) & (preco < teto)
    a = np.full(preco.shape, lo)
    b = np.full(preco.shape, hi)
    ok &= preco_bs(S, K, T, r, b, is_call) >= preco
    for _ in range(iters):
        m = 0.5 * (a + b)
        acima = preco_bs(S, K, T, r, m, is_call) > preco
        b = np.where(acima, m, b)
        a = np.where(acima, a, m)
    iv = 0.5 * (a + b)
    return np.where(ok, iv, np.nan)


def ajustar_smile(df: pd.DataFrame, iv_padrao: float | None) -> np.ndarray:
    """IV de modelo para todas as séries de UM vencimento.

    Ajusta IV = a + b·x + c·x² (x = ln(K/F)/√T) por mínimos quadrados ponderados pelo nº de negócios,
    usando apenas séries OTM negociadas no último pregão. Extrapolação plana fora da faixa observada.
    """
    n = len(df)
    if n == 0:
        return np.array([])
    T = float(df["T"].iloc[0])
    F = float(df["forward"].iloc[0])
    x_all = np.log(df["strike"].to_numpy(float) / F) / np.sqrt(max(T, 1e-6))
    iv = df["iv_mercado"].to_numpy(float)
    w = np.sqrt(df["negocios"].fillna(0).to_numpy(float).clip(min=1))
    otm = np.where(df["tipo"].to_numpy() == "CALL", df["strike"] >= F * 0.98, df["strike"] <= F * 1.02)
    usar = np.isfinite(iv) & otm & (iv > 0.02) & (iv < 3)
    if usar.sum() < 2:  # poucos pontos OTM: aceita ITM também
        usar = np.isfinite(iv) & (iv > 0.02) & (iv < 3)
    if usar.sum() == 0:
        return np.full(n, iv_padrao if iv_padrao else np.nan)
    x, y, ww = x_all[usar], iv[usar], w[usar]
    if usar.sum() >= 5 and np.ptp(x) > 0.15:
        grau = 2
    elif usar.sum() >= 3 and np.ptp(x) > 0.05:
        grau = 1
    else:
        nivel = float(np.average(y, weights=ww))
        return np.full(n, nivel)
    coef = np.polyfit(x, y, grau, w=ww)
    xc = np.clip(x_all, x.min(), x.max())
    out = np.polyval(coef, xc)
    return np.clip(out, max(0.03, y.min() * 0.7), min(3.0, y.max() * 1.3))


# ---------------------------------------------------------------------------
# Distribuição do preço no vencimento: simulação histórica filtrada (FHS)
# ---------------------------------------------------------------------------

def vol_referencia(retornos: np.ndarray, lam: float = 0.94, corte: float = 4.0) -> tuple[float, np.ndarray]:
    """Vol diária de referência + choques padronizados do histórico.

    A variância EWMA é robusta: cada retorno entra limitado a ±`corte`·σ, para que um salto isolado
    não domine a vol por semanas. Os choques padronizados (`z`) mantêm o salto inteiro — as caudas
    da distribuição continuam lá. Referência = média (em variância) da EWMA robusta e da vol de 1 ano.
    """
    r = retornos[np.isfinite(retornos)]
    if len(r) < 120:
        return np.nan, np.array([])
    var = np.empty(len(r))
    var[0] = np.var(r[:60])
    for i in range(1, len(r)):
        lim = corte * np.sqrt(var[i - 1])
        var[i] = lam * var[i - 1] + (1 - lam) * np.clip(r[i - 1], -lim, lim) ** 2
    z = r[20:] / np.sqrt(var[20:])
    lim = corte * np.sqrt(var[-1])
    var_atual = lam * var[-1] + (1 - lam) * np.clip(r[-1], -lim, lim) ** 2
    ult = r[-252:]
    lim_1a = corte * ult.std()
    var_1a = np.mean(np.clip(ult - ult.mean(), -lim_1a, lim_1a) ** 2)
    return float(np.sqrt(0.5 * var_atual + 0.5 * var_1a)), z


def amostras_preco_final(
    retornos: np.ndarray | None, S0: float, du: int, r: float, n: int = 800, iv_fallback: float | None = None,
    ref: tuple[float, np.ndarray] | None = None,
) -> np.ndarray:
    """`n` cenários equiprováveis para o preço no vencimento.

    Usa os choques históricos padronizados do próprio ativo, somados em janelas de `du` pregões e
    reescalados pela vol atual (preserva caudas gordas e assimetria). Média ajustada ao forward
    (drift = taxa livre de risco). Sem histórico suficiente, usa lognormal com `iv_fallback`.
    """
    du = max(int(du), 1)
    T = du / DU_ANO
    if ref is not None:  # já calculado por quem chama (`vol_referencia`)
        sig_d, z = ref
    else:
        sig_d, z = vol_referencia(retornos) if retornos is not None else (np.nan, np.array([]))
    if np.isfinite(sig_d) and len(z) > du + 120:
        cs = np.concatenate([[0.0], np.cumsum(z)])
        janelas = (cs[du:] - cs[:-du]) / np.sqrt(du)
        janelas = janelas / janelas.std()
        X = janelas * sig_d * np.sqrt(du)
    else:
        sig = iv_fallback if iv_fallback and np.isfinite(iv_fallback) else 0.35
        q = (np.arange(n) + 0.5) / n
        from scipy.special import ndtri

        X = ndtri(q) * sig * np.sqrt(T)
    X = np.sort(X)
    if len(X) > n:
        X = X[np.linspace(0, len(X) - 1, n).round().astype(int)]
    X = X + (r * T - np.log(np.mean(np.exp(X))))
    return S0 * np.exp(X)


def vol_historica(fechamentos: pd.Series, janela: int) -> float:
    r = np.log(fechamentos.astype(float)).diff().dropna()
    if len(r) < janela:
        return np.nan
    return float(r.tail(janela).std() * np.sqrt(DU_ANO))
