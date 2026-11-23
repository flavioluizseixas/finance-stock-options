"""Calendário de pregões da B3 (feriados nacionais + dias sem pregão)."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd


def _pascoa(ano: int) -> date:
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    mes = (h + l_ - 7 * m + 114) // 31
    dia = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(ano, mes, dia)


@lru_cache(maxsize=None)
def feriados(ano: int) -> tuple[date, ...]:
    p = _pascoa(ano)
    dias = [
        date(ano, 1, 1),
        p - timedelta(days=48),  # carnaval (segunda)
        p - timedelta(days=47),  # carnaval (terça)
        p - timedelta(days=2),  # sexta-feira santa
        date(ano, 4, 21),
        date(ano, 5, 1),
        p + timedelta(days=60),  # Corpus Christi
        date(ano, 9, 7),
        date(ano, 10, 12),
        date(ano, 11, 2),
        date(ano, 11, 15),
        date(ano, 12, 24),
        date(ano, 12, 25),
        date(ano, 12, 31),
    ]
    if ano >= 2024:
        dias.append(date(ano, 11, 20))  # Consciência Negra (feriado nacional)
    return tuple(sorted(dias))


@lru_cache(maxsize=None)
def _feriados_np(ano_ini: int, ano_fim: int) -> np.ndarray:
    dias = [d for a in range(ano_ini, ano_fim + 1) for d in feriados(a)]
    return np.array(dias, dtype="datetime64[D]")


def _hol() -> np.ndarray:
    hoje = date.today().year
    return _feriados_np(2015, hoje + 4)


def _to_d64(x) -> np.ndarray:
    if isinstance(x, (date, str, np.datetime64)):
        return np.datetime64(pd.Timestamp(x).date(), "D")
    return pd.to_datetime(pd.Series(x)).to_numpy().astype("datetime64[D]")


def dias_uteis(inicio, fim) -> np.ndarray | int:
    """Pregões em (inicio, fim] — tempo restante a partir do fechamento de `inicio` até `fim`.

    Aceita datas escalares ou arrays (NaT vira 0); valores negativos são truncados em 0.
    """
    ini = _to_d64(inicio)
    fim_ = _to_d64(fim)
    ini, fim_ = np.broadcast_arrays(ini, fim_)
    nat = np.isnat(ini) | np.isnat(fim_)
    um = np.timedelta64(1, "D")
    a = np.where(nat, np.datetime64("2000-01-03"), ini) + um
    b = np.where(nat, np.datetime64("2000-01-03"), fim_) + um
    out = np.maximum(np.busday_count(a, b, holidays=_hol()), 0)
    out = np.where(nat, 0, out)
    return int(out) if np.ndim(out) == 0 else out


def eh_pregao(d: date) -> bool:
    return bool(np.is_busday(np.datetime64(d, "D"), holidays=_hol()))


def pregao_anterior(d: date) -> date:
    x = np.busday_offset(np.datetime64(d, "D"), -1, roll="backward", holidays=_hol())
    return pd.Timestamp(x).date()


def ultimo_pregao(ate: date | None = None) -> date:
    d = ate or date.today()
    x = np.busday_offset(np.datetime64(d, "D"), 0, roll="backward", holidays=_hol())
    return pd.Timestamp(x).date()
