"""Formatação de números no padrão brasileiro."""

from __future__ import annotations

import numpy as np


def num(v, casas: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def brl(v, sinal: bool = False) -> str:
    if v is None:
        return "—"
    if np.isinf(v):
        return "ilimitado"
    if not np.isfinite(v):
        return "—"
    pre = "-" if v < 0 else ("+" if sinal and v > 0 else "")
    return f"{pre}R$ {num(abs(v))}"


def pct(v, casas: int = 1, sinal: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v * 100:{'+' if sinal else ''}.{casas}f}%".replace(".", ",")
