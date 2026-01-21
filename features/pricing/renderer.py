import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from .payoff import payoff_strategy, break_evens, minmax
from .calendar_approx import payoff_booster_puts_approx


def plot_payoff_expiry(
    op: dict,
    spot: float,
    multiplier: float,
    show_pct: bool,
    st_lo: float | None = None,
    st_hi: float | None = None,
):
    """Payoff no vencimento.

    `st_lo`/`st_hi` permitem expandir o eixo X para deixar o tail-risk visível
    (ex.: ratio spreads).
    """
    s0 = float(spot) if (spot is not None and np.isfinite(spot) and spot > 0) else 1.0
    lo = max(0.01, float(st_lo) if (st_lo is not None and np.isfinite(st_lo) and st_lo > 0) else 0.5 * s0)
    hi = float(st_hi) if (st_hi is not None and np.isfinite(st_hi) and st_hi > lo) else 1.5 * s0
    ST = np.linspace(lo, hi, 800)

    pnl = payoff_strategy(op, ST, s0) * float(multiplier)
    denom = s0 * float(multiplier)
    pnl_plot = (pnl / denom) * 100.0 if show_pct else pnl

    be = break_evens(ST, pnl, tol=1e-6)
    mn, mx = minmax(pnl)

    fig = plt.figure()
    plt.plot(ST, pnl_plot, linestyle="-")
    plt.axhline(0.0)
    plt.axvline(s0)
    for xbe in be:
        plt.axvline(xbe, linestyle="--")
    plt.xlabel("Preço do ativo no vencimento (ST)")
    plt.ylabel("Payoff (% do spot)" if show_pct else "Payoff (P&L no vencimento)")
    plt.title(op["label"])
    st.pyplot(fig, clear_figure=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Break-even(s)**")
        st.write(", ".join([f"{v:.2f}" for v in be]) if be else "—")
    with c2:
        st.write("**Perda máx (na faixa do gráfico)**")
        st.write(f"{mn:.2f}" + (f" ({(mn/denom*100):.2f}%)" if show_pct else ""))
    with c3:
        st.write("**Ganho máx (na faixa do gráfico)**")
        st.write(f"{mx:.2f}" + (f" ({(mx/denom*100):.2f}%)" if show_pct else ""))

    st.caption(f"Multiplicador: {multiplier:g} | S0={s0:.2f}")

def plot_payoff_booster_calendar(spec: dict, multiplier: float, show_pct: bool, st_lo: float | None = None, st_hi: float | None = None):
    s0 = float(spec.get("spot_ref", 1.0))
    lo = max(0.01, float(st_lo) if (st_lo is not None and np.isfinite(st_lo) and st_lo > 0) else 0.5 * s0)
    hi = float(st_hi) if (st_hi is not None and np.isfinite(st_hi) and st_hi > lo) else 1.5 * s0
    ST = np.linspace(lo, hi, 800)

    K = float(spec["K"])
    P_long = float(spec["premium_long"])
    P_short = float(spec["premium_short"])
    t_short = float(spec.get("t_short", 0.0))
    t_long  = float(spec.get("t_long", 0.0))

    pnl = payoff_booster_puts_approx(ST, K, P_long, P_short, t_short, t_long) * float(multiplier)
    denom = s0 * float(multiplier)
    pnl_plot = (pnl / denom) * 100.0 if show_pct else pnl

    fig = plt.figure()
    plt.plot(ST, pnl_plot, linestyle="-")
    plt.axhline(0.0)
    plt.axvline(s0)
    plt.xlabel("Preço do ativo no venc. curto (ST)")
    plt.ylabel("Payoff (% do spot)" if show_pct else "Payoff (aprox.)")
    plt.title(spec["label"] + " (aprox no venc. curto)")
    st.pyplot(fig, clear_figure=True)

    mn, mx = float(np.min(pnl)), float(np.max(pnl))
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Perda máx (faixa do gráfico)**")
        st.write(f"{mn:.2f}" + (f" ({(mn/denom*100):.2f}%)" if show_pct else ""))
    with c2:
        st.write("**Ganho máx (faixa do gráfico)**")
        st.write(f"{mx:.2f}" + (f" ({(mx/denom*100):.2f}%)" if show_pct else ""))

    st.caption(
        f"Multiplicador: {multiplier:g} | S0={s0:.2f} | "
        f"K={K:.2f} | net={P_short-P_long:.2f} | "
        f"t_short={t_short:.4f} | t_long={t_long:.4f}"
    )

def render_payoff(spec: dict, multiplier: float, show_pct: bool, payoff_cfg: dict | None = None):
    payoff_cfg = payoff_cfg or {}
    mode = spec.get("payoff_mode", "expiry")
    # Range selection: manual (sidebar) > strategy hint > default
    s0 = float(spec.get("spot_ref", 1.0)) if np.isfinite(float(spec.get("spot_ref", 1.0))) else 1.0
    st_lo = None
    st_hi = None
    if payoff_cfg.get("range_mode") == "manual":
        lo_mult = float(payoff_cfg.get("lo_mult", 0.5))
        hi_mult = float(payoff_cfg.get("hi_mult", 1.5))
        st_lo = max(0.01, lo_mult * s0)
        st_hi = max(st_lo * 1.001, hi_mult * s0)
    else:
        hint = spec.get("payoff_range_mult")
        if isinstance(hint, (list, tuple)) and len(hint) == 2:
            st_lo = max(0.01, float(hint[0]) * s0)
            st_hi = max(st_lo * 1.001, float(hint[1]) * s0)
        else:
            st_lo = max(0.01, 0.5 * s0)
            st_hi = 1.5 * s0

    if mode == "expiry":
        plot_payoff_expiry(spec, float(spec.get("spot_ref", 1.0)), multiplier, show_pct, st_lo=st_lo, st_hi=st_hi)
    elif mode == "calendar_approx":
        plot_payoff_booster_calendar(spec, multiplier, show_pct, st_lo=st_lo, st_hi=st_hi)
    else:
        st.info("Sem payoff para esta estratégia.")
