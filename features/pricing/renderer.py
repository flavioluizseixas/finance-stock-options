import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def payoff_long_call(ST, K, premium): return np.maximum(ST - K, 0.0) - premium
def payoff_short_call(ST, K, premium): return premium - np.maximum(ST - K, 0.0)
def payoff_long_put(ST, K, premium): return np.maximum(K - ST, 0.0) - premium
def payoff_short_put(ST, K, premium): return premium - np.maximum(K - ST, 0.0)
def payoff_stock(ST, S0): return ST - S0

def payoff_strategy(spec: dict, ST: np.ndarray, spot: float) -> np.ndarray:
    total = np.zeros_like(ST, dtype=float)
    if bool(spec.get("include_stock", False)):
        total += float(spec.get("stock_qty", 1.0)) * payoff_stock(ST, spot)
    for lg in spec.get("legs", []):
        typ = str(lg.get("type","")).upper()
        side = str(lg.get("side","")).upper()
        K = float(lg.get("K", 0.0))
        prem = float(lg.get("premium", 0.0))
        if typ=="CALL" and side=="LONG": total += payoff_long_call(ST, K, prem)
        elif typ=="CALL" and side=="SHORT": total += payoff_short_call(ST, K, prem)
        elif typ=="PUT" and side=="LONG": total += payoff_long_put(ST, K, prem)
        elif typ=="PUT" and side=="SHORT": total += payoff_short_put(ST, K, prem)
    return total

def find_break_evens(ST, pnl, tol=1e-8):
    be=[]
    y=pnl; x=ST
    near=np.where(np.abs(y)<tol)[0]
    for i in near: be.append(float(x[i]))
    s=np.sign(y)
    for i in range(len(x)-1):
        if s[i]==0: 
            continue
        if s[i]*s[i+1]<0:
            x0,x1=x[i],x[i+1]; y0,y1=y[i],y[i+1]
            # linear interpolation root
            if (y1 - y0) != 0:
                be.append(float(x0 - y0*(x1-x0)/(y1-y0)))
    return sorted(set([round(v,6) for v in be]))

def _infer_s0(spec: dict) -> float:
    """Try spot_ref/spot/S0; if missing, infer from strikes."""
    for k in ("spot_ref","spot","S0","s0"):
        v = spec.get(k, None)
        try:
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv) and fv > 0:
                return fv
        except Exception:
            pass
    strikes = []
    for lg in spec.get("legs", []):
        try:
            strikes.append(float(lg.get("K", np.nan)))
        except Exception:
            pass
    strikes = [x for x in strikes if np.isfinite(x) and x > 0]
    if strikes:
        return float(np.median(strikes))
    return 1.0

def _auto_range_from_legs(s0: float, legs: list[dict], lo_mult: float, hi_mult: float):
    """Make sure the plotted ST range covers relevant strikes (so BE is visible)."""
    lo = max(0.01, lo_mult * s0)
    hi = max(lo * 1.01, hi_mult * s0)

    strikes = []
    has_put = False
    has_call = False
    for lg in legs or []:
        try:
            K = float(lg.get("K", np.nan))
            if np.isfinite(K) and K > 0:
                strikes.append(K)
        except Exception:
            pass
        t = str(lg.get("type","")).upper()
        has_put = has_put or (t == "PUT")
        has_call = has_call or (t == "CALL")

    if strikes:
        kmin = float(min(strikes))
        kmax = float(max(strikes))

        # Expand to include strikes comfortably.
        lo = min(lo, 0.5 * kmin, 0.25 * s0)
        hi = max(hi, 1.5 * kmax, 1.75 * s0)

        # If there are puts, allow more downside so the tail shows up.
        if has_put:
            lo = min(lo, 0.10 * s0)

        # If there are calls, allow more upside so ratio spreads show tail risk.
        if has_call:
            hi = max(hi, 2.50 * s0)

    lo = max(0.01, float(lo))
    hi = max(lo * 1.01, float(hi))
    return lo, hi

def render_payoff(spec: dict, multiplier: float, show_pct: bool, payoff_cfg: dict | None = None):
    payoff_cfg = payoff_cfg or {}

    s0 = _infer_s0(spec)

    # Range config
    if payoff_cfg.get("range_mode","auto") == "manual":
        lo_mult = float(payoff_cfg.get("lo_mult", 0.5))
        hi_mult = float(payoff_cfg.get("hi_mult", 1.5))
        lo = max(0.01, lo_mult * s0)
        hi = max(lo * 1.01, hi_mult * s0)
    else:
        lo, hi = _auto_range_from_legs(s0, spec.get("legs", []), lo_mult=0.5, hi_mult=1.5)

    ST = np.linspace(lo, hi, 900)
    pnl = payoff_strategy(spec, ST, s0) * float(multiplier)

    denom = s0 * float(multiplier)
    if (not np.isfinite(denom)) or denom == 0:
        denom = 1.0
    pnl_plot = (pnl / denom) * 100.0 if show_pct else pnl

    be = find_break_evens(ST, pnl, 1e-6)
    mn = float(np.min(pnl)) if len(pnl) else 0.0
    mx = float(np.max(pnl)) if len(pnl) else 0.0

    fig = plt.figure()
    plt.plot(ST, pnl_plot, linestyle="-")
    plt.axhline(0.0)
    plt.axvline(s0)
    for xbe in be:
        plt.axvline(xbe, linestyle="--")
    plt.xlabel("Preço do ativo no vencimento (ST)")
    plt.ylabel("Payoff (% do spot)" if show_pct else "Payoff (P&L no vencimento)")
    plt.title(spec.get("label","Payoff"))
    st.pyplot(fig, clear_figure=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Break-even(s)**")
        st.write(", ".join([f"{v:.2f}" for v in be]) if be else "—")

    with c2:
        st.write("**Perda máx (na faixa)**")
        st.write(f"{mn:.2f}" + (f" ({(mn/denom*100):.2f}%)" if show_pct else ""))

    with c3:
        st.write("**Ganho máx (na faixa)**")
        st.write(f"{mx:.2f}" + (f" ({(mx/denom*100):.2f}%)" if show_pct else ""))

    st.caption(f"S0={s0:.2f} | ST range=[{lo:.2f}, {hi:.2f}] | multiplicador={multiplier:g}")
