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
        typ = str(lg["type"]).upper()
        side = str(lg["side"]).upper()
        K = float(lg["K"])
        prem = float(lg["premium"])
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
        if s[i]==0: continue
        if s[i]*s[i+1]<0:
            x0,x1=x[i],x[i+1]; y0,y1=y[i],y[i+1]
            be.append(float(x0 - y0*(x1-x0)/(y1-y0)))
    return sorted(set([round(v,6) for v in be]))

def render_payoff(spec: dict, multiplier: float, show_pct: bool, payoff_cfg: dict | None = None):
    payoff_cfg = payoff_cfg or {}
    s0=float(spec.get("spot_ref",1.0))
    s0 = 1.0 if (not np.isfinite(s0) or s0<=0) else s0

    if payoff_cfg.get("range_mode","auto")=="manual":
        lo_mult=float(payoff_cfg.get("lo_mult",0.5))
        hi_mult=float(payoff_cfg.get("hi_mult",1.5))
    else:
        lo_mult, hi_mult = 0.5, 1.5

    lo=max(0.01, lo_mult*s0)
    hi=max(lo*1.01, hi_mult*s0)
    ST=np.linspace(lo, hi, 700)
    pnl=payoff_strategy(spec, ST, s0)*float(multiplier)

    denom = s0 * float(multiplier)
    # Safety: avoid division by zero when showing % payoff.
    if (not np.isfinite(denom)) or denom == 0:
        denom = 1.0
    pnl_plot = (pnl / denom) * 100.0 if show_pct else pnl

    be = find_break_evens(ST, pnl, 1e-6)
    mn = float(np.min(pnl)) if len(pnl) else 0.0
    mx = float(np.max(pnl)) if len(pnl) else 0.0
    fig=plt.figure()
    plt.plot(ST,pnl_plot)
    plt.axhline(0.0)
    plt.axvline(s0)
    for xbe in be: plt.axvline(xbe, linestyle="--")
    plt.title(spec.get("label","Payoff"))
    st.pyplot(fig, clear_figure=True)

    # --- Breakeven + min/max (na faixa do gráfico) ---
    c1, c2, c3 = st.columns(3)
    denom = (s0 * float(multiplier)) if (s0 and np.isfinite(s0)) else 1.0

    with c1:
        st.write("**Break-even(s)**")
        if be:
            st.write(", ".join([f"{v:.2f}" for v in be]))
        else:
            st.write("—")

    with c2:
        st.write("**Perda máx (na faixa)**")
        st.write(f"{mn:.2f}" + (f" ({(mn/denom*100):.2f}%)" if show_pct else ""))

    with c3:
        st.write("**Ganho máx (na faixa)**")
        st.write(f"{mx:.2f}" + (f" ({(mx/denom*100):.2f}%)" if show_pct else ""))
