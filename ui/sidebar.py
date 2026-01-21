import streamlit as st

def build_sidebar(assets, expiry_list):
    st.sidebar.markdown("## Universo")
    ticker_ui = st.sidebar.selectbox("Ativo (opcional)", ["(Todos)"] + assets["ticker"].tolist())

    st.sidebar.markdown("## Payoff (config)")
    show_pct = st.sidebar.toggle("Exibir payoff em % do spot", value=False)
    mult_100 = st.sidebar.toggle("Payoff por 100 ações (multiplicador=100)", value=True)
    multiplier = 100.0 if mult_100 else 1.0

    st.sidebar.markdown("## Payoff (range)")
    range_mode = st.sidebar.radio("Faixa ST", options=["Auto", "Manual"], index=0, horizontal=True)
    if range_mode == "Manual":
        lo_mult = st.sidebar.slider("ST mínimo (×S0)", 0.05, 1.00, 0.50, 0.05)
        hi_mult = st.sidebar.slider("ST máximo (×S0)", 1.10, 4.00, 1.50, 0.10)
    else:
        lo_mult, hi_mult = 0.5, 1.5


    st.sidebar.markdown("## Tendência (daily_bars)")
    rsi_hi = st.sidebar.slider("RSI alto (força)", 50, 70, 55, 1)
    rsi_lo = st.sidebar.slider("RSI baixo (fraqueza)", 30, 50, 45, 1)

    st.sidebar.markdown("## ATM")
    atm_mode_ui = st.sidebar.radio("ATM", options=["Faixa percentual", "Mais próximo"], index=0)
    atm_mode = "pct" if atm_mode_ui == "Faixa percentual" else "nearest"
    atm_pct = st.sidebar.slider("Faixa ATM (|K−S|/S)", 0.001, 0.05, 0.01, 0.001, disabled=(atm_mode != "pct"))

    st.sidebar.markdown("## Liquidez (gates)")
    liq_single_filter_hard = st.sidebar.toggle("Filtrar RUIM (perna única)", value=True)
    liq_pair_filter_hard = st.sidebar.toggle("Filtrar RUIM (pares)", value=True)

    st.sidebar.markdown("## Filtro do universo (recomendado)")
    use_universe_filter = st.sidebar.toggle("Ativar filtro strike/delta/último/vol financeiro", value=True)
    mny_log_max = st.sidebar.slider("|ln(K/S)| máx", 0.10, 1.00, 0.40, 0.05)
    delta_abs_min = st.sidebar.slider("|delta| mín", 0.00, 0.50, 0.05, 0.01)
    last_price_min = st.sidebar.number_input("Último (last_price) mín", min_value=0.0, value=0.05, step=0.01)

    opt_contract_mult = st.sidebar.number_input("Multiplicador do contrato (p/ volume financeiro)", min_value=1, value=100, step=1)
    vol_fin_min = st.sidebar.number_input("Volume financeiro mín (R$)", min_value=0.0, value=5000.0, step=500.0)

    expiry_ui = st.sidebar.selectbox("Vencimento (opcional)", ["(Todos)"] + [str(x) for x in expiry_list])
    expiry_sel = None if expiry_ui == "(Todos)" else __import__("pandas").to_datetime(expiry_ui).date()

    return {
        "ticker_ui": ticker_ui,
        "show_pct": show_pct,
        "multiplier": multiplier,
        "payoff_range_mode": ("manual" if range_mode=="Manual" else "auto"),
        "payoff_lo_mult": float(lo_mult),
        "payoff_hi_mult": float(hi_mult),
        "rsi_hi": float(rsi_hi),
        "rsi_lo": float(rsi_lo),
        "atm_mode": atm_mode,
        "atm_pct": float(atm_pct),
        "liq_single_filter_hard": bool(liq_single_filter_hard),
        "liq_pair_filter_hard": bool(liq_pair_filter_hard),
        "use_universe_filter": bool(use_universe_filter),
        "mny_log_max": float(mny_log_max),
        "delta_abs_min": float(delta_abs_min),
        "last_price_min": float(last_price_min),
        "opt_contract_mult": float(opt_contract_mult),
        "vol_fin_min": float(vol_fin_min),
        "expiry_sel": expiry_sel,
    }
