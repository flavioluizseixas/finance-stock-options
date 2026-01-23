import streamlit as st
import pandas as pd

def build_sidebar(assets: pd.DataFrame, expiry_list0: list):
    st.sidebar.markdown("## Universo")
    tickers = ["(Todos)"] + assets["ticker"].tolist()
    ticker_ui = st.sidebar.selectbox("Ativo", tickers, index=0)

    st.sidebar.markdown("## Vencimento")
    exp_opts = ["(Todos)"] + [str(x) for x in expiry_list0]
    exp_pick = st.sidebar.selectbox("Vencimento", exp_opts, index=0)
    expiry_sel = None if exp_pick == "(Todos)" else pd.to_datetime(exp_pick).date()

    st.sidebar.markdown("## Payoff")
    show_pct = st.sidebar.toggle("Exibir payoff em % do spot", value=False)
    mult_100 = st.sidebar.toggle("Payoff por 100 ações (multiplicador=100)", value=True)
    multiplier = 100.0 if mult_100 else 1.0

    payoff_range_mode = st.sidebar.radio("Faixa do gráfico", ["auto","manual"], index=0)
    payoff_lo_mult = st.sidebar.number_input("lo_mult (manual)", value=0.5, step=0.1, disabled=(payoff_range_mode!="manual"))
    payoff_hi_mult = st.sidebar.number_input("hi_mult (manual)", value=1.5, step=0.1, disabled=(payoff_range_mode!="manual"))

    st.sidebar.markdown("## Tendência (regime)")
    rsi_hi = st.sidebar.slider("RSI alto (força)", 50, 70, 55, 1)
    rsi_lo = st.sidebar.slider("RSI baixo (fraqueza)", 30, 50, 45, 1)

    st.sidebar.markdown("## ATM")
    atm_mode_ui = st.sidebar.radio("Modo ATM", ["Faixa percentual","Mais próximo"], index=0)
    atm_mode = "pct" if atm_mode_ui=="Faixa percentual" else "nearest"
    atm_pct = st.sidebar.slider("Faixa ATM (|K−S|/S)", 0.001, 0.05, 0.01, 0.001, disabled=(atm_mode!="pct"))

    st.sidebar.markdown("## Filtros de Universo (opcional)")
    use_universe_filter = st.sidebar.toggle("Ativar filtro de universo", value=False)
    mny_log_max = st.sidebar.slider("Mny log max (placeholder)", 1.0, 10.0, 6.0, 0.5, disabled=not use_universe_filter)
    delta_abs_min = st.sidebar.slider("|delta| mínimo", 0.0, 0.9, 0.05, 0.01, disabled=not use_universe_filter)
    last_price_min = st.sidebar.number_input("Preço mínimo (last_price)", value=0.01, step=0.01, disabled=not use_universe_filter)
    vol_fin_min = st.sidebar.number_input("Volume financeiro mínimo (preço*volume)", value=0.0, step=100.0, disabled=not use_universe_filter)
    opt_contract_mult = st.sidebar.number_input("Multiplicador contrato (informativo)", value=100.0, step=1.0, disabled=not use_universe_filter)

    st.sidebar.markdown("## Liquidez")
    liq_single_filter_hard = st.sidebar.toggle("Filtro hard (perna)", value=False)
    liq_pair_filter_hard = st.sidebar.toggle("Filtro hard (par)", value=False)

    return {
        "ticker_ui": ticker_ui,
        "expiry_sel": expiry_sel,
        "show_pct": show_pct,
        "multiplier": multiplier,
        "payoff_range_mode": payoff_range_mode,
        "payoff_lo_mult": payoff_lo_mult,
        "payoff_hi_mult": payoff_hi_mult,
        "rsi_hi": rsi_hi,
        "rsi_lo": rsi_lo,
        "atm_mode": atm_mode,
        "atm_pct": atm_pct,
        "use_universe_filter": use_universe_filter,
        "mny_log_max": mny_log_max,
        "delta_abs_min": delta_abs_min,
        "last_price_min": last_price_min,
        "vol_fin_min": vol_fin_min,
        "opt_contract_mult": opt_contract_mult,
        "liq_single_filter_hard": liq_single_filter_hard,
        "liq_pair_filter_hard": liq_pair_filter_hard,
    }
