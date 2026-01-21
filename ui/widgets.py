import numpy as np
import pandas as pd
import streamlit as st

def format_table(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    num_cols = [
        "strike","last_price","trades","volume","moneyness_dist",
        "iv","bsm_price","bsm_price_histvol","mispricing","mispricing_pct",
        "delta","gamma","vega","theta","rho","rate_r","spot_ref",
        "liq","liq_buy","liq_sell","liq_call","liq_put",
        "liq_min","liq_ratio","premium_total","debit","credit","rr","cr","rr_adj","cr_adj",
        "mny_log_abs","volume_fin","K_buy","K_sell","K","P_long","P_short","net_credit"
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "theta" in out.columns:
        out["theta_day_365"] = out["theta"] / 365.0
        out["theta_day_252"] = out["theta"] / 252.0
    if "vega" in out.columns:
        out["vega_1pct"] = out["vega"] * 0.01
    if "iv" in out.columns:
        out["iv_pct"] = out["iv"] * 100.0

    # só mostra colunas existentes
    cols = [
        "strategy","ticker","trade_date","regime",
        "expiry_date","expiry_short","expiry_long",
        "spot_ref",
        "option_symbol","option_type",
        "buy","sell","call","put",
        "strike","K","K_buy","K_sell",
        "last_price","P_buy","P_sell","P_call","P_put","P_long","P_short",
        "debit","credit","net_credit","width","max_profit_est","max_loss_est","rr","cr","rr_adj","cr_adj",
        "liq","liq_class","liq_min","liq_ratio",
        "trades","volume","moneyness","moneyness_dist",
        "iv","iv_pct","delta","gamma","vega","theta",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()

    for c in ["spot_ref","strike","K","K_buy","K_sell","last_price","P_buy","P_sell","P_call","P_put","P_long","P_short","debit","credit","net_credit","width","max_profit_est","max_loss_est"]:
        if c in out.columns:
            out[c] = out[c].round(4 if c=="spot_ref" else 2)
    for c in ["moneyness_dist","liq","liq_min","liq_ratio","iv","iv_pct","delta","gamma","vega","theta","rr","cr","rr_adj","cr_adj"]:
        if c in out.columns:
            out[c] = out[c].round(4)

    return out

def selectable_table(df: pd.DataFrame, key: str, label: str):
    if df is None or df.empty:
        st.info("Sem operações candidatas para esta estratégia.")
        return None

    disp = df.copy().reset_index(drop=True)
    st.caption(label)

    try:
        evt = st.dataframe(
            disp,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"{key}_df",
        )
        if evt and getattr(evt, "selection", None):
            rows = evt.selection.get("rows", [])
            if rows:
                return int(rows[0])
        return None
    except TypeError:
        try:
            st.dataframe(disp, width="stretch", hide_index=True)
        except TypeError:
            st.dataframe(disp)

        options = [f"{i}: {disp.iloc[i].to_dict()}" for i in range(len(disp))]
        pick = st.selectbox("Escolha a operação para ver o payoff:", ["(nenhuma)"] + options, key=f"{key}_sb")
        if pick == "(nenhuma)":
            return None
        return int(pick.split(":")[0])

def criteria_box(title: str, bullets: list[str], notes: list[str] | None = None):
    st.markdown(f"**Critérios – {title}:**")
    st.markdown("\n".join([f"- {b}" for b in bullets]))
    if notes:
        st.markdown("\n".join([f"> {n}" for n in notes]))
