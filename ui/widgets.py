import pandas as pd
import streamlit as st

def selectable_table(df: pd.DataFrame, key: str, label: str):
    if df is None or df.empty:
        st.info("Sem operações candidatas para esta estratégia.")
        return None

    disp = df.copy().reset_index(drop=True)
    st.caption(label)

    try:
        evt = st.dataframe(
            disp,
            use_container_width=True,
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
        options = [f"{i}: {disp.iloc[i].to_dict()}" for i in range(len(disp))]
        pick = st.selectbox("Escolha a operação para ver o payoff:", ["(nenhuma)"] + options, key=f"{key}_sb")
        if pick == "(nenhuma)":
            return None
        return int(pick.split(":")[0])

def format_table(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    return out  # Mantemos simples; você pode plugar o formatter completo aqui.
