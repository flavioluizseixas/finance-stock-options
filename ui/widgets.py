# ui/widgets.py
import pandas as pd
import streamlit as st

def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Exemplos de formatação segura (opcional)
    # for c in out.columns:
    #     if pd.api.types.is_float_dtype(out[c]):
    #         out[c] = out[c].round(4)

    return out

def selectable_table(df: pd.DataFrame, key: str, label: str = "Selecione uma linha"):
    """
    Exibe um dataframe com seleção de linha via st.data_editor e retorna o índice selecionado (int) ou None.
    """
    if df is None or df.empty:
        st.info("Sem dados para exibir.")
        return None

    dfx = df.reset_index(drop=True).copy()

    # cria coluna de seleção
    sel_col = "__sel__"
    if sel_col not in dfx.columns:
        dfx.insert(0, sel_col, False)

    st.caption(label)

    edited = st.data_editor(
        dfx,
        key=f"{key}_editor",
        hide_index=True,
        width="stretch",
        height=min(420, 35 + 35 * (len(dfx) if len(dfx) < 10 else 10)),
        column_config={
            sel_col: st.column_config.CheckboxColumn(""),
        },
        disabled=[c for c in dfx.columns if c != sel_col],
    )

    # retorna a primeira linha marcada
    picked = edited.index[edited[sel_col] == True].tolist()
    if not picked:
        return None
    return int(picked[0])
