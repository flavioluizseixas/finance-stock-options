from __future__ import annotations

import io
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from store import fetch_csv, normalize, load_parquet, update_parquet, PARQUET_PATH

st.set_page_config(page_title="Tesouro Direto — Vendas", layout="wide")
st.title("Tesouro Direto — Vendas (PU e Quantidade)")

# =========================
# Sidebar: atualização + parâmetros do scanner
# =========================
with st.sidebar:
    st.header("Dados")
    if st.button("🔄 Atualizar (CSV → Parquet incremental)", use_container_width=True):
        with st.spinner("Baixando CSV, normalizando e atualizando Parquet..."):
            raw = fetch_csv()
            df_new = normalize(raw)
            stats = update_parquet(df_new)
        st.success("Atualizado!")
        st.json(stats)

    st.caption(f"Parquet: {PARQUET_PATH}")

    st.divider()
    st.header("Scanner de oportunidade")
    st.caption("Heurística baseada em **PU histórico** + **tempo até vencimento** (proxy de sensibilidade a queda de juros).")

    lookback_years = st.slider("Janela histórica (anos)", min_value=1, max_value=10, value=5, step=1)
    weight_maturity = st.slider("Peso do vencimento (duration proxy)", 0.0, 1.0, 0.55, 0.05)
    top_n = st.slider("Mostrar Top N", 5, 30, 12, 1)

# =========================
# Carrega Parquet
# =========================
df = load_parquet()
if df.empty:
    st.warning("Base Parquet vazia/inexistente. Clique em **Atualizar** na barra lateral.")
    st.stop()

df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
df["vencimento"] = pd.to_datetime(df["vencimento"], errors="coerce")
df = df.dropna(subset=["data_venda", "tipo_titulo", "ano_vencimento", "vencimento", "pu", "quantidade"])

ultima_data_global = df["data_venda"].max()

top1, top2, top3 = st.columns([1, 1, 2])
top1.metric("Linhas no Parquet", f"{len(df):,}".replace(",", "."))
top2.metric("Última data global", str(ultima_data_global.date()))
top3.caption("Os gráficos respeitam a sequência temporal pela coluna **Data Venda**.")

st.divider()

# =====================================================================
# 📌 Scanner: "melhor oportunidade" (heurística para queda de juros)
# =====================================================================
st.subheader("📌 Oportunidade de compra (cenário de queda de juros)")

# Para ficar coerente, usamos série diária agregada por (tipo, vencimento, dia)
df_sc = df.copy()
df_sc["data"] = df_sc["data_venda"].dt.floor("D")

# janela histórica
start_date = (ultima_data_global - pd.DateOffset(years=lookback_years)).floor("D")
df_sc = df_sc[df_sc["data"] >= start_date].copy()

daily = (
    df_sc.groupby(["tipo_titulo", "vencimento", "ano_vencimento", "data"], as_index=False)
         .agg(pu=("pu", "mean"), quantidade=("quantidade", "sum"))
)
daily = daily.sort_values(["tipo_titulo", "vencimento", "data"]).reset_index(drop=True)

# último dia por título (tipo + vencimento)
last_day = daily.groupby(["tipo_titulo", "vencimento"], as_index=False)["data"].max().rename(columns={"data":"last_date"})
last_vals = daily.merge(last_day, on=["tipo_titulo","vencimento"], how="inner")
last_vals = last_vals[last_vals["data"] == last_vals["last_date"]].copy()

# métricas por título (na janela)
agg_stats = (
    daily.groupby(["tipo_titulo","vencimento","ano_vencimento"], as_index=False)
         .agg(
             pu_min=("pu","min"),
             pu_max=("pu","max"),
             pu_mean=("pu","mean"),
             pu_std=("pu","std"),
             pu_p10=("pu", lambda s: np.nanpercentile(s, 10)),
             pu_p25=("pu", lambda s: np.nanpercentile(s, 25)),
             pu_p50=("pu", lambda s: np.nanpercentile(s, 50)),
         )
)

rank = last_vals.merge(agg_stats, on=["tipo_titulo","vencimento","ano_vencimento"], how="left")
rank = rank.rename(columns={"pu":"pu_atual", "quantidade":"qtd_ultima_data"})
rank["pu_std"] = rank["pu_std"].replace({0: np.nan})

# drawdown vs máximo da janela: quanto abaixo do pico o PU está
rank["drawdown_pct"] = (rank["pu_atual"] / rank["pu_max"] - 1.0) * 100.0

# z-score: quão abaixo/ acima da média
rank["zscore"] = (rank["pu_atual"] - rank["pu_mean"]) / rank["pu_std"]

# percentil aproximado via posição entre min e max (robusto e simples)
# (0 = mínimo histórico da janela, 1 = máximo)
rank["pos_01"] = (rank["pu_atual"] - rank["pu_min"]) / (rank["pu_max"] - rank["pu_min"])
rank["pos_01"] = rank["pos_01"].clip(0, 1)

# "barato" = 1 - pos_01  (quanto mais perto do mínimo, melhor)
rank["cheapness"] = 1.0 - rank["pos_01"]

# tempo até vencimento (anos) no último dia disponível
rank["anos_ate_venc"] = (rank["vencimento"] - rank["last_date"]).dt.days / 365.25
rank["anos_ate_venc"] = rank["anos_ate_venc"].clip(lower=0)

# normaliza maturidade para 0..1
max_years = max(rank["anos_ate_venc"].max(), 1e-9)
rank["maturity_01"] = (rank["anos_ate_venc"] / max_years).clip(0, 1)

# score final (0..1): barato vs histórico + maturidade (proxy duration)
w = float(weight_maturity)
rank["score"] = (1 - w) * rank["cheapness"] + w * rank["maturity_01"]

# ordena por score desc
rank = rank.sort_values(["score", "cheapness", "anos_ate_venc"], ascending=[False, False, False]).reset_index(drop=True)

# Mostra explicação curta (sem recomendação determinística)
st.caption(
    "Interpretação: **score maior** sugere (i) PU relativamente mais baixo no histórico recente "
    "e (ii) vencimento mais longo (tende a reagir mais a queda de juros). "
    "Isso é um **scanner heurístico**; não substitui avaliação de risco/objetivo."
)

# tabela top
show_cols = [
    "tipo_titulo", "ano_vencimento", "last_date",
    "pu_atual", "pu_p25", "pu_p50", "pu_max",
    "drawdown_pct", "anos_ate_venc", "cheapness", "score"
]
top = rank[show_cols].head(top_n).copy()
top = top.rename(columns={
    "tipo_titulo": "Tipo",
    "ano_vencimento": "Venc",
    "last_date": "Última Data",
    "pu_atual": "PU (atual)",
    "pu_p25": "PU P25",
    "pu_p50": "PU P50",
    "pu_max": "PU Máx (janela)",
    "drawdown_pct": "Drawdown %",
    "anos_ate_venc": "Anos até venc",
    "cheapness": "Barato (0-1)",
    "score": "Score (0-1)"
})
st.dataframe(top, use_container_width=True)

st.divider()

# =========================
# Filtros para gráfico (visualização)
# =========================
st.subheader("Seleção")

col1, col2, col3, col4 = st.columns([2.2, 2.2, 1.2, 2.4])

tipos = sorted(df["tipo_titulo"].dropna().unique().tolist())
with col1:
    tipo_sel = st.selectbox("Tipo do título", tipos, index=0)

anos_disponiveis = sorted(df.loc[df["tipo_titulo"] == tipo_sel, "ano_vencimento"].dropna().unique().tolist())
default_anos = [a for a in [2035, 2045] if a in anos_disponiveis] or (anos_disponiveis[:2] if len(anos_disponiveis) >= 2 else anos_disponiveis)

with col2:
    anos_sel = st.multiselect(
        "Ano(s) do vencimento para comparar",
        options=anos_disponiveis,
        default=default_anos,
    )

with col3:
    mm_window = st.number_input("Média móvel (dias)", min_value=1, max_value=365, value=20, step=1)

with col4:
    agg_mode = st.selectbox(
        "Agregação",
        ["Diário (média PU, soma quantidade)", "Sem agregação (cada venda)"],
        index=0
    )

if not anos_sel:
    st.info("Selecione pelo menos um ano de vencimento para visualizar.")
    st.stop()

# =========================
# Filtra
# =========================
dff = df[(df["tipo_titulo"] == tipo_sel) & (df["ano_vencimento"].isin(anos_sel))].copy()
dff = dff.dropna(subset=["data_venda", "pu", "quantidade"]).sort_values("data_venda")

ultima_data_filtro = dff["data_venda"].max()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Última data (filtro)", str(ultima_data_filtro.date()) if pd.notna(ultima_data_filtro) else "—")
k2.metric("Vencimentos", ", ".join(map(str, sorted(anos_sel))))
k3.metric("Registros brutos (filtro)", f"{len(dff):,}".replace(",", "."))
k4.metric("PU (último registro)", f"{dff['pu'].iloc[-1]:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# =========================
# Normaliza eixo temporal + ordena cronologicamente
# =========================
if agg_mode.startswith("Diário"):
    dff["data"] = dff["data_venda"].dt.floor("D")
    dff = (
        dff.groupby(["ano_vencimento", "data"], as_index=False)
           .agg(pu=("pu", "mean"), quantidade=("quantidade", "sum"))
    )
else:
    dff["data"] = dff["data_venda"]

dff["data"] = pd.to_datetime(dff["data"], errors="coerce")
dff = dff.dropna(subset=["data"]).sort_values(["data", "ano_vencimento"]).reset_index(drop=True)

# =========================
# Média móvel do PU
# =========================
dff = dff.sort_values(["ano_vencimento", "data"])
dff["pu_mm"] = dff.groupby("ano_vencimento")["pu"].transform(
    lambda s: s.rolling(mm_window, min_periods=1).mean()
)

# volume total por data
vol = (
    dff.groupby("data", as_index=False)["quantidade"]
       .sum()
       .sort_values("data")
)

# =========================
# Gráfico apresentável: 2 painéis
# =========================
st.subheader("PU e Quantidade (sequência temporal por Data Venda)")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=("PU (com média móvel)", "Quantidade (volume)")
)

for ano in sorted(anos_sel):
    sub = dff[dff["ano_vencimento"] == ano].sort_values("data")
    if sub.empty:
        continue

    fig.add_trace(
        go.Scatter(
            x=sub["data"], y=sub["pu"],
            mode="lines",
            name=f"PU {ano}",
            line=dict(width=2),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sub["data"], y=sub["pu_mm"],
            mode="lines",
            name=f"MM{mm_window} {ano}",
            line=dict(width=2, dash="dot"),
        ),
        row=1, col=1
    )

fig.add_trace(
    go.Bar(
        x=vol["data"], y=vol["quantidade"],
        name="Quantidade (total)",
        opacity=0.55,
    ),
    row=2, col=1
)

fig.update_layout(
    height=680,
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(title_text="Data Venda", type="date", row=2, col=1)
fig.update_yaxes(title_text="PU", row=1, col=1)
fig.update_yaxes(title_text="Quantidade", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Exportação (CSV / Parquet)
# =========================
st.subheader("Exportar dados filtrados")

export_df = dff[["data", "ano_vencimento", "pu", "pu_mm", "quantidade"]].copy()
export_df = export_df.sort_values(["data", "ano_vencimento"]).reset_index(drop=True)

c1, c2 = st.columns(2)

with c1:
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar CSV (filtrado)",
        data=csv_bytes,
        file_name=f"tesouro_vendas_{tipo_sel.replace(' ','_')}_{'_'.join(map(str,sorted(anos_sel)))}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    buf = io.BytesIO()
    export_df.to_parquet(buf, index=False)
    st.download_button(
        label="⬇️ Baixar Parquet (filtrado)",
        data=buf.getvalue(),
        file_name=f"tesouro_vendas_{tipo_sel.replace(' ','_')}_{'_'.join(map(str,sorted(anos_sel)))}.parquet",
        mime="application/octet-stream",
        use_container_width=True,
    )

with st.expander("Ver tabela (top 1.000 linhas)"):
    st.dataframe(export_df.head(1000), use_container_width=True)
