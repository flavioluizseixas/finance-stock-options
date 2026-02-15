from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from store import (
    update_all,
    load_parquet,
    PARQUET_VENDAS,
    PARQUET_PRECOTAXA,
)

st.set_page_config(page_title="Tesouro Direto — Vendas + Taxas", layout="wide")
st.title("Tesouro Direto — Dashboard (Vendas + Preço/Taxa)")

# =========================
# Sidebar: atualização
# =========================
with st.sidebar:
    st.header("Dados")
    if st.button("🔄 Atualizar (Vendas + Preço/Taxa)", use_container_width=True):
        with st.spinner("Baixando e atualizando Parquets..."):
            stats = update_all()
        st.success("Atualizado!")
        st.json(stats)

    st.caption(f"Vendas parquet: {PARQUET_VENDAS}")
    st.caption(f"Preço/Taxa parquet: {PARQUET_PRECOTAXA}")

    st.divider()
    st.header("Scanner (mais financeiro)")
    lookback_years = st.slider("Janela histórica (anos)", 1, 10, 5, 1)
    mm_window = st.slider("Média móvel PU (dias)", 5, 120, 20, 5)
    top_n = st.slider("Top N recomendações", 5, 30, 12, 1)

# =========================
# Carrega dados
# =========================
vendas = load_parquet(PARQUET_VENDAS)
pt = load_parquet(PARQUET_PRECOTAXA)

if vendas.empty or pt.empty:
    st.warning("Parquets vazios/inexistentes. Clique em **Atualizar** na barra lateral.")
    st.stop()

vendas["data_venda"] = pd.to_datetime(vendas["data_venda"], errors="coerce")
pt["data_base"] = pd.to_datetime(pt["data_base"], errors="coerce")
vendas["vencimento"] = pd.to_datetime(vendas["vencimento"], errors="coerce")
pt["vencimento"] = pd.to_datetime(pt["vencimento"], errors="coerce")

vendas = vendas.dropna(subset=["data_venda", "tipo_titulo", "vencimento", "pu_venda", "quantidade"])
pt = pt.dropna(subset=["data_base", "tipo_titulo", "vencimento", "taxa_venda", "pu_venda"])

ultima_vendas = vendas["data_venda"].max()
ultima_pt = pt["data_base"].max()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Última data (Vendas)", str(ultima_vendas.date()))
k2.metric("Última data (Preço/Taxa)", str(ultima_pt.date()))
k3.metric("Linhas Vendas", f"{len(vendas):,}".replace(",", "."))
k4.metric("Linhas Preço/Taxa", f"{len(pt):,}".replace(",", "."))

st.divider()

# =========================
# Scanner com taxa
# =========================
st.subheader("📌 Scanner de oportunidade (queda de juros → PU tende a subir)")
st.caption(
    "Selecione **uma linha** na tabela (clique) para atualizar o gráfico abaixo. "
    "Score combina: taxa alta (vs histórico) + PU barato (vs histórico) + maturidade (proxy)."
)

# janela histórica
end_date = min(ultima_vendas, ultima_pt)
start_date = (end_date - pd.DateOffset(years=lookback_years)).floor("D")

# séries diárias (vendas)
v_daily = vendas.copy()
v_daily["data"] = v_daily["data_venda"].dt.floor("D")
v_daily = v_daily[v_daily["data"] >= start_date]
v_daily = (
    v_daily.groupby(["tipo_titulo", "vencimento", "data"], as_index=False)
           .agg(pu=("pu_venda", "mean"), volume=("quantidade", "sum"))
)

# séries diárias (preço/taxa)
pt_daily = pt.copy()
pt_daily["data"] = pt_daily["data_base"].dt.floor("D")
pt_daily = pt_daily[pt_daily["data"] >= start_date]
pt_daily = (
    pt_daily.groupby(["tipo_titulo", "vencimento", "data"], as_index=False)
            .agg(taxa=("taxa_venda", "mean"), pu_ref=("pu_venda", "mean"))
)

# inner join por data/título/vencimento
m = pd.merge(v_daily, pt_daily, on=["tipo_titulo", "vencimento", "data"], how="inner")
m = m.sort_values(["tipo_titulo", "vencimento", "data"]).reset_index(drop=True)

if m.empty:
    st.warning("Não há interseção de datas entre Vendas e Preço/Taxa na janela selecionada.")
    st.stop()

# último ponto por série
last = (
    m.groupby(["tipo_titulo", "vencimento"], as_index=False)["data"]
     .max()
     .rename(columns={"data": "last_date"})
)

ml = m.merge(last, on=["tipo_titulo", "vencimento"], how="inner")
ml = ml[ml["data"] == ml["last_date"]].copy()
ml["ano_venc"] = pd.to_datetime(ml["vencimento"]).dt.year.astype(int)

# estatísticas por série
stats = (
    m.groupby(["tipo_titulo", "vencimento"], as_index=False)
     .agg(
         pu_min=("pu", "min"),
         pu_max=("pu", "max"),
         pu_p25=("pu", lambda s: np.nanpercentile(s, 25)),
         pu_p50=("pu", lambda s: np.nanpercentile(s, 50)),
         taxa_min=("taxa", "min"),
         taxa_max=("taxa", "max"),
         taxa_p50=("taxa", lambda s: np.nanpercentile(s, 50)),
         taxa_p75=("taxa", lambda s: np.nanpercentile(s, 75)),
     )
)

rank = ml.merge(stats, on=["tipo_titulo", "vencimento"], how="left")

# normalizações (0..1)
rank["taxa_01"] = (rank["taxa"] - rank["taxa_min"]) / (rank["taxa_max"] - rank["taxa_min"])
rank["taxa_01"] = rank["taxa_01"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

rank["pu_01"] = (rank["pu"] - rank["pu_min"]) / (rank["pu_max"] - rank["pu_min"])
rank["pu_01"] = rank["pu_01"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

rank["taxa_alta"] = rank["taxa_01"]      # 1 = taxa perto do máximo da janela
rank["pu_barato"] = 1.0 - rank["pu_01"]  # 1 = PU perto do mínimo da janela

# maturidade (anos até vencimento)
rank["anos_ate_venc"] = (pd.to_datetime(rank["vencimento"]) - pd.to_datetime(rank["last_date"])).dt.days / 365.25
rank["anos_ate_venc"] = rank["anos_ate_venc"].clip(lower=0)
maxy = max(rank["anos_ate_venc"].max(), 1e-9)
rank["maturity_01"] = (rank["anos_ate_venc"] / maxy).clip(0, 1)

# score final
rank["score"] = 0.45 * rank["taxa_alta"] + 0.35 * rank["pu_barato"] + 0.20 * rank["maturity_01"]
rank = rank.sort_values("score", ascending=False).reset_index(drop=True)

# ===== tabela topN + ID interno =====
rank_top = rank.head(top_n).copy().reset_index(drop=True)
rank_top["id"] = rank_top.index  # id estável dentro do top N

# colunas para exibição
rank_view = rank_top[[
    "id",
    "tipo_titulo",
    "ano_venc",
    "last_date",
    "taxa",
    "pu",
    "pu_p25",
    "pu_p50",
    "taxa_p50",
    "taxa_p75",
    "anos_ate_venc",
    "score",
]].copy()

rank_view = rank_view.rename(columns={
    "id": "ID",
    "tipo_titulo": "Tipo",
    "ano_venc": "Venc",
    "last_date": "Última data",
    "taxa": "Taxa (%)",
    "pu": "PU",
    "pu_p25": "PU P25",
    "pu_p50": "PU P50",
    "taxa_p50": "Taxa P50",
    "taxa_p75": "Taxa P75",
    "anos_ate_venc": "Anos até venc",
    "score": "Score",
})

# ===== seleção na tabela (single row) =====
# mantém seleção entre reruns
if "selected_rank_id" not in st.session_state:
    st.session_state.selected_rank_id = int(rank_view.loc[0, "ID"])

event = st.dataframe(
    rank_view,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# event.selection -> {"rows":[...]}
rows = event.selection.get("rows", []) if event is not None else []
if rows:
    # rows traz índice posicional do dataframe exibido
    selected_row_idx = rows[0]
    st.session_state.selected_rank_id = int(rank_view.iloc[selected_row_idx]["ID"])

# obtém o selecionado pelo ID
sel_row = rank_top[rank_top["id"] == st.session_state.selected_rank_id].iloc[0]
sel_tipo = sel_row["tipo_titulo"]
sel_venc = pd.to_datetime(sel_row["vencimento"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Selecionado", (sel_tipo[:18] + "…") if len(sel_tipo) > 18 else sel_tipo)
c2.metric("Vencimento", str(sel_venc.date()))
c3.metric("Taxa (última)", f"{sel_row['taxa']:.2f}%")
c4.metric("PU (último)", f"{sel_row['pu']:.2f}")
c5.metric("Score", f"{sel_row['score']:.3f}")

st.divider()

# =========================
# Gráfico sincronizado: selecionado na tabela
# =========================
st.subheader("📈 Série do título selecionado (PU, taxa e volume)")

ms = m[(m["tipo_titulo"] == sel_tipo) & (pd.to_datetime(m["vencimento"]) == sel_venc)].copy()
ms = ms.sort_values("data").reset_index(drop=True)

if ms.empty:
    st.warning("Não encontrei pontos em comum (vendas x precotaxa) para esse título na janela selecionada.")
    st.stop()

ms["pu_mm"] = ms["pu"].rolling(mm_window, min_periods=1).mean()

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    subplot_titles=("PU (linha) + Taxa (eixo direito)", "Volume (quantidade)")
)

fig.add_trace(go.Scatter(x=ms["data"], y=ms["pu"], mode="lines", name="PU"),
              row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=ms["data"], y=ms["pu_mm"], mode="lines", name=f"MM{mm_window}"),
              row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=ms["data"], y=ms["taxa"], mode="lines", name="Taxa (%)"),
              row=1, col=1, secondary_y=True)

fig.add_trace(go.Bar(x=ms["data"], y=ms["volume"], name="Volume", opacity=0.6),
              row=2, col=1)

fig.update_layout(
    height=680,
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(title_text="Data", type="date", row=2, col=1)
fig.update_yaxes(title_text="PU", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Taxa (%)", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Volume", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Downloads do selecionado
# =========================
st.subheader("⬇️ Exportar série selecionada")

export_df = ms[["data", "pu", "pu_mm", "taxa", "volume"]].copy().sort_values("data")

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Baixar CSV (selecionado)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"tesouro_{sel_tipo.replace(' ','_')}_{sel_venc.date()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with d2:
    buf = io.BytesIO()
    export_df.to_parquet(buf, index=False)
    st.download_button(
        "Baixar Parquet (selecionado)",
        data=buf.getvalue(),
        file_name=f"tesouro_{sel_tipo.replace(' ','_')}_{sel_venc.date()}.parquet",
        mime="application/octet-stream",
        use_container_width=True,
    )
