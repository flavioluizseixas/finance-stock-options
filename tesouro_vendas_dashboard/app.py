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
# Helpers: indexador e “rende”
# =========================
def classify_indexer(tipo: str) -> str:
    t = (tipo or "").lower()
    if "selic" in t:
        return "SELIC"
    if "prefixado" in t:
        return "PREFIXADO"
    if "ipca" in t or "renda+" in t or "educa" in t:
        return "IPCA"
    return "OUTRO"


def format_rende(tipo: str, taxa: float) -> str:
    idx = classify_indexer(tipo)
    if pd.isna(taxa):
        return "—"
    if idx == "IPCA":
        return f"IPCA + {taxa:.2f}% a.a."
    if idx == "PREFIXADO":
        return f"Prefixado {taxa:.2f}% a.a."
    if idx == "SELIC":
        return f"Selic (taxa indicativa {taxa:.2f}% a.a.)"
    return f"Taxa {taxa:.2f}% a.a."


# =========================
# Sidebar
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
# Load parquets
# =========================
vendas = load_parquet(PARQUET_VENDAS)
pt = load_parquet(PARQUET_PRECOTAXA)

if vendas.empty or pt.empty:
    st.warning("Parquets vazios/inexistentes. Clique em **Atualizar** na barra lateral.")
    st.stop()

vendas["data_venda"] = pd.to_datetime(vendas["data_venda"], errors="coerce")
vendas["vencimento"] = pd.to_datetime(vendas["vencimento"], errors="coerce")

pt["data_base"] = pd.to_datetime(pt["data_base"], errors="coerce")
pt["vencimento"] = pd.to_datetime(pt["vencimento"], errors="coerce")

vendas = vendas.dropna(subset=["data_venda", "tipo_titulo", "vencimento", "pu_venda", "quantidade"])
pt = pt.dropna(subset=["data_base", "tipo_titulo", "vencimento", "taxa_compra", "taxa_venda", "pu_compra", "pu_venda"])

ultima_vendas = vendas["data_venda"].max()
ultima_pt = pt["data_base"].max()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Última data (Vendas)", str(ultima_vendas.date()))
k2.metric("Última data (Preço/Taxa)", str(ultima_pt.date()))
k3.metric("Linhas Vendas", f"{len(vendas):,}".replace(",", "."))
k4.metric("Linhas Preço/Taxa", f"{len(pt):,}".replace(",", "."))

st.divider()


# =========================
# Base histórica para “barato/caro” (PU de vendas) + volume
# =========================
st.subheader("📌 Scanner de oportunidade (queda de juros → PU tende a subir)")
st.caption(
    "Agora o ranking usa **taxa/PU vigentes** do `precotaxa` (Data Base mais recente), "
    "e usa o histórico de vendas apenas para contexto (PU barato/caro e volume)."
)

# Janela histórica para vendas (contexto)
end_date = min(ultima_vendas, ultima_pt)
start_date = (end_date - pd.DateOffset(years=lookback_years)).floor("D")

v_daily = vendas.copy()
v_daily["data"] = v_daily["data_venda"].dt.floor("D")
v_daily = v_daily[v_daily["data"] >= start_date]
v_daily = (
    v_daily.groupby(["tipo_titulo", "vencimento", "data"], as_index=False)
           .agg(pu=("pu_venda", "mean"), volume=("quantidade", "sum"))
)

# Estatísticas históricas de PU (vendas) por papel
hist_stats = (
    v_daily.groupby(["tipo_titulo", "vencimento"], as_index=False)
           .agg(
               pu_min=("pu", "min"),
               pu_max=("pu", "max"),
               pu_p25=("pu", lambda s: np.nanpercentile(s, 25)),
               pu_p50=("pu", lambda s: np.nanpercentile(s, 50)),
               vol_med=("volume", "median"),
           )
)

# =========================
# Snapshot "vigente" do precotaxa: última data_base por papel
# (CORREÇÃO PRINCIPAL: NÃO depende de merge com vendas)
# =========================
pt_sorted = pt.sort_values(["tipo_titulo", "vencimento", "data_base"]).copy()
pt_last = (
    pt_sorted.groupby(["tipo_titulo", "vencimento"], as_index=False)
             .tail(1)
             .rename(columns={"data_base": "last_date"})
)

pt_last["ano_venc"] = pd.to_datetime(pt_last["vencimento"]).dt.year.astype(int)
pt_last["spread_bps"] = (pt_last["taxa_venda"] - pt_last["taxa_compra"]) * 100.0

pt_last["indexador"] = pt_last["tipo_titulo"].apply(classify_indexer)
pt_last["rende_venda"] = [format_rende(t, x) for t, x in zip(pt_last["tipo_titulo"], pt_last["taxa_venda"])]
pt_last["rende_compra"] = [format_rende(t, x) for t, x in zip(pt_last["tipo_titulo"], pt_last["taxa_compra"])]

# Junta com stats históricas (pode faltar histórico de vendas para alguns papéis)
rank = pt_last.merge(hist_stats, on=["tipo_titulo", "vencimento"], how="left")

# Se não tiver histórico, evita NaNs explosivos
rank["pu_min"] = rank["pu_min"].fillna(rank["pu_venda"])
rank["pu_max"] = rank["pu_max"].fillna(rank["pu_venda"])
rank["pu_p25"] = rank["pu_p25"].fillna(rank["pu_venda"])
rank["pu_p50"] = rank["pu_p50"].fillna(rank["pu_venda"])
rank["vol_med"] = rank["vol_med"].fillna(0)

# “PU barato” relativo ao histórico de vendas (quando existir)
den = (rank["pu_max"] - rank["pu_min"]).replace(0, np.nan)
rank["pu_01"] = ((rank["pu_venda"] - rank["pu_min"]) / den).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
rank["pu_barato"] = 1.0 - rank["pu_01"]

# “taxa alta” relativo ao próprio histórico de taxa (precotaxa)
# (usa janela lookback_years também)
pt_window = pt_sorted[pt_sorted["data_base"] >= start_date].copy()
taxa_stats = (
    pt_window.groupby(["tipo_titulo", "vencimento"], as_index=False)
             .agg(
                 taxa_min=("taxa_venda", "min"),
                 taxa_max=("taxa_venda", "max"),
                 taxa_p50=("taxa_venda", lambda s: np.nanpercentile(s, 50)),
                 taxa_p75=("taxa_venda", lambda s: np.nanpercentile(s, 75)),
             )
)
rank = rank.merge(taxa_stats, on=["tipo_titulo", "vencimento"], how="left")

den_t = (rank["taxa_max"] - rank["taxa_min"]).replace(0, np.nan)
rank["taxa_01"] = ((rank["taxa_venda"] - rank["taxa_min"]) / den_t).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
rank["taxa_alta"] = rank["taxa_01"]

# Maturidade (anos até vencimento), usando last_date do precotaxa
rank["anos_ate_venc"] = (pd.to_datetime(rank["vencimento"]) - pd.to_datetime(rank["last_date"])).dt.days / 365.25
rank["anos_ate_venc"] = rank["anos_ate_venc"].clip(lower=0)
maxy = max(rank["anos_ate_venc"].max(), 1e-9)
rank["maturity_01"] = (rank["anos_ate_venc"] / maxy).clip(0, 1)

# Score final (taxa alta + PU barato + maturidade)
rank["score"] = 0.45 * rank["taxa_alta"] + 0.35 * rank["pu_barato"] + 0.20 * rank["maturity_01"]
rank = rank.sort_values("score", ascending=False).reset_index(drop=True)

# Top N
rank_top = rank.head(top_n).copy().reset_index(drop=True)
rank_top["id"] = rank_top.index

rank_view = rank_top[[
    "id",
    "tipo_titulo",
    "indexador",
    "ano_venc",
    "last_date",
    "rende_venda",
    "rende_compra",
    "taxa_compra",
    "taxa_venda",
    "spread_bps",
    "pu_compra",
    "pu_venda",
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
    "indexador": "Indexador",
    "ano_venc": "Venc",
    "last_date": "Data base (vigente)",
    "rende_venda": "Rende (Venda)",
    "rende_compra": "Rende (Compra)",
    "taxa_compra": "Taxa Compra",
    "taxa_venda": "Taxa Venda",
    "spread_bps": "Spread (bps)",
    "pu_compra": "PU Compra",
    "pu_venda": "PU Venda",
    "pu_p25": "PU P25 (hist vendas)",
    "pu_p50": "PU P50 (hist vendas)",
    "taxa_p50": "Taxa P50 (hist PT)",
    "taxa_p75": "Taxa P75 (hist PT)",
    "anos_ate_venc": "Anos até venc",
    "score": "Score",
})

# =========================
# Seleção por clique na tabela
# =========================
if "selected_rank_id" not in st.session_state:
    st.session_state.selected_rank_id = int(rank_view.loc[0, "ID"])

event = st.dataframe(
    rank_view,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

rows = event.selection.get("rows", []) if event is not None else []
if rows:
    st.session_state.selected_rank_id = int(rank_view.iloc[rows[0]]["ID"])

sel_row = rank_top[rank_top["id"] == st.session_state.selected_rank_id].iloc[0]
sel_tipo = sel_row["tipo_titulo"]
sel_venc = pd.to_datetime(sel_row["vencimento"])

# Cards
c1, c2, c3, c4, c5, c6 = st.columns([1.7, 1.2, 1.0, 1.1, 1.0, 2.0])
c1.metric("Selecionado", (sel_tipo[:24] + "…") if len(sel_tipo) > 24 else sel_tipo)
c2.metric("Vencimento", str(sel_venc.date()))
c3.metric("Data base", str(pd.to_datetime(sel_row["last_date"]).date()))
c4.metric("Taxa Venda", f"{sel_row['taxa_venda']:.2f}%")
c5.metric("PU Venda", f"{sel_row['pu_venda']:.2f}")
c6.metric("Rende (Venda)", sel_row["rende_venda"])

c7, c8, c9, c10 = st.columns(4)
c7.metric("Taxa Compra", f"{sel_row['taxa_compra']:.2f}%")
c8.metric("PU Compra", f"{sel_row['pu_compra']:.2f}")
c9.metric("Spread", f"{sel_row['spread_bps']:.0f} bps")
c10.metric("Score", f"{sel_row['score']:.3f}")

st.divider()

# =========================
# Série para gráfico:
# PU/volume de vendas (histórico) + taxa_venda do precotaxa (histórico)
# =========================
st.subheader("📈 Série do título selecionado (PU precotaxa + PU vendas + Taxa Venda + Volume)")

# Vendas do papel (diário)
vs = v_daily[(v_daily["tipo_titulo"] == sel_tipo) & (pd.to_datetime(v_daily["vencimento"]) == sel_venc)].copy()
vs = vs.sort_values("data").reset_index(drop=True)
vs["pu_mm"] = vs["pu"].rolling(mm_window, min_periods=1).mean()

# Preço/taxa do papel (histórico, diários)
pt_sel = pt_window[(pt_window["tipo_titulo"] == sel_tipo) & (pd.to_datetime(pt_window["vencimento"]) == sel_venc)].copy()
pt_sel = pt_sel.sort_values("data_base").reset_index(drop=True)
pt_sel["data"] = pt_sel["data_base"].dt.floor("D")

# Agrega precotaxa por dia (às vezes pode ter duplicatas por motivos de arquivo)
pt_day = (
    pt_sel.groupby("data", as_index=False)
          .agg(
              taxa_venda=("taxa_venda", "mean"),
              taxa_compra=("taxa_compra", "mean"),
              pu_venda_pt=("pu_venda", "mean"),
              pu_compra_pt=("pu_compra", "mean"),
          )
          .sort_values("data")
)

# Merge por data para plot (outer para não perder dias)
plot_df = pd.merge(
    pt_day,
    vs[["data", "pu", "pu_mm", "volume"]],
    on="data",
    how="outer",
).sort_values("data")

# Média móvel do PU do precotaxa (contínuo)
plot_df["pu_venda_pt_mm"] = plot_df["pu_venda_pt"].rolling(mm_window, min_periods=1).mean()

# Se não houver nada, aborta
if plot_df[["pu_venda_pt", "taxa_venda"]].notna().sum().sum() == 0:
    st.warning("Não há pontos suficientes para plotar esse papel na janela selecionada.")
    st.stop()

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.72, 0.28],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    subplot_titles=(
        "PU (precotaxa) + Taxa Venda (eixo direito)  —  PU de vendas como pontos",
        "Volume (vendas)"
    )
)

# 1) PU do precotaxa (linha principal contínua)
fig.add_trace(
    go.Scatter(
        x=plot_df["data"], y=plot_df["pu_venda_pt"],
        mode="lines", name="PU (precotaxa)",
        connectgaps=True
    ),
    row=1, col=1, secondary_y=False
)

# 2) MM do PU precotaxa
fig.add_trace(
    go.Scatter(
        x=plot_df["data"], y=plot_df["pu_venda_pt_mm"],
        mode="lines", name=f"MM{mm_window} (PU precotaxa)",
        connectgaps=True
    ),
    row=1, col=1, secondary_y=False
)

# 3) PU de vendas (somente quando existe venda) como pontos
# (isso evita “sumir” e evita linha quebrada/enganosa)
fig.add_trace(
    go.Scatter(
        x=plot_df["data"], y=plot_df["pu"],
        mode="markers", name="PU (vendas)",
        marker=dict(size=5),
    ),
    row=1, col=1, secondary_y=False
)

# 4) Taxa venda (eixo direito)
fig.add_trace(
    go.Scatter(
        x=plot_df["data"], y=plot_df["taxa_venda"],
        mode="lines", name="Taxa Venda (%)",
        connectgaps=True
    ),
    row=1, col=1, secondary_y=True
)

# 5) Volume (barras) — dias sem venda ficam NaN -> vira 0
vol = plot_df["volume"].fillna(0)
fig.add_trace(
    go.Bar(x=plot_df["data"], y=vol, name="Volume", opacity=0.5),
    row=2, col=1
)

fig.update_layout(
    height=720,
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

fig.update_xaxes(title_text="Data", type="date", row=2, col=1)
fig.update_yaxes(title_text="PU", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Taxa (%)", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Volume", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Export
# =========================
st.subheader("⬇️ Exportar série selecionada")

export_df = plot_df.copy()
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
