"""Gráficos Altair (payoff, histórico de resultado, smile de volatilidade)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# Paleta categórica validada (slots 1 e 2) e tons neutros para eixos/referências.
AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#8a8984"
GRADE = "#e6e5e1"


def _eixos(ch: alt.Chart) -> alt.Chart:
    return ch.configure_axis(gridColor=GRADE, gridWidth=1, domainColor=CINZA, tickColor=CINZA, labelFontSize=11,
                             titleFontSize=12, titleFontWeight="normal").configure_view(strokeWidth=0).configure_legend(
        orient="top", title=None, labelFontSize=12, symbolStrokeWidth=3)


def _crosshair(base: alt.Chart, x: str, campos_tooltip: list) -> list[alt.Chart]:
    sel = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False, clear="pointerout")
    seletor = base.mark_point(opacity=0, size=400).encode(x=f"{x}:Q" if x != "data" else f"{x}:T").add_params(sel)
    regra = base.mark_rule(color=CINZA, strokeWidth=1).encode(
        x=f"{x}:Q" if x != "data" else f"{x}:T", tooltip=campos_tooltip
    ).transform_filter(sel)
    return [seletor, regra], sel


def payoff(S: np.ndarray, venc: np.ndarray, hoje: np.ndarray | None, spot: float, titulo_y: str = "Resultado (R$)",
           breakevens: list[float] | None = None, altura: int = 340) -> alt.Chart:
    linhas = [pd.DataFrame({"S": S, "valor": venc, "curva": "No vencimento"})]
    if hoje is not None:
        linhas.append(pd.DataFrame({"S": S, "valor": hoje, "curva": "Hoje (modelo)"}))
    df = pd.concat(linhas, ignore_index=True)
    larga = df.pivot(index="S", columns="curva", values="valor").reset_index()
    larga["variacao"] = larga["S"] / spot - 1

    cores = alt.Scale(domain=["No vencimento", "Hoje (modelo)"], range=[AZUL, LARANJA])
    base = alt.Chart(df)
    linha = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
        x=alt.X("S:Q", title="Preço do ativo (R$)", scale=alt.Scale(zero=False)),
        y=alt.Y("valor:Q", title=titulo_y),
        color=alt.Color("curva:N", scale=cores),
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=CINZA, strokeWidth=1).encode(y="y:Q")
    ref = alt.Chart(pd.DataFrame({"S": [spot], "rot": [f"Atual {spot:,.2f}"]}))
    regra_spot = ref.mark_rule(color=CINZA, strokeDash=[4, 3], strokeWidth=1).encode(x="S:Q")
    rot_spot = ref.mark_text(align="left", dx=4, dy=-6, color=CINZA, fontSize=11).encode(x="S:Q", y=alt.value(8), text="rot:N")

    tooltip = [alt.Tooltip("S:Q", title="Ativo", format=",.2f"), alt.Tooltip("variacao:Q", title="Variação", format="+.1%"),
               alt.Tooltip("No vencimento:Q", format=",.2f")]
    if hoje is not None:
        tooltip.append(alt.Tooltip("Hoje (modelo):Q", format=",.2f"))
    camadas, _ = _crosshair(alt.Chart(larga), "S", tooltip)

    partes = [zero, linha, regra_spot, rot_spot, *camadas]
    if breakevens:
        be = pd.DataFrame({"S": [b for b in breakevens if np.isfinite(b)], "y": 0.0})
        if not be.empty:
            partes.append(alt.Chart(be).mark_point(filled=True, size=70, color=AZUL, stroke="white", strokeWidth=2).encode(
                x="S:Q", y="y:Q", tooltip=[alt.Tooltip("S:Q", title="Equilíbrio", format=",.2f")]))
    return _eixos(alt.layer(*partes).properties(height=altura))


def historico(df: pd.DataFrame, altura: int = 260) -> alt.Chart:
    base = alt.Chart(df)
    linha = base.mark_line(strokeWidth=2, color=AZUL, strokeCap="round").encode(
        x=alt.X("data:T", title=None), y=alt.Y("resultado:Q", title="Resultado (R$)"))
    area = base.mark_area(color=AZUL, opacity=0.10).encode(x="data:T", y="resultado:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=CINZA, strokeWidth=1).encode(y="y:Q")
    fim = base.transform_window(rank="rank()", sort=[alt.SortField("data", order="descending")]).transform_filter(
        alt.datum.rank == 1)
    ponto = fim.mark_point(filled=True, size=70, color=AZUL, stroke="white", strokeWidth=2).encode(x="data:T", y="resultado:Q")
    rotulo = fim.mark_text(align="left", dx=8, fontSize=11).encode(x="data:T", y="resultado:Q",
                                                                   text=alt.Text("resultado:Q", format=",.0f"))
    tooltip = [alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
               alt.Tooltip("resultado:Q", title="Resultado", format=",.2f"),
               alt.Tooltip("spot:Q", title="Ativo", format=",.2f")]
    camadas, _ = _crosshair(base, "data", tooltip)
    return _eixos(alt.layer(area, zero, linha, ponto, rotulo, *camadas).properties(height=altura))


def smile(df: pd.DataFrame, spot: float, altura: int = 300) -> alt.Chart:
    """IV ajustada (linha) e IV dos negócios do dia (pontos) por strike."""
    d = df.sort_values("strike")
    linha = pd.DataFrame({"strike": d["strike"], "iv": d["iv_modelo"], "serie": "Smile ajustado"}).drop_duplicates("strike")
    pts = d[d["iv_mercado"].notna()]
    pts = pd.DataFrame({"strike": pts["strike"], "iv": pts["iv_mercado"], "serie": "IV dos negócios",
                        "simbolo": pts["simbolo"], "negocios": pts["negocios"]})
    cores = alt.Scale(domain=["Smile ajustado", "IV dos negócios"], range=[AZUL, LARANJA])
    l = alt.Chart(linha).mark_line(strokeWidth=2).encode(
        x=alt.X("strike:Q", title="Strike", scale=alt.Scale(zero=False)),
        y=alt.Y("iv:Q", title="Volatilidade implícita", axis=alt.Axis(format=".0%")),
        color=alt.Color("serie:N", scale=cores))
    p = alt.Chart(pts).mark_point(filled=True, size=64, stroke="white", strokeWidth=2).encode(
        x="strike:Q", y="iv:Q", color=alt.Color("serie:N", scale=cores),
        tooltip=[alt.Tooltip("simbolo:N", title="Série"), alt.Tooltip("strike:Q", format=",.2f"),
                 alt.Tooltip("iv:Q", title="IV", format=".1%"), alt.Tooltip("negocios:Q", title="Negócios")])
    regra = alt.Chart(pd.DataFrame({"strike": [spot]})).mark_rule(color=CINZA, strokeDash=[4, 3]).encode(x="strike:Q")
    return _eixos(alt.layer(l, p, regra).properties(height=altura))
