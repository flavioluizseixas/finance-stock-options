"""Painel de Opções — estratégias, registro e acompanhamento de operações (execução local).

    streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Painel de Opções", page_icon="📈", layout="wide")

from opcoes import config, dados, estrategias, graficos, operacoes  # noqa: E402
from opcoes.formato import brl, num, pct  # noqa: E402

LIQ = config.LIQ_ROTULOS
CRITERIOS = {
    "Vantagem ÷ risco (recomendado)": "indice",
    "Vantagem esperada sobre o capital (anualizada)": "vantagem_aa",
    "Probabilidade de lucro": "pop",
    "Retorno máximo sobre o capital": "retorno_max",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _ler(caminho: str, versao: float) -> pd.DataFrame:
    return dados.ler(Path(caminho))


def ler(path: Path) -> pd.DataFrame:
    return _ler(str(path), dados.mtime(path))


@st.cache_data(show_spinner=False)
def _indicadores(versao: float) -> pd.DataFrame:
    p = ler(config.ARQ_PRECOS)
    return dados.indicadores(p) if not p.empty else pd.DataFrame()


@st.cache_data(show_spinner="Avaliando todas as estruturas com opções líquidas…")
def _ranquear(v_cad: float, v_pr: float, ativos: tuple, du_min: int, du_max: int, liq: int, chaves: tuple) -> pd.DataFrame:
    return estrategias.ranquear(ler(config.ARQ_CADEIA), ler(config.ARQ_PRECOS), list(ativos), du_min, du_max, liq, list(chaves))


def pernas_df_para_avaliacao(pernas: list[dict], du: int, r: float) -> pd.DataFrame:
    """Converte pernas sugeridas no formato de `operacoes.avaliar_pernas` (para payoff hoje/vencimento)."""
    df = pd.DataFrame(pernas)
    df["sinal"] = np.where(df["lado"] == "COMPRA", 1.0, -1.0)
    df["strike_atual"] = df["strike"]
    df["preco_entrada"] = df["preco"]
    df["aberta"] = True
    df["du"] = du
    df["taxa_r"] = r
    return df


def executar_atualizacao(somente_operacoes: bool = False) -> None:
    barra = st.progress(0.0, text="Iniciando…")
    try:
        res = dados.atualizar_mercado(progresso=lambda f, m: barra.progress(f, text=m), somente_operacoes=somente_operacoes)
    except Exception as exc:  # noqa: BLE001
        barra.empty()
        st.error(f"Falha na atualização: {exc}")
        return
    barra.empty()
    st.cache_data.clear()
    msg = f"Cotações atualizadas — pregão {res.get('sessao', '?')}, {len(res['ativos'])} ativos."
    if res.get("erros"):
        msg += " Avisos: " + "; ".join(f"{k}: {v}" for k, v in res["erros"].items())
    st.session_state["aviso"] = ("warning" if res.get("erros") else "success", msg)
    st.rerun()


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

meta = dados.meta()
cadeia = ler(config.ARQ_CADEIA)
precos = ler(config.ARQ_PRECOS)
ind = _indicadores(dados.mtime(config.ARQ_PRECOS))

with st.sidebar:
    st.title("📈 Painel de Opções")
    if "aviso" in st.session_state:
        tipo_aviso, texto_aviso = st.session_state.pop("aviso")
        (st.warning if tipo_aviso == "warning" else st.success)(texto_aviso)
    if meta.get("sessao"):
        st.markdown(
            f"**Pregão:** {pd.Timestamp(meta['sessao']):%d/%m/%Y}  \n"
            f"**Atualizado:** {pd.Timestamp(meta['atualizado_em']):%d/%m %H:%M}  \n"
            f"**Selic:** {pct(meta.get('taxa_juros', np.nan), 2)} a.a.  \n"
            f"**opcoes.net.br:** {'conta conectada' if meta.get('login') else 'sem login'}"
        )
    else:
        st.info("Ainda não há cotações. Clique em **Atualizar cotações**.")
    acompanhados = dados.ativos_acompanhados()
    st.caption(f"{len(acompanhados)} ativos acompanhados (aba Mercado) + ativos com operações abertas. "
               "O opcoes.net.br limita requisições: ~5–10 s por ativo; se o site bloquear, o painel salva o que "
               "já coletou e avisa.")
    if st.button("🔄 Atualizar cotações", type="primary", width="stretch"):
        executar_atualizacao()
    if st.button("⚡ Só ativos com operações abertas", width="stretch"):
        executar_atualizacao(somente_operacoes=True)
    lote = st.number_input("Lote padrão (opções)", min_value=100, step=100, value=100,
                           help="Quantidade usada para mostrar valores das estratégias (B3: lote de 100).")
    with st.expander("Custos considerados"):
        st.write(f"Emolumentos/registro: {num(config.CUSTO_PCT, 3)}% do volume  \n"
                 f"Corretagem: {brl(config.CORRETAGEM_ORDEM)} por ordem  \n"
                 f"Meio spread estimado: {brl(config.CUSTO_TICK)} por opção")
        st.caption("Ajuste em `.env` (CUSTO_PCT, CORRETAGEM_ORDEM, CUSTO_TICK).")

aba_est, aba_ops, aba_mkt = st.tabs(["🎯 Estratégias", "📒 Operações", "🌐 Mercado"])


# ---------------------------------------------------------------------------
# Aba Estratégias
# ---------------------------------------------------------------------------

def form_registro(pernas: list[dict], ativo: str, estrategia: str, spot: float, chave: str) -> None:
    with st.form(f"registrar_{chave}"):
        st.markdown("**Registrar esta operação**")
        c1, c2, c3 = st.columns([1, 1, 2])
        modo = c1.radio("Tipo", ["ESTUDO", "REAL"], horizontal=True,
                        help="ESTUDO acompanha uma simulação; REAL entra nos resultados da sua carteira.")
        data_ent = c2.date_input("Data de entrada", value=date.today(), format="DD/MM/YYYY")
        notas = c3.text_input("Notas / tese")
        tab = pd.DataFrame(pernas)[["lado", "tipo", "simbolo", "strike", "quantidade", "preco"]]
        ed = st.data_editor(
            tab, hide_index=True, width="stretch", key=f"ed_{chave}", disabled=["lado", "tipo", "simbolo", "strike"],
            column_config={
                "lado": "Lado", "tipo": "Tipo", "simbolo": "Série",
                "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "quantidade": st.column_config.NumberColumn("Quantidade", min_value=1, step=100),
                "preco": st.column_config.NumberColumn("Preço de entrada", format="%.2f", min_value=0.0,
                                                       help="Edite com o preço efetivamente executado."),
            },
        )
        pernas_final = [{**p, "quantidade": int(ed.loc[i, "quantidade"]), "preco": float(ed.loc[i, "preco"])}
                        for i, p in enumerate(pernas)]
        custos = st.number_input("Custos de entrada (R$)", min_value=0.0, step=1.0,
                                 value=round(operacoes.custo_estimado(pernas_final), 2))
        if st.form_submit_button("💾 Registrar", type="primary"):
            op_id = operacoes.criar(modo, ativo, estrategia, data_ent, pernas_final, custos, spot, notas)
            simb = {p["simbolo"] for p in pernas_final if p["tipo"] != "ACAO"}
            dados.atualizar_historico_pernas(simb, cadeia, buscar_site=False)
            st.cache_data.clear()
            st.success(f"Operação #{op_id} registrada como {modo}. Acompanhe na aba **Operações**.")


def detalhe_candidato(row: pd.Series) -> None:
    est = estrategias.ESTRATEGIAS[row["chave"]]
    st.subheader(f"{est.nome} · {row['ativo']} · vence {pd.Timestamp(row['vencimento']):%d/%m/%Y} ({int(row['du'])} pregões)")
    st.caption(est.descricao)
    k = lote
    c = st.columns(6)
    c[0].metric("Prêmio (crédito + / débito −)", brl(row["premio"] * k, True),
                help="Pelo preço justo do smile. Pelo último negócio: " + brl(row["premio_ultimo"] * k, True))
    c[1].metric("Capital/risco alocado", brl(row["capital"] * k))
    c[2].metric("Lucro máximo", brl(row["max_lucro"] * k))
    c[3].metric("Perda máxima", brl(row["max_perda"] * k))
    c[4].metric("Prob. de lucro", pct(row["pop"], 0))
    c[5].metric("Vantagem esperada", brl(row["vantagem"] * k, True), pct(row["vantagem_pct"], 1, True) + " do capital",
                help="Valor esperado acima do CDI até o vencimento, na distribuição histórica do ativo escalada pela "
                     f"vol atual ({pct(row['vantagem_aa'], 0, True)} se anualizado).")
    st.caption(
        f"CVaR 5% (média dos 5% piores cenários): {brl(row['cvar5'] * k)} · Δ {num(row['delta'] * k, 0)} ações · "
        f"Θ {brl(row['theta'] * k, True)}/dia · "
        f"equilíbrio: {', '.join(num(x) for x in (row['be1'], row['be2']) if np.isfinite(x)) or '—'}"
    )
    vm, iv = row.get("vol_modelo", np.nan), row["iv_atm"]
    if np.isfinite(vm):
        sinal = "acima" if vm > iv else "abaixo"
        st.caption(
            f"🔎 **O que move a vantagem:** vol. de referência do modelo {pct(vm)} a.a. (histórico recente do ativo) "
            f"está {sinal} da IV ATM de mercado {pct(iv)}. Vol. do modelo maior favorece **comprar** opções; menor favorece "
            "**vender**. Se você espera que a volatilidade normalize (ex.: após um evento pontual), desconte essa vantagem."
        )
    pernas = estrategias.pernas_da_linha(row, k)
    r = float(cadeia["taxa_r"].iloc[0])
    S0 = float(row["spot"])
    strikes = [x for x in (row.get(f"k{j}") for j in range(4)) if pd.notna(x)]
    lo, hi = min([S0 * 0.75] + [x * 0.92 for x in strikes]), max([S0 * 1.25] + [x * 1.08 for x in strikes])
    S = np.linspace(lo, hi, 300)
    # payoff com o preço justo (mesma base das métricas do ranking)
    av_justo = pernas_df_para_avaliacao(estrategias.pernas_da_linha(row, k, preco="justo"), int(row["du"]), r)
    venc = operacoes.resultado_vencimento(av_justo, S) - row["custo"] * k
    hoje = operacoes.resultado_hoje_curva(av_justo, S) - row["custo"] * k
    st.altair_chart(graficos.payoff(S, venc, hoje, S0, breakevens=[row["be1"], row["be2"]]), width="stretch")
    form_registro(pernas, row["ativo"], est.nome, S0, f"{row.name}")


with aba_est:
    if cadeia.empty:
        st.info("Atualize as cotações para ver estratégias.")
    else:
        ativos_disp = sorted(cadeia["ativo"].unique())
        f1, f2, f3, f4 = st.columns([3, 1.2, 2, 1.2])
        ativos_sel = f1.multiselect("Ativos", ativos_disp, default=ativos_disp)
        visao = f2.selectbox("Visão de mercado", estrategias.VISOES)
        du_min, du_max = f3.slider("Prazo até o vencimento (pregões)", 1, 250, (5, 45))
        liq_txt = f4.selectbox("Liquidez mínima por perna", ["Média", "Alta"],
                               help="Alta: negociada no último pregão com volume relevante ou formador de mercado. "
                                    "Média: negociada nos últimos pregões com pelo menos 5 negócios.")
        liq_min = 3 if liq_txt == "Alta" else 2
        chaves_visao = [e.chave for e in estrategias.ESTRATEGIAS.values() if visao == "Todas" or visao in e.visoes]
        g1, g2, g3, g4 = st.columns([3, 1.2, 2, 1.2])
        chaves = g1.multiselect("Estratégias", chaves_visao, default=chaves_visao,
                                format_func=lambda c: estrategias.ESTRATEGIAS[c].nome)
        pop_min = g2.slider("Prob. de lucro mínima", 0, 95, 0, step=5, format="%d%%") / 100
        criterio = g3.selectbox("Ordenar por", list(CRITERIOS))
        por_grupo = g4.number_input("Máx. por ativo/estratégia", 1, 50, 3)
        perda_lim = st.number_input("Perda máxima aceitável por lote (R$, 0 = sem limite)", min_value=0.0, step=100.0, value=0.0)

        if not ativos_sel or not chaves:
            st.warning("Selecione pelo menos um ativo e uma estratégia.")
        else:
            res = _ranquear(dados.mtime(config.ARQ_CADEIA), dados.mtime(config.ARQ_PRECOS), tuple(ativos_sel),
                            du_min, du_max, liq_min, tuple(sorted(chaves)))
            if res.empty:
                st.info("Nenhuma estrutura com séries líquidas nesses filtros.")
            else:
                f = res[res["pop"] >= pop_min]
                if perda_lim > 0:
                    f = f[-f["max_perda"] * lote <= perda_lim]
                col = CRITERIOS[criterio]
                f = f.assign(_o=f[col].replace(np.inf, np.nan)).sort_values("_o", ascending=False)
                f = f.groupby(["ativo", "chave"]).head(int(por_grupo)).head(150)
                st.caption(
                    f"{len(res):,} estruturas avaliadas · {len(f)} exibidas. Preços pelo **smile ajustado aos negócios do dia** "
                    "(o último negócio de cada série ocorre em horários diferentes). Vantagem = valor esperado acima do CDI "
                    "na distribuição histórica do ativo, já descontados custos e meio spread."
                )
                tab = pd.DataFrame(
                    {
                        "Ativo": f["ativo"],
                        "Estratégia": f["estrategia"],
                        "Pernas": f["descricao_pernas"],
                        "Vencimento": pd.to_datetime(f["vencimento"]).dt.date,
                        "Pregões": f["du"],
                        "Prêmio": f["premio"] * lote,
                        "Capital": f["capital"] * lote,
                        "Lucro máx.": f["max_lucro"].replace(np.inf, np.nan) * lote,
                        "Perda máx.": f["max_perda"] * lote,
                        "Prob. lucro": f["pop"],
                        "Vantagem": f["vantagem_pct"],
                        "Vantagem÷risco": f["indice"],
                        "Delta (ações)": f["delta"] * lote,
                        "Liquidez": f["liquidez_min"].map(LIQ),
                    }
                )
                ev = st.dataframe(
                    tab, hide_index=True, width="stretch", height=420, on_select="rerun", selection_mode="single-row",
                    key="tab_estrategias",
                    column_config={
                        "Pernas": st.column_config.TextColumn(width="large"),
                        "Vencimento": st.column_config.DateColumn(format="DD/MM/YY"),
                        "Prêmio": st.column_config.NumberColumn(format="localized", help="Crédito (+) ou débito (−) por lote, R$"),
                        "Capital": st.column_config.NumberColumn(format="localized", help="Capital ou margem de risco, R$"),
                        "Lucro máx.": st.column_config.NumberColumn(format="localized", help="Vazio = ilimitado"),
                        "Perda máx.": st.column_config.NumberColumn(format="localized"),
                        "Prob. lucro": st.column_config.NumberColumn(format="percent"),
                        "Vantagem": st.column_config.NumberColumn(
                            format="percent", help="Valor esperado acima do CDI até o vencimento, em % do capital"),
                        "Vantagem÷risco": st.column_config.NumberColumn(format="%.2f",
                                                                        help="Vantagem esperada ÷ desvio-padrão do resultado, anualizado"),
                        "Delta (ações)": st.column_config.NumberColumn(format="%.0f"),
                    },
                )
                if ev.selection.rows:
                    detalhe_candidato(f.iloc[ev.selection.rows[0]])
                else:
                    st.info("Selecione uma linha para ver o payoff e registrar a operação (estudo ou real).")


# ---------------------------------------------------------------------------
# Aba Operações
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _avaliar_tudo(v_ops: float, v_pernas: float, v_cad: float, v_pr: float, v_hist: float):
    ops, pernas = operacoes.carregar()
    if ops.empty:
        return ops, pd.DataFrame(), {}
    av = operacoes.avaliar_pernas(pernas, cadeia, precos, ler(config.ARQ_HIST_PERNAS))
    resumos = {}
    for op in ops.itertuples(index=False):
        resumos[op.op_id] = operacoes.resumo(pd.Series(op._asdict()), av[av["op_id"] == op.op_id])
    return ops, av, resumos


def nova_operacao_manual() -> None:
    with st.expander("➕ Nova operação manual (montar pernas)", expanded=False):
        if cadeia.empty:
            st.info("Atualize as cotações primeiro.")
            return
        rascunho = st.session_state.setdefault("rascunho", [])
        c1, c2, c3 = st.columns(3)
        ativo = c1.selectbox("Ativo", sorted(cadeia["ativo"].unique()), key="man_ativo")
        ch = cadeia[cadeia["ativo"] == ativo]
        spot = float(ch["spot"].iloc[0])
        tipo = c2.selectbox("Instrumento", ["CALL", "PUT", "ACAO"], key="man_tipo")
        lado = c3.selectbox("Lado", ["COMPRA", "VENDA"], key="man_lado")
        d1, d2, d3 = st.columns(3)
        if tipo == "ACAO":
            simbolo, strike, venc, preco_sug = ativo, np.nan, pd.NaT, spot
        else:
            vencs = sorted(ch.loc[ch["tipo"] == tipo, "vencimento"].unique())
            venc = d1.selectbox("Vencimento", vencs, format_func=lambda v: f"{pd.Timestamp(v):%d/%m/%Y}", key="man_venc")
            series = ch[(ch["tipo"] == tipo) & (ch["vencimento"] == venc)].sort_values("strike")
            perto = int((series["strike"] - spot).abs().reset_index(drop=True).idxmin()) if not series.empty else 0
            simbolo = d2.selectbox(
                "Série", series["simbolo"].tolist(), index=perto, key="man_serie",
                format_func=lambda s: (lambda r: f"{s} · K {num(r['strike'])} · {num(r['preco_ref'])} · liq. {LIQ[int(r['liquidez'])]}")(
                    series.set_index("simbolo").loc[s]),
            )
            linha = series.set_index("simbolo").loc[simbolo]
            strike, preco_sug = float(linha["strike"]), float(linha["preco_ref"])
        qtd = d3.number_input("Quantidade", min_value=1, step=100, value=100, key="man_qtd")
        preco = st.number_input("Preço de entrada", min_value=0.0, value=round(preco_sug, 2), step=0.01, key=f"man_preco_{simbolo}")
        if st.button("Adicionar perna"):
            rascunho.append({"tipo": tipo, "simbolo": simbolo, "lado": lado, "quantidade": int(qtd), "strike": strike,
                             "vencimento": venc, "preco": float(preco)})
        if rascunho:
            st.dataframe(pd.DataFrame(rascunho), hide_index=True, width="stretch")
            e1, e2, e3, e4 = st.columns(4)
            modo = e1.radio("Tipo", ["ESTUDO", "REAL"], horizontal=True, key="man_modo")
            nome = e2.text_input("Estratégia", value="Personalizada", key="man_nome")
            data_ent = e3.date_input("Data de entrada", value=date.today(), format="DD/MM/YYYY", key="man_data")
            custos = e4.number_input("Custos (R$)", min_value=0.0, value=round(operacoes.custo_estimado(rascunho), 2), key="man_custos")
            notas = st.text_input("Notas", key="man_notas")
            b1, b2 = st.columns(2)
            if b1.button("💾 Salvar operação", type="primary"):
                op_id = operacoes.criar(modo, ativo, nome, data_ent, rascunho, custos, spot, notas)
                simb = {p["simbolo"] for p in rascunho if p["tipo"] != "ACAO"}
                dados.atualizar_historico_pernas(simb, cadeia, buscar_site=data_ent < date.today())
                st.session_state["rascunho"] = []
                st.cache_data.clear()
                st.success(f"Operação #{op_id} salva.")
                st.rerun()
            if b2.button("Limpar pernas"):
                st.session_state["rascunho"] = []
                st.rerun()


def detalhe_operacao(op: pd.Series, av: pd.DataFrame, rs: dict) -> None:
    st.subheader(f"#{int(op['op_id'])} · {op['estrategia']} · {op['ativo']}  ·  {op['modo']} · {op['status']}")
    if op["notas"]:
        st.caption(op["notas"])
    abertas = av[av["aberta"]]
    spot = rs["spot"]

    if not abertas.empty and np.isfinite(spot):
        S_venc = st.slider("Preço do ativo no vencimento (simulação)", float(round(spot * 0.6, 2)), float(round(spot * 1.4, 2)),
                           float(round(spot, 2)), step=0.01, key=f"sv_{op['op_id']}")
        rs = operacoes.resumo(op, av, S_venc)
    c = st.columns(5)
    c[0].metric("Desfazer hoje: resultado", brl(rs["resultado_hoje"], True), pct(rs["resultado_hoje"] / rs["base_capital"], 1, True),
                help="Realizado + marcação das pernas abertas − custos já pagos − custos estimados para zerar + proventos.")
    c[1].metric("Caixa para desfazer hoje", brl(rs["caixa_desfazer"], True),
                help="Positivo: você recebe ao zerar tudo hoje. Negativo: você paga.")
    if not abertas.empty:
        c[2].metric(f"No vencimento (ativo a {num(rs['S_venc'])})", brl(rs["resultado_venc"], True),
                    pct(rs["resultado_venc"] / rs["base_capital"], 1, True))
        c[3].metric("Pregões até o vencimento", f"{rs['du']}" if rs["du"] else "—",
                    f"{pd.Timestamp(rs['vencimento']):%d/%m/%Y}" if pd.notna(rs["vencimento"]) else None, delta_color="off")
        c[4].metric("Delta · Theta/dia", f"{num(rs['delta'], 0)} · {brl(rs['theta'], True)}")
    else:
        c[2].metric("Resultado realizado", brl(rs["realizado"] - rs["custos"] + rs["proventos"], True))
    if rs["teorico"]:
        st.caption("⚠️ Alguma perna não negociou no último pregão: preço atual pelo smile ajustado (preço justo estimado).")

    tab = pd.DataFrame(
        {
            "Série": av["simbolo"], "Tipo": av["tipo"], "Lado": av["lado"], "Qtd": av["quantidade"].astype(int),
            "Strike entrada": av["strike"], "Strike atual": av["strike_atual"],
            "Vencimento": pd.to_datetime(av["vencimento"]).dt.date,
            "Entrada": av["preco_entrada"], "Atual/saída": np.where(av["aberta"], av["preco_atual"], av["preco_saida"]),
            "Fonte": np.where(av["aberta"], av["fonte"], "encerrada " + pd.to_datetime(av["data_saida"]).dt.strftime("%d/%m/%y")),
            "Resultado": av["resultado"], "Caixa p/ desfazer": av["caixa_desfazer"],
            "Liquidez": av["liquidez"].map(lambda x: LIQ.get(int(x), "—") if pd.notna(x) else "—"),
        }
    )
    st.dataframe(tab, hide_index=True, width="stretch", column_config={
        "Strike entrada": st.column_config.NumberColumn(format="%.2f"),
        "Strike atual": st.column_config.NumberColumn(format="%.2f", help="Strikes da B3 são ajustados por proventos."),
        "Vencimento": st.column_config.DateColumn(format="DD/MM/YY"),
        "Entrada": st.column_config.NumberColumn(format="%.2f"), "Atual/saída": st.column_config.NumberColumn(format="%.2f"),
        "Resultado": st.column_config.NumberColumn(format="localized"),
        "Caixa p/ desfazer": st.column_config.NumberColumn(format="localized"),
    })
    if av["strike_ajustado"].any():
        st.caption("ℹ️ Strike ajustado por proventos desde a entrada — informe os proventos recebidos das ações em **Gerenciar** "
                   "para o resultado ficar completo.")

    if not abertas.empty and np.isfinite(spot):
        msgs = operacoes.exercicio(av, rs["S_venc"])
        if msgs:
            with st.container(border=True):
                st.markdown(f"**No vencimento, com o ativo a {brl(rs['S_venc'])}:**")
                for m in msgs:
                    st.markdown(f"- {m}")
        ks = [x for x in abertas["strike_atual"].dropna()]
        lo, hi = min([spot * 0.7] + [k * 0.9 for k in ks]), max([spot * 1.3] + [k * 1.1 for k in ks])
        S = np.linspace(lo, hi, 300)
        fixo = rs["realizado"] - rs["custos"] + rs["proventos"]
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Resultado por preço do ativo**")
            st.altair_chart(graficos.payoff(S, operacoes.resultado_vencimento(av, S) + fixo,
                                            operacoes.resultado_hoje_curva(av, S) + fixo, spot, altura=300), width="stretch")
        with g2:
            st.markdown("**Evolução do resultado (marcação diária)**")
            h = operacoes.historico_resultado(op, av, ler(config.ARQ_HIST_PERNAS), precos)
            if len(h) >= 2:
                st.altair_chart(graficos.historico(h, altura=300), width="stretch")
            else:
                st.caption("O histórico se forma a cada atualização diária de cotações.")

        vendidas = abertas[(abertas["tipo"] != "ACAO") & (abertas["sinal"] < 0)]
        for _, perna in vendidas.iterrows():
            with st.expander(f"🔁 Sugestões de rolagem para {perna['simbolo']} (recomprar e vender série mais longa)"):
                sg = operacoes.sugestoes_rolagem(perna, cadeia)
                if sg.empty:
                    st.caption("Sem séries líquidas para rolagem.")
                else:
                    st.dataframe(sg.assign(vencimento=pd.to_datetime(sg["vencimento"]).dt.date,
                                           liquidez=sg["liquidez"].map(LIQ)), hide_index=True, width="stretch",
                                 column_config={
                                     "simbolo": "Nova série", "vencimento": st.column_config.DateColumn("Vencimento", format="DD/MM/YY"),
                                     "du": "Pregões", "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                                     "preco_ref": st.column_config.NumberColumn("Prêmio", format="%.2f"), "fonte_preco": "Fonte",
                                     "credito": st.column_config.NumberColumn("Crédito/un.", format="%.2f"),
                                     "credito_total": st.column_config.NumberColumn("Crédito total", format="localized"),
                                     "ganho_se_exercido": st.column_config.NumberColumn("Ganho/un. se exercido", format="%.2f"),
                                     "rend_mes_exercido": st.column_config.NumberColumn("Rend. a.m. se exercido", format="percent"),
                                     "rend_mes_credito": st.column_config.NumberColumn("Rend. a.m. do crédito", format="percent"),
                                     "dist_strike": st.column_config.NumberColumn("Strike vs ativo", format="percent"),
                                     "delta": st.column_config.NumberColumn("Delta", format="%.2f"), "liquidez": "Liquidez",
                                     "oi_total": st.column_config.NumberColumn("Em aberto", format="localized"),
                                 })
                    st.caption("Para executar: em **Gerenciar**, encerre a perna atual e adicione a nova série.")

    gerenciar_operacao(op, av)


def gerenciar_operacao(op: pd.Series, av: pd.DataFrame) -> None:
    op_id = int(op["op_id"])
    with st.expander("⚙️ Gerenciar operação"):
        t1, t2, t3, t4 = st.tabs(["Encerrar pernas", "Adicionar perna / rolar", "Editar", "Excluir"])
        abertas = av[av["aberta"]]
        with t1:
            if abertas.empty:
                st.caption("Todas as pernas já estão encerradas.")
            else:
                sessao = pd.Timestamp(cadeia["sessao"].max()) if not cadeia.empty else pd.Timestamp.today()
                vencidas = abertas[(abertas["tipo"] != "ACAO") & (abertas["vencimento"] <= sessao)]
                if not vencidas.empty and st.button("Liquidar pernas vencidas pelo valor intrínseco", key=f"liq_{op_id}"):
                    operacoes.encerrar_pernas(op_id, dict(zip(vencidas["perna_id"], vencidas["preco_atual"].fillna(0))),
                                              vencidas["vencimento"].max(), custos=0.0)
                    st.cache_data.clear()
                    st.rerun()
                with st.form(f"enc_{op_id}"):
                    escolha = st.multiselect("Pernas a encerrar", abertas["perna_id"].tolist(), default=abertas["perna_id"].tolist(),
                                             format_func=lambda i: (lambda r: f"{r['lado']} {r['simbolo']} ({int(r['quantidade'])})")(
                                                 abertas.set_index("perna_id").loc[i]))
                    precos_saida = {}
                    cols = st.columns(max(len(abertas), 1))
                    for col, (_, r) in zip(cols, abertas.iterrows()):
                        precos_saida[r["perna_id"]] = col.number_input(f"Saída {r['simbolo']}", min_value=0.0, step=0.01,
                                                                       value=float(round(r["preco_atual"], 2)) if np.isfinite(r["preco_atual"]) else 0.0)
                    d = st.date_input("Data de saída", value=date.today(), format="DD/MM/YYYY")
                    custo = st.number_input("Custos de saída (R$)", min_value=0.0, step=1.0,
                                            value=round(float(abertas["custo_saida"].sum()), 2))
                    if st.form_submit_button("Encerrar selecionadas", type="primary") and escolha:
                        operacoes.encerrar_pernas(op_id, {i: precos_saida[i] for i in escolha}, d, custo)
                        st.cache_data.clear()
                        st.rerun()
        with t2:
            ch = cadeia[cadeia["ativo"] == op["ativo"]] if not cadeia.empty else pd.DataFrame()
            if ch.empty:
                st.caption("Ativo sem cadeia carregada — atualize as cotações.")
            else:
                with st.form(f"add_{op_id}"):
                    a1, a2, a3, a4 = st.columns([3, 1, 1, 1])
                    serie = a1.selectbox("Série", ch.sort_values(["vencimento", "tipo", "strike"])["simbolo"].tolist(),
                                         format_func=lambda s: (lambda r: f"{s} · {r['tipo']} K {num(r['strike'])} · "
                                                                f"{pd.Timestamp(r['vencimento']):%d/%m/%y} · {num(r['preco_ref'])}")(
                                             ch.set_index("simbolo").loc[s]))
                    lado = a2.selectbox("Lado", ["VENDA", "COMPRA"])
                    qtd = a3.number_input("Qtd", min_value=1, step=100, value=int(abertas["quantidade"].max()) if not abertas.empty else 100)
                    preco = a4.number_input("Preço", min_value=0.0, step=0.01, value=0.0, help="0 = usar o preço atual da série")
                    d = st.date_input("Data", value=date.today(), format="DD/MM/YYYY", key=f"add_d_{op_id}")
                    if st.form_submit_button("Adicionar perna"):
                        r = ch.set_index("simbolo").loc[serie]
                        p = {"tipo": r["tipo"], "simbolo": serie, "lado": lado, "quantidade": int(qtd), "strike": float(r["strike"]),
                             "vencimento": r["vencimento"], "preco": float(preco) if preco > 0 else round(float(r["preco_ref"]), 2)}
                        operacoes.adicionar_pernas(op_id, [p], d)
                        dados.atualizar_historico_pernas({serie}, cadeia, buscar_site=False)
                        st.cache_data.clear()
                        st.rerun()
        with t3:
            with st.form(f"edit_{op_id}"):
                e1, e2, e3 = st.columns(3)
                modo = e1.selectbox("Tipo", operacoes.MODOS, index=operacoes.MODOS.index(op["modo"]))
                nome = e2.text_input("Estratégia", value=op["estrategia"])
                data_ent = e3.date_input("Data de entrada", value=pd.Timestamp(op["data_entrada"]).date(), format="DD/MM/YYYY")
                e4, e5 = st.columns(2)
                custos = e4.number_input("Custos totais pagos (R$)", min_value=0.0, value=float(op["custos"]), step=1.0)
                prov = e5.number_input("Proventos recebidos (R$)", min_value=0.0, value=float(op["proventos"]), step=1.0,
                                       help="Dividendos/JCP das ações da operação (o strike das opções é reduzido nesses eventos).")
                notas = st.text_area("Notas", value=op["notas"])
                pernas_ed = st.data_editor(
                    av[operacoes.COLS_PERNA].drop(columns=["op_id"]), hide_index=True, width="stretch", num_rows="dynamic",
                    column_config={
                        "perna_id": st.column_config.NumberColumn("id", disabled=True),
                        "tipo": st.column_config.SelectboxColumn("Tipo", options=["CALL", "PUT", "ACAO"], required=True),
                        "lado": st.column_config.SelectboxColumn("Lado", options=["COMPRA", "VENDA"], required=True),
                        "vencimento": st.column_config.DateColumn("Vencimento", format="DD/MM/YYYY"),
                        "data_entrada": st.column_config.DateColumn("Entrada", format="DD/MM/YYYY"),
                        "data_saida": st.column_config.DateColumn("Saída", format="DD/MM/YYYY"),
                    },
                )
                if st.form_submit_button("Salvar alterações", type="primary"):
                    operacoes.editar(op_id, modo=modo, estrategia=nome, data_entrada=data_ent, custos=custos, proventos=prov, notas=notas)
                    operacoes.substituir_pernas(op_id, pernas_ed)
                    st.cache_data.clear()
                    st.rerun()
        with t4:
            if st.checkbox("Confirmo excluir esta operação definitivamente", key=f"conf_{op_id}") and st.button(
                "Excluir operação", key=f"del_{op_id}"
            ):
                operacoes.excluir(op_id)
                st.cache_data.clear()
                st.rerun()


with aba_ops:
    nova_operacao_manual()
    ops, av_all, resumos = _avaliar_tudo(dados.mtime(config.ARQ_OPERACOES), dados.mtime(config.ARQ_PERNAS),
                                         dados.mtime(config.ARQ_CADEIA), dados.mtime(config.ARQ_PRECOS),
                                         dados.mtime(config.ARQ_HIST_PERNAS))
    if ops.empty:
        st.info("Nenhuma operação registrada. Registre a partir da aba **Estratégias** ou monte uma operação manual acima.")
    else:
        lin = []
        for op in ops.itertuples(index=False):
            rs = resumos[op.op_id]
            lin.append({
                "op_id": op.op_id, "Tipo": op.modo, "Status": op.status, "Ativo": op.ativo, "Estratégia": op.estrategia,
                "Entrada": pd.Timestamp(op.data_entrada).date(),
                "Vencimento": pd.Timestamp(rs["vencimento"]).date() if pd.notna(rs["vencimento"]) else None,
                "Pregões": rs["du"] if op.status == "ABERTA" else None,
                "Resultado hoje": rs["resultado_hoje"],
                "Resultado %": rs["resultado_hoje"] / rs["base_capital"],
                "No venc. (ativo atual)": rs["resultado_venc"] if op.status == "ABERTA" else None,
                "Caixa p/ desfazer": rs["caixa_desfazer"] if op.status == "ABERTA" else None,
            })
        resumo_df = pd.DataFrame(lin)

        m = st.columns(4)
        for i, (modo, status, rot) in enumerate([("REAL", "ABERTA", "Real · abertas"), ("REAL", "ENCERRADA", "Real · realizado"),
                                                 ("ESTUDO", "ABERTA", "Estudo · abertas"), ("ESTUDO", "ENCERRADA", "Estudo · realizado")]):
            sub = resumo_df[(resumo_df["Tipo"] == modo) & (resumo_df["Status"] == status)]
            m[i].metric(f"{rot} ({len(sub)})", brl(sub["Resultado hoje"].sum(), True))

        f1, f2 = st.columns(2)
        modo_f = f1.segmented_control("Tipo", ["Todas", "REAL", "ESTUDO"], default="Todas", key="f_modo")
        status_f = f2.segmented_control("Status", ["ABERTA", "ENCERRADA", "Todas"], default="ABERTA", key="f_status")
        vis = resumo_df
        if modo_f and modo_f != "Todas":
            vis = vis[vis["Tipo"] == modo_f]
        if status_f and status_f != "Todas":
            vis = vis[vis["Status"] == status_f]
        ev = st.dataframe(
            vis, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="tab_ops",
            column_config={
                "op_id": st.column_config.NumberColumn("#", format="%d"),
                "Entrada": st.column_config.DateColumn(format="DD/MM/YY"),
                "Vencimento": st.column_config.DateColumn(format="DD/MM/YY"),
                "Resultado hoje": st.column_config.NumberColumn(format="localized", help="Se desfizer tudo hoje, com custos"),
                "Resultado %": st.column_config.NumberColumn(format="percent"),
                "No venc. (ativo atual)": st.column_config.NumberColumn(format="localized"),
                "Caixa p/ desfazer": st.column_config.NumberColumn(format="localized"),
            },
        )
        if ev.selection.rows:
            op_id = int(vis.iloc[ev.selection.rows[0]]["op_id"])
            op = ops[ops["op_id"] == op_id].iloc[0]
            detalhe_operacao(op, av_all[av_all["op_id"] == op_id].reset_index(drop=True), resumos[op_id])
        else:
            st.caption("Selecione uma operação para ver detalhes, payoff, evolução, rolagens e gerenciar.")


# ---------------------------------------------------------------------------
# Aba Mercado
# ---------------------------------------------------------------------------

with aba_mkt:
    uni = ler(config.ARQ_UNIVERSO)
    st.markdown("#### Universo de ativos com opções na B3")
    u1, u2 = st.columns([3, 1])
    por_estudo = meta.get("fonte_universo") == dados.FONTE_ESTUDO
    if por_estudo:
        u1.caption(
            "Volume e negócios: **média diária** em opções (todos os vencimentos) do estudo de liquidez do opcoes.net.br; "
            "contratos em aberto de todo o mercado; séries líquidas e formador de mercado medidos nas cadeias dos ativos "
            "acompanhados. Marque os ativos que o painel deve acompanhar."
        )
    else:
        u1.caption(
            "Pré-ranking pelos contratos em aberto de todo o mercado; volume financeiro em opções no último pregão "
            f"(vencimento mensal principal) medido nos {config.UNIVERSO_VARREDURA} maiores. Com login no opcoes.net.br "
            "(`.env`) o ranking usa o estudo de liquidez do site. Marque os ativos que o painel deve acompanhar."
        )
    tempo = "~1 min" if config.OPCOESNET_CPF else f"~{config.UNIVERSO_VARREDURA * 8 // 60 + 1} min"
    if u2.button(f"Reavaliar ranking ({tempo})", width="stretch"):
        barra = st.progress(0.0, text="Iniciando…")
        try:
            dados.atualizar_universo(progresso=lambda f, m: barra.progress(f, text=m))
        except Exception as exc:  # noqa: BLE001
            st.session_state["aviso"] = ("warning", f"Falha ao reavaliar o ranking: {exc}")
        barra.empty()
        st.cache_data.clear()
        st.rerun()
    if meta.get("aviso_universo"):
        st.warning(f"Último ranking ficou parcial — {meta['aviso_universo']}. Ativos sem medição aparecem como “Não medida”.")

    if uni.empty:
        st.info("Clique em **Reavaliar liquidez** para montar o ranking de todos os ativos com opções.")
    else:
        u = uni.copy()
        ivs = dados.iv_atm(cadeia) if not cadeia.empty else pd.Series(dtype=float)
        if not ind.empty:
            u = u.merge(ind[["ativo", "hv21", "hv63", "tendencia", "var_3m"]], how="left", on="ativo")
        else:
            u[["hv21", "hv63", "tendencia", "var_3m"]] = np.nan
        u["iv_atm"] = u["ativo"].map(ivs)
        u["iv_hv"] = u["iv_atm"] / u["hv63"]
        u.insert(0, "Acompanhar", u["ativo"].isin(acompanhados))
        ed = st.data_editor(
            u[["Acompanhar", "ativo", "classe_liquidez", "volume_fin", "negocios", "series_liquidas", "series_fm", "oi_financeiro",
               "series_em_aberto", "spot", "iv_atm", "hv21", "hv63", "iv_hv", "tendencia", "var_3m"]],
            hide_index=True, width="stretch", height=420, key="ed_universo",
            disabled=[c for c in u.columns if c != "Acompanhar"],
            column_config={
                "ativo": "Ativo", "classe_liquidez": "Liquidez",
                "volume_fin": st.column_config.NumberColumn("Volume opções (R$/dia)" if por_estudo else "Volume opções (R$)",
                                                            format="compact"),
                "negocios": st.column_config.NumberColumn("Negócios/dia" if por_estudo else "Negócios", format="%.0f"),
                "series_liquidas": st.column_config.NumberColumn("Séries líquidas", format="%d"),
                "series_fm": st.column_config.NumberColumn("Séries c/ formador", format="%d"),
                "oi_financeiro": st.column_config.NumberColumn("Em aberto (R$)", format="compact",
                                                               help="Contratos em aberto × strike, todas as séries"),
                "series_em_aberto": st.column_config.NumberColumn("Séries em aberto", format="%d"),
                "spot": st.column_config.NumberColumn("Preço", format="%.2f"),
                "iv_atm": st.column_config.NumberColumn("IV ATM 1m", format="percent"),
                "hv21": st.column_config.NumberColumn("Vol. hist. 1m", format="percent"),
                "hv63": st.column_config.NumberColumn("Vol. hist. 3m", format="percent"),
                "iv_hv": st.column_config.NumberColumn("IV ÷ vol. hist.", format="%.2f",
                                                       help="> 1: opções caras vs. a vol. realizada (favorece venda)"),
                "tendencia": "Tendência", "var_3m": st.column_config.NumberColumn("Var. 3m", format="percent"),
            },
        )
        s1, s2, s3 = st.columns([1, 1, 2])
        sel = ed.loc[ed["Acompanhar"], "ativo"].tolist()
        if s1.button(f"Salvar lista ({len(sel)} ativos)", width="stretch"):
            dados.salvar_ativos_acompanhados(sel)
            st.toast("Lista de ativos acompanhados salva.", icon="✅")
            st.rerun()
        if s2.button(f"Top {config.UNIVERSO_TOP_N} mais líquidos", width="stretch"):
            dados.salvar_ativos_acompanhados(dados.top_liquidos(uni))
            st.rerun()
        s3.caption(f"Ranking de {pd.Timestamp(uni['avaliado_em'].iloc[0]):%d/%m/%Y %H:%M}. IV/vol. histórica só para ativos já "
                   "atualizados (acompanhados).")

    st.markdown("#### Cadeia de opções")
    if cadeia.empty:
        st.info("Atualize as cotações para ver as cadeias.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        ativo = c1.selectbox("Ativo", sorted(cadeia["ativo"].unique()), key="mkt_ativo")
        ch = cadeia[(cadeia["ativo"] == ativo) & (cadeia["du"] > 0)]
        vencs = sorted(ch["vencimento"].unique())
        venc = c2.selectbox("Vencimento", vencs, format_func=lambda v: f"{pd.Timestamp(v):%d/%m/%Y} ({int(ch.loc[ch['vencimento'] == v, 'du'].iloc[0])} pregões)",
                            key="mkt_venc")
        so_liq = c3.toggle("Só séries com liquidez média/alta", value=True)
        g = ch[ch["vencimento"] == venc]
        spot = float(g["spot"].iloc[0])
        st.caption(f"{ativo} a {brl(spot)} · pregão {pd.Timestamp(g['sessao'].iloc[0]):%d/%m/%Y} · "
                   f"IV ATM {pct(dados.iv_atm(g, int(g['du'].iloc[0])).get(ativo, np.nan))}")
        st.altair_chart(graficos.smile(g[g["strike"].between(spot * 0.7, spot * 1.3)], spot, altura=260), width="stretch")
        if so_liq:
            g = g[g["liquidez"] >= 2]
        cols = {"simbolo": "Série", "strike": "Strike", "ultimo": "Último", "data_ultimo": "Últ. negócio", "negocios": "Negócios",
                "volume_fin": "Volume (R$)", "teorico": "Justo", "iv_mercado": "IV negócio", "iv_modelo": "IV smile",
                "delta": "Delta", "theta": "Theta/dia", "oi_total": "Em aberto", "fm": "Formador", "liquidez": "Liquidez"}
        if "iv_site" in g.columns and g["iv_site"].notna().any():
            cols = {**{k: v for k, v in cols.items() if k != "delta"}, "iv_site": "IV site", "delta": "Delta"}
        cfg = {
            "Strike": st.column_config.NumberColumn(format="%.2f"), "Último": st.column_config.NumberColumn(format="%.2f"),
            "Últ. negócio": st.column_config.DateColumn(format="DD/MM"), "Volume (R$)": st.column_config.NumberColumn(format="compact"),
            "Justo": st.column_config.NumberColumn(format="%.2f"), "IV negócio": st.column_config.NumberColumn(format="percent"),
            "IV smile": st.column_config.NumberColumn(format="percent"), "Delta": st.column_config.NumberColumn(format="%.2f"),
            "IV site": st.column_config.NumberColumn(format="percent", help="IV calculada pelo opcoes.net.br (com login)"),
            "Theta/dia": st.column_config.NumberColumn(format="%.3f"), "Em aberto": st.column_config.NumberColumn(format="compact"),
        }
        k1, k2 = st.columns(2)
        for col, tipo in ((k1, "CALL"), (k2, "PUT")):
            t = g[g["tipo"] == tipo].sort_values("strike")[list(cols)].rename(columns=cols)
            t["Liquidez"] = t["Liquidez"].map(LIQ)
            col.markdown(f"**{tipo}s**")
            col.dataframe(t, hide_index=True, width="stretch", height=480, column_config=cfg)
