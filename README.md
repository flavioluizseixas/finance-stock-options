# Painel de Opções (B3)

Painel local (Streamlit) para **encontrar estratégias com opções** nos ativos mais líquidos da B3 e
**registrar e acompanhar operações** — reais ou de estudo — até o vencimento.

```bash
pip install -r requirements.txt
python atualizar.py --universo   # 1ª vez: ranking de liquidez (~6 min) + cotações dos mais líquidos
streamlit run app.py
```

No dia a dia: `python atualizar.py` (ou o botão **Atualizar cotações** no painel), de preferência após o
fechamento; `python atualizar.py --operacoes` (botão **Só ativos com operações abertas**) é a versão rápida.
O opcoes.net.br limita requisições por IP: a coleta espera ~5 s entre chamadas e, se o site pedir uma
espera longa, para, salva o que já coletou e avisa — basta rodar de novo mais tarde.

`run_pipeline.ps1` atualiza e abre o painel; `run_pipeline.ps1 -SemApp` serve para o Agendador de Tarefas
do Windows, por exemplo:

```powershell
schtasks /Create /TN "Painel Opcoes" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:30 `
  /TR "powershell -ExecutionPolicy Bypass -File C:\caminho\finance-stock-options\run_pipeline.ps1 -SemApp"
```

## O que o painel faz

**🎯 Estratégias** — avalia todas as combinações com séries líquidas dos ativos acompanhados:
venda coberta, venda de put, travas de alta/baixa (crédito e débito), iron condor, collar, compra de
call/put e straddle/strangle. Filtros por visão de mercado (alta, baixa, neutra, volatilidade), prazo,
liquidez mínima por perna, probabilidade de lucro e perda máxima. Cada candidato mostra prêmio, capital,
lucro/perda máximos, pontos de equilíbrio, probabilidade de lucro, CVaR 5%, gregas e o payoff
(no vencimento e hoje). Um clique registra a operação como **ESTUDO** ou **REAL**.

**📒 Operações** — para cada operação:
- **Desfazer hoje**: caixa que entra/sai ao zerar tudo e o resultado final, com custos;
- **No vencimento**: resultado se o ativo terminar no preço escolhido (padrão: preço atual) e o que
  acontece no exercício (quem exerce, quantas ações, a que preço);
- strikes atualizados pelos ajustes de proventos da B3;
- evolução diária do resultado, sugestões de rolagem para pernas vendidas;
- encerrar pernas (parcial ou total), liquidar vencidas, adicionar pernas (rolagem), editar, excluir.

**🌐 Mercado** — ranking de liquidez de todos os ativos com opções em aberto na B3 (contratos em aberto
de todo o mercado; volume financeiro, séries líquidas e formador de mercado medidos nos 40 maiores),
IV ATM × volatilidade histórica, tendência; escolha dos ativos acompanhados; cadeia completa com smile.

## Metodologia

- **Fontes**: opcoes.net.br (cadeias, contratos em aberto, histórico de 5 pregões por série),
  Yahoo Finance (histórico diário do ativo), Banco Central (Selic). IV e gregas são calculadas localmente.
  Com conta gratuita no site (`OPCOESNET_CPF`/`OPCOESNET_SENHA` no `.env`), o ranking usa o estudo de
  liquidez do opcoes.net.br (média diária de 10 pregões, ~175 ativos) e a cadeia mostra também a IV do site.
  Ofertas de compra/venda (bid/ask) só com assinatura Premium.
- **Liquidez por série**: *Alta* = negociou no último pregão com ≥ 20 negócios e ≥ R$ 50 mil (ou tem
  formador de mercado); *Média* = negociou nos últimos 2 pregões com ≥ 5 negócios; *Baixa* = demais.
- **Preço justo**: IV de cada negócio do dia (Black-Scholes, dias úteis/252) → smile quadrático por
  vencimento, ponderado pelo nº de negócios. As estratégias são avaliadas por esse preço: o último negócio
  de cada série acontece em horários diferentes e, combinado entre pernas, cria arbitragens falsas.
- **Distribuição no vencimento**: simulação histórica filtrada — choques diários padronizados do próprio
  ativo desde 2018, somados no prazo da operação e escalados pela vol atual (EWMA robusta + 1 ano),
  com média no forward. Preserva caudas gordas e assimetria.
- **Vantagem esperada**: valor esperado da operação acima do CDI (inclui o rendimento do prêmio recebido
  e o custo do capital aplicado), descontados emolumentos e meio spread estimado. Ranking padrão:
  vantagem ÷ desvio-padrão do resultado, anualizado.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `app.py` | painel Streamlit |
| `atualizar.py` | atualização pela linha de comando |
| `opcoes/fontes.py` | coleta (opcoes.net.br, Yahoo, BCB) com respeito ao limite de requisições do site |
| `opcoes/dados.py` | armazenamento parquet, IV/smile/gregas/liquidez, atualização diária |
| `opcoes/estrategias.py` | geração e ranqueamento das estruturas |
| `opcoes/operacoes.py` | registro, avaliação, histórico e rolagem de operações |
| `data/operacoes.parquet`, `data/pernas.parquet` | suas operações (versionadas) |
| `data/historico_pernas.parquet` | fechamento diário das séries usadas nas operações |
| `data/cadeia_atual.parquet`, `data/cadeias/` | última cadeia enriquecida e arquivo diário (não versionados) |

Configurações opcionais (custos, taxa de juros, nº de ativos, limites de liquidez): copie `.env.example`
para `.env`.

`export_ms7_incremental.py` (exportação MetaStock) e `tesouro_vendas_dashboard/` são ferramentas
independentes do painel.
