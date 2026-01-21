# Finance Options Dashboard (Modular)

Este pacote reorganiza o `app_streamlit_options.py` em módulos menores para facilitar a adição de novas estratégias.

## Como rodar
1. Copie a pasta inteira para seu projeto (ou descompacte).
2. Garanta um `.env` na mesma pasta do `app_streamlit_options.py` com:
   - DB_HOST, DB_PORT, DB_USER, DB_PASSWORD (ou DB_PASS), DB_NAME
3. Instale dependências:
   - streamlit, pandas, numpy, pymysql, python-dotenv, matplotlib
4. Rode:
   ```bash
   streamlit run app_streamlit_options.py
   ```

## Estrutura
- `db/` conexão e repositório de consultas
- `features/` moneyness, liquidez, regime, payoff
- `strategies/` cada estratégia em arquivo (plugin)
- `ui/` widgets (tabela selecionável, formatador)

## Booster Horizontal (PUTs)
- Usa o vencimento selecionado como **curto**.
- Para o **longo**, pega 2 ou 3 vencimentos à frente (`booster_long_steps`).
- Payoff é uma **aproximação no vencimento curto**, com valor residual do longo proporcional ao tempo restante.
