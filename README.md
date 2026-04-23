# Finance Options Dashboard — Parquet First

Projeto consolidado em **arquivos Parquet**, sem depender de MySQL para o fluxo principal de atualização e consulta.

## Estrutura esperada de dados (Parquet)
Por padrão, o app procura em `./data/`:

- `data/assets.parquet`
- `data/daily_bars.parquet`
- `data/option_quote.parquet`
- `data/option_model.parquet`

Você pode sobrescrever via `.env`:

```bash
DATA_DIR=./data
PATH_ASSETS=./data/assets.parquet
PATH_DAILY=./data/daily_bars.parquet
PATH_QUOTE=./data/option_quote.parquet
PATH_MODEL=./data/option_model.parquet
```

## Apps principais
```bash
streamlit run app_streamlit_options.py
streamlit run app_streamlit_trade_log.py
streamlit run app_streamlit_convered_roll.py
```

## Pipeline de atualização
```bash
pip install -r requirements.txt
python finance_options_pipeline.py
```

O `finance_options_pipeline.py` lê os tickers do `.env.bsm`, atualiza:

- `daily_bars.parquet` via Yahoo Finance
- `option_quote.parquet` via `opcoes.net.br`
- `option_model.parquet` com IV e gregas Black-Scholes-Merton
- `yield_curve.parquet` com a curva usada no cálculo

O histórico de operações estruturadas fica em:

- `data/structured_trades.parquet`
- `data/structured_trade_legs.parquet`
