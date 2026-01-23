# Finance Options Dashboard — Modular (Parquet backend)

Versão **modular** do dashboard de opções, usando **arquivos Parquet** em vez de conexão com banco.

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

## Rodando localmente
```bash
pip install -r requirements.txt
streamlit run app_streamlit_options.py
```
