#!/usr/bin/env python3
"""
Export MySQL/MariaDB tables from schema finance_options to Parquet.

Supports:
- Full export (default)
- Incremental export by trade_date >= START_DATE
- "latest-only" export (keeps only the most recent trade_date per asset for option_quote/option_model/option_chain)
- Compression (snappy by default)

Tables exported (if present):
- assets
- daily_bars
- option_chain
- option_quote
- option_model
- yield_curve

Outputs:
- data_parquet/<table>.parquet

Env vars in .env:
- MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
- PARQUET_OUTDIR (default: ./data_parquet)
- PARQUET_COMPRESSION (default: snappy)
"""

import os
import sys
import argparse
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def die(msg: str, code: int = 1):
    print(f"[ERRO] {msg}", file=sys.stderr)
    sys.exit(code)


def env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        die(f"Variável de ambiente ausente: {name}")
    return v

def build_engine():
    host = env_required("DB_HOST")
    port = os.getenv("DB_PORT", "3306")
    user = env_required("DB_USER")
    pwd = env_required("DB_PASSWORD")
    db  = env_required("DB_NAME")

    # Using PyMySQL driver (pure python). Install: pip install pymysql
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def table_exists(conn, table_name: str) -> bool:
    q = text("""
        SELECT COUNT(*) AS c
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = :t
    """)
    r = conn.execute(q, {"t": table_name}).mappings().first()
    return bool(r and int(r["c"]) > 0)


def read_sql_df(conn, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    return pd.read_sql(text(sql), conn, params=params or {})


def export_df(df: pd.DataFrame, out_path: str, compression: str):
    # Normalize dtypes a bit (optional)
    # Keep as-is to preserve fidelity.
    df.to_parquet(out_path, index=False, compression=compression)


def max_trade_date_by_asset(conn, table: str) -> pd.DataFrame:
    # Works for option_quote/option_model/daily_bars
    sql = f"""
        SELECT asset_id, MAX(trade_date) AS max_trade_date
        FROM {table}
        GROUP BY asset_id
    """
    return read_sql_df(conn, sql)


def export_assets(conn, outdir: str, compression: str):
    sql = "SELECT * FROM assets"
    df = read_sql_df(conn, sql)
    export_df(df, os.path.join(outdir, "assets.parquet"), compression)
    print(f"[OK] assets: {len(df):,} linhas")


def export_daily_bars(conn, outdir: str, compression: str, start_date: Optional[str]):
    if start_date:
        sql = "SELECT * FROM daily_bars WHERE trade_date >= :d"
        df = read_sql_df(conn, sql, {"d": start_date})
    else:
        sql = "SELECT * FROM daily_bars"
        df = read_sql_df(conn, sql)

    export_df(df, os.path.join(outdir, "daily_bars.parquet"), compression)
    print(f"[OK] daily_bars: {len(df):,} linhas")


def export_yield_curve(conn, outdir: str, compression: str, start_date: Optional[str]):
    df = pd.DataFrame()  # garante variável definida

    if start_date:
        sql = "SELECT * FROM yield_curve WHERE trade_date >= :d"
        df = read_sql_df(conn, sql, {"d": start_date})
    else:
        sql = "SELECT * FROM yield_curve"
        df = read_sql_df(conn, sql)

    export_df(df, os.path.join(outdir, "yield_curve.parquet"), compression)
    print(f"[OK] yield_curve: {len(df):,} linhas")


def export_option_chain(conn, outdir: str, compression: str, start_date: Optional[str], latest_only: bool):
    if latest_only:
        # Keep only latest trade_date per asset_id in option_chain
        maxd = max_trade_date_by_asset(conn, "option_chain")
        if maxd.empty:
            df = pd.DataFrame()
        else:
            # join back to get all expiries for that latest trade_date
            # This avoids large IN clauses
            df = maxd.merge(
                read_sql_df(conn, "SELECT * FROM option_chain"),
                left_on=["asset_id", "max_trade_date"],
                right_on=["asset_id", "trade_date"],
                how="inner"
            ).drop(columns=["max_trade_date"])
    else:
        if start_date:
            sql = "SELECT * FROM option_chain WHERE trade_date >= :d"
            df = read_sql_df(conn, sql, {"d": start_date})
        else:
            sql = "SELECT * FROM option_chain"
            df = read_sql_df(conn, sql)

    export_df(df, os.path.join(outdir, "option_chain.parquet"), compression)
    print(f"[OK] option_chain: {len(df):,} linhas")


def export_option_quote(conn, outdir: str, compression: str, start_date: Optional[str], latest_only: bool):
    if latest_only:
        maxd = max_trade_date_by_asset(conn, "option_quote")
        if maxd.empty:
            df = pd.DataFrame()
        else:
            # Use join in SQL for efficiency
            sql = """
                SELECT q.*
                FROM option_quote q
                INNER JOIN (
                    SELECT asset_id, MAX(trade_date) AS max_trade_date
                    FROM option_quote
                    GROUP BY asset_id
                ) t
                ON q.asset_id = t.asset_id AND q.trade_date = t.max_trade_date
            """
            df = read_sql_df(conn, sql)
    else:
        if start_date:
            sql = "SELECT * FROM option_quote WHERE trade_date >= :d"
            df = read_sql_df(conn, sql, {"d": start_date})
        else:
            sql = "SELECT * FROM option_quote"
            df = read_sql_df(conn, sql)

    export_df(df, os.path.join(outdir, "option_quote.parquet"), compression)
    print(f"[OK] option_quote: {len(df):,} linhas")


def export_option_model(conn, outdir: str, compression: str, start_date: Optional[str], latest_only: bool):
    if latest_only:
        sql = """
            SELECT m.*
            FROM option_model m
            INNER JOIN (
                SELECT asset_id, MAX(trade_date) AS max_trade_date
                FROM option_model
                GROUP BY asset_id
            ) t
            ON m.asset_id = t.asset_id AND m.trade_date = t.max_trade_date
        """
        df = read_sql_df(conn, sql)
    else:
        if start_date:
            sql = "SELECT * FROM option_model WHERE trade_date >= :d"
            df = read_sql_df(conn, sql, {"d": start_date})
        else:
            sql = "SELECT * FROM option_model"
            df = read_sql_df(conn, sql)

    export_df(df, os.path.join(outdir, "option_model.parquet"), compression)
    print(f"[OK] option_model: {len(df):,} linhas")


def main():
    parser = argparse.ArgumentParser(description="Export finance_options MySQL tables to Parquet.")
    parser.add_argument("--start-date", default=None,
                        help="Export incremental: trade_date >= YYYY-MM-DD (applies to date-based tables).")
    parser.add_argument("--latest-only", action="store_true",
                        help="Export only latest trade_date per asset_id for option_quote/option_model/option_chain.")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="List of tables to skip (e.g., yield_curve option_chain).")
    args = parser.parse_args()

    load_dotenv(override=False)

    outdir = os.getenv("PARQUET_OUTDIR", os.path.join(os.getcwd(), "data_parquet"))
    compression = os.getenv("PARQUET_COMPRESSION", "snappy")

    ensure_outdir(outdir)

    engine = build_engine()

    with engine.connect() as conn:
        # Validate schema
        db = os.getenv("DB_NAME", "")
        if db != "finance_options":
            print(f"[AVISO] MYSQL_DATABASE={db} (não é 'finance_options'). Vou exportar do schema configurado mesmo.")

        exporters = [
            ("assets",       export_assets),
            ("daily_bars",   export_daily_bars),
            ("option_chain", export_option_chain),
            ("option_quote", export_option_quote),
            ("option_model", export_option_model),
            ("yield_curve",  export_yield_curve),
        ]

        for tname, fn in exporters:
            if tname in args.skip:
                print(f"[SKIP] {tname}")
                continue

            if not table_exists(conn, tname):
                print(f"[SKIP] {tname} (tabela não existe no schema atual)")
                continue

            try:
                if tname == "assets":
                    fn(conn, outdir, compression)
                elif tname in ("daily_bars", "yield_curve"):
                    fn(conn, outdir, compression, args.start_date)
                elif tname in ("option_chain", "option_quote", "option_model"):
                    fn(conn, outdir, compression, args.start_date, args.latest_only)
                else:
                    die(f"Exporter não definido para {tname}")
            except Exception as e:
                die(f"Falha exportando {tname}: {e}")

    print(f"\n[OK] Export concluído em: {outdir}")


if __name__ == "__main__":
    main()
