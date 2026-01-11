#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv


def load_config(env_path=".env"):
    load_dotenv(env_path)

    cfg = {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
    }

    missing = [k for k in ["user", "password", "database"] if not cfg.get(k)]
    if missing:
        raise ValueError(f"Variáveis faltando no .env: {', '.join(missing)}")

    return cfg


def read_sql_file(sql_path: str) -> str:
    p = Path(sql_path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo SQL não encontrado: {sql_path}")
    return p.read_text(encoding="utf-8")


def execute_sql_file(sql_path: str, env_path=".env"):
    cfg = load_config(env_path)
    sql = read_sql_file(sql_path)

    conn = None
    try:
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            charset=cfg["charset"],
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
            client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS,
        )

        with conn.cursor() as cursor:
            for statement in filter(None, sql.split(";")):
                stmt = statement.strip()
                if not stmt:
                    continue

                cursor.execute(stmt)

                if cursor.description:
                    rows = cursor.fetchall()
                    print(f"[SELECT] {stmt[:80]}... -> {len(rows)} linhas")
                else:
                    print(f"[OK] {stmt[:80]}... -> {cursor.rowcount} linhas afetadas")

        conn.commit()
        print("✅ Commit realizado com sucesso.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ Erro ao executar SQL: {e}", file=sys.stderr)
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python run_sql_file_pymysql.py arquivo.sql [.env]", file=sys.stderr)
        sys.exit(1)

    sql_file = sys.argv[1]
    env_file = sys.argv[2] if len(sys.argv) >= 3 else ".env"

    execute_sql_file(sql_file, env_file)
