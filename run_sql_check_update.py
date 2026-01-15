#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pymysql


# -----------------------------
# Helpers: path + SQL loading
# -----------------------------
def find_project_root(start: Path) -> Path:
    """
    Encontra a raiz do projeto procurando marcadores ('.env' ou '.git')
    subindo a partir do arquivo atual.
    """
    for p in [start] + list(start.parents):
        if (p / ".env").exists() or (p / ".git").exists():
            return p
    # fallback: pasta do script
    return start.parent


def load_sql(sql_path: Path) -> str:
    if not sql_path.exists():
        raise FileNotFoundError(f"Arquivo SQL não encontrado: {sql_path}")
    return sql_path.read_text(encoding="utf-8")


def normalize_sql(sql: str) -> str:
    # remove BOM + whitespace
    return sql.replace("\ufeff", "").strip()


# -----------------------------
# Helpers: formatting
# -----------------------------
def fmt(x, nd=2):
    if x is None or x == "":
        return "-"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return f"{x}"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def fmt_date(x):
    return "-" if (x is None or x == "") else str(x)


def fmt_dt(x):
    return "-" if (x is None or x == "") else str(x)


# -----------------------------
# Output: SUMMARY
# -----------------------------
def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 132)
    print("RESUMO (sanity check — opções agregadas / vencimento mais próximo)".ljust(132))
    print("=" * 132)

    header = (
        "ticker".ljust(10)
        + " | spot_last".ljust(20)
        + " | spot_prev".ljust(20)
        + " | exp_near".ljust(12)
        + " | opt_last_day".ljust(13)
        + " | n_ctrt".ljust(8)
        + " | vol".ljust(10)
        + " | trades".ljust(9)
        + " | vwap".ljust(10)
        + " | collected_at".ljust(20)
        + " | status"
    )
    print(header)
    print("-" * 132)

    for r in sorted(rows, key=lambda x: x.get("ticker") or ""):
        ticker = r.get("ticker", "-")

        # Spot
        spot_last_txt = (
            f"{fmt_date(r.get('bar_last_trade_date'))} "
            f"{fmt(r.get('bar_last_close'), 2)}"
        )
        spot_prev_txt = (
            f"{fmt_date(r.get('bar_prev_trade_date'))} "
            f"{fmt(r.get('bar_prev_close'), 2)}"
        )

        # Opções (agregado)
        exp_near = r.get("opt_nearest_expiry_date")
        opt_last_day = r.get("exp_last_trade_date")
        n_ctrt = r.get("exp_last_n_contracts")
        vol = r.get("exp_last_total_volume")
        trades = r.get("exp_last_total_trades")
        vwap = r.get("exp_last_vwap_last_price")
        colat = r.get("exp_last_collected_at")

        # Checks
        issues = []
        if r.get("bar_last_trade_date") is None:
            issues.append("NO_SPOT_LAST")
        if r.get("bar_prev_trade_date") is None:
            issues.append("NO_SPOT_PREV")
        if exp_near is None:
            issues.append("NO_OPT_NEAR_EXP")
        if opt_last_day is None:
            issues.append("NO_OPT_DAY")
        if vol in (None, 0):
            issues.append("VOL_0")
        if vwap is None:
            issues.append("VWAP_NULL")
        if colat is None:
            issues.append("NO_COLLECTED_AT")

        status = "OK" if not issues else "ATENÇÃO: " + ",".join(issues)

        line = (
            ticker.ljust(10)
            + " | " + spot_last_txt.ljust(19)
            + " | " + spot_prev_txt.ljust(19)
            + " | " + fmt_date(exp_near).ljust(11)
            + " | " + fmt_date(opt_last_day).ljust(12)
            + " | " + fmt(n_ctrt).ljust(7)
            + " | " + fmt(vol).ljust(9)
            + " | " + fmt(trades).ljust(8)
            + " | " + fmt(vwap, 4).ljust(9)
            + " | " + fmt_dt(colat).ljust(19)
            + " | " + status
        )
        print(line)

    print("=" * 132 + "\n")


# -----------------------------
# Output: CARDS
# -----------------------------
def print_cards(rows: list[dict]) -> None:
    for r in sorted(rows, key=lambda x: x.get("ticker") or ""):
        ticker = r.get("ticker", "-")
        expiry = r.get("opt_latest_expiry_date")

        # Spot
        bar_last_d = r.get("bar_last_trade_date")
        bar_last_c = r.get("bar_last_close")
        bar_prev_d = r.get("bar_prev_trade_date")
        bar_prev_c = r.get("bar_prev_close")

        # ATM último
        atm_last_d = r.get("atm_last_trade_date")
        atm_last_spot = r.get("atm_last_spot_close")
        atm_last_sym = r.get("atm_last_option_symbol")
        atm_last_type = r.get("atm_last_option_type")
        atm_last_k = r.get("atm_last_strike")
        atm_last_diff = r.get("atm_last_abs_strike_diff")
        atm_last_px = r.get("atm_last_last_price")
        atm_last_tr = r.get("atm_last_trades")
        atm_last_vol = r.get("atm_last_volume")
        atm_last_col = r.get("atm_last_collected_at")

        # ATM penúltimo
        atm_prev_d = r.get("atm_prev_trade_date")
        atm_prev_spot = r.get("atm_prev_spot_close")
        atm_prev_sym = r.get("atm_prev_option_symbol")
        atm_prev_type = r.get("atm_prev_option_type")
        atm_prev_k = r.get("atm_prev_strike")
        atm_prev_diff = r.get("atm_prev_abs_strike_diff")
        atm_prev_px = r.get("atm_prev_last_price")
        atm_prev_tr = r.get("atm_prev_trades")
        atm_prev_vol = r.get("atm_prev_volume")
        atm_prev_col = r.get("atm_prev_collected_at")

        # VWAP agregado
        exp_last_d = r.get("exp_last_trade_date")
        exp_last_n = r.get("exp_last_n_contracts")
        exp_last_vol = r.get("exp_last_total_volume")
        exp_last_tr = r.get("exp_last_total_trades")
        exp_last_vwap = r.get("exp_last_vwap_last_price")

        exp_prev_d = r.get("exp_prev_trade_date")
        exp_prev_n = r.get("exp_prev_n_contracts")
        exp_prev_vol = r.get("exp_prev_total_volume")
        exp_prev_tr = r.get("exp_prev_total_trades")
        exp_prev_vwap = r.get("exp_prev_vwap_last_price")

        print("=" * 88)
        print(f"{ticker}  |  Vencimento mais recente: {fmt_date(expiry)}")
        print("-" * 88)

        print("SPOT (daily_bars)")
        print(f"  Último     : {fmt_date(bar_last_d)}  close={fmt(bar_last_c, 2)}")
        print(f"  Penúltimo  : {fmt_date(bar_prev_d)}  close={fmt(bar_prev_c, 2)}")
        print()

        print("ATM (strike mais próximo do spot) dentro do vencimento mais recente")
        print(f"  Último dia : {fmt_date(atm_last_d)}  spot={fmt(atm_last_spot,2)}")
        print(f"             contrato={fmt(atm_last_sym)}  {fmt(atm_last_type)}  K={fmt(atm_last_k,2)}  |K-spot|={fmt(atm_last_diff,2)}")
        print(f"             quote: last_price={fmt(atm_last_px,2)}  trades={fmt(atm_last_tr)}  volume={fmt(atm_last_vol)}  collected_at={fmt_dt(atm_last_col)}")
        print()
        print(f"  Penúltimo  : {fmt_date(atm_prev_d)}  spot={fmt(atm_prev_spot,2)}")
        print(f"             contrato={fmt(atm_prev_sym)}  {fmt(atm_prev_type)}  K={fmt(atm_prev_k,2)}  |K-spot|={fmt(atm_prev_diff,2)}")
        print(f"             quote: last_price={fmt(atm_prev_px,2)}  trades={fmt(atm_prev_tr)}  volume={fmt(atm_prev_vol)}  collected_at={fmt_dt(atm_prev_col)}")
        print()

        print("AGREGADO do vencimento (sanity check) — VWAP por dia (ponderado por volume)")
        print(f"  Último     : {fmt_date(exp_last_d)}  n_contracts={fmt(exp_last_n)}  vol={fmt(exp_last_vol)}  trades={fmt(exp_last_tr)}  VWAP={fmt(exp_last_vwap,4)}")
        print(f"  Penúltimo  : {fmt_date(exp_prev_d)}  n_contracts={fmt(exp_prev_n)}  vol={fmt(exp_prev_vol)}  trades={fmt(exp_prev_tr)}  VWAP={fmt(exp_prev_vwap,4)}")
        print()

        alerts = []
        if expiry is None:
            alerts.append("SEM option_quote para o ativo (expiry NULL).")
        if bar_last_d is None:
            alerts.append("SEM daily_bars (spot) para o ativo.")
        if atm_last_sym in (None, "", "-"):
            alerts.append("SEM símbolo ATM no último dia (join falhou).")
        if atm_last_sym not in (None, "", "-") and atm_last_px is None:
            alerts.append("ATM do último dia sem last_price (campo NULL na fonte ou import).")
        if exp_last_vol in (None, 0):
            alerts.append("Volume agregado no último dia é 0/NULL (verifique import de volume).")

        if alerts:
            print("ALERTAS:")
            for a in alerts:
                print(f"  - {a}")
            print()

    print("=" * 88)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa SQL de sanity check e imprime resumo + cards por ticker."
    )
    parser.add_argument(
        "--sql",
        default=None,
        help="Caminho do arquivo .sql (default: <project_root>/sql/check_import_agg_near_expiry.sql)",
    )
    parser.add_argument("--summary-only", action="store_true", help="Mostra apenas o resumo.")
    parser.add_argument("--cards-only", action="store_true", help="Mostra apenas os cards.")
    args = parser.parse_args()

    # Carrega .env (procura no cwd e no project_root)
    load_dotenv()

    # Detecta raiz
    project_root = find_project_root(Path(__file__).resolve())

    # Resolve caminho do SQL
    if args.sql:
        sql_file = Path(args.sql).expanduser().resolve()
    else:
        sql_file = project_root / "sql" / "check_import_agg_near_expiry.sql"

    # DB config
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    db = os.getenv("DB_NAME", "finance_options")

    if not user or password is None:
        print("ERRO: defina DB_USER e DB_PASSWORD no .env", file=sys.stderr)
        return 2

    # Lê SQL
    try:
        sql = normalize_sql(load_sql(sql_file))
        if not sql:
            print(f"ERRO: SQL vazio (arquivo: {sql_file})", file=sys.stderr)
            return 3
    except Exception as e:
        print(f"ERRO lendo SQL: {e}", file=sys.stderr)
        return 3

    # Executa
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        if not rows:
            print("Sem resultados.")
            return 0

        # Imprime outputs
        if args.cards_only and args.summary_only:
            # se o usuário passar os dois, mostramos ambos
            args.cards_only = False
            args.summary_only = False

        if not args.cards_only:
            print_summary(rows)

        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
