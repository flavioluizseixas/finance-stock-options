# app_trade_log.py
# ------------------------------------------------------------
# Log de Operações de Opções (trade + legs) com:
# - Entrada por LEG (ordem obrigatória):
#   1) Underlying -> 2) Expiry (com negócios) -> 3) Tipo -> 4) Opção (com negócios, default ATM)
#   5) last_price + ajuste entry_price -> 6) data operação -> 7) Incluir LEG (editor)
# - Quantidade em AÇÕES (100 em 100), default 100 => contracts = qty/100
#
# Histórico:
# - ver/editar cabeçalho, editar LEG (qty/entry/exit), remover operação
# - P&L por trade (entry/exit, BUY/SELL) + fees
# - "Fechar operação automaticamente": pede exit_price por leg, grava exit_dt e seta status=CLOSED
# - "Duplicar operação": clona trade+legs (sem exit), cria novo trade OPEN com trade_date=hoje
#
# NOVO (pedido):
# - direction calculado automaticamente (DEBIT/CREDIT) a partir dos LEGS (cashflow de entrada)
# - P&L exibido também:
#   - Total (R$)
#   - "por 100 ações" (por 1 unidade de estratégia = 1 contrato-base)
#   - "% do spot" (normalizado pelo spot * 100 ações * contrato-base)
#
# Dependências:
#   pip install streamlit pymysql python-dotenv
#
# .env (exemplo):
#   DB_HOST=localhost
#   DB_PORT=3306
#   DB_USER=root
#   DB_PASSWORD=******
#   DB_NAME=finance_options
#
# Exec:
#   streamlit run app_trade_log.py

import os
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import pymysql
from dotenv import load_dotenv

load_dotenv()


# ---------------- DB helpers ----------------
def get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "finance_options"),
        port=int(os.getenv("DB_PORT", "3306")),
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )

def q_all(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()

def q_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    rows = q_all(sql, params)
    return rows[0] if rows else None


# ---------------- Finance logic ----------------
def direction_from_legs(legs: List[Dict[str, Any]], multiplier: int = 100) -> str:
    """
    direction pela ENTRADA (cashflow):
      SELL => entrada de prêmio (positivo)
      BUY  => saída de prêmio (negativo)
    net_cash = sum( sign * entry_price * contracts * multiplier )
      sign = +1 para SELL, -1 para BUY
    Se net_cash > 0 => CREDIT, senão => DEBIT (inclui 0)
    """
    net_cash = 0.0
    for l in legs:
        side = (l.get("side") or "").upper()
        entry = float(l.get("entry_price") or 0.0)
        c = int(l.get("contracts") or 0)
        sign = 1.0 if side == "SELL" else -1.0
        net_cash += sign * entry * c * multiplier
    return "CREDIT" if net_cash > 0 else "DEBIT"

def recompute_and_update_direction(trade_id: int) -> None:
    """
    Recalcula direction com base nos legs atuais e atualiza option_trades.direction.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT quantity_multiplier FROM option_trades WHERE id=%s", (trade_id,))
            row = cur.fetchone()
            mult = int(row["quantity_multiplier"] or 100) if row else 100

            cur.execute("SELECT side, entry_price, contracts FROM option_trade_legs WHERE trade_id=%s", (trade_id,))
            legs = cur.fetchall()

            new_dir = direction_from_legs(legs, multiplier=mult)
            cur.execute("UPDATE option_trades SET direction=%s WHERE id=%s", (new_dir, trade_id))
    finally:
        conn.close()

def calc_leg_pnl(leg: Dict[str, Any], multiplier: int = 100) -> Optional[float]:
    """
    P&L do leg baseado em entry/exit.
       BUY:  (exit - entry) * contracts * multiplier
       SELL: (entry - exit) * contracts * multiplier
       Se exit_price for NULL -> None
    """
    entry = leg.get("entry_price")
    exitp = leg.get("exit_price")
    if entry is None or exitp is None:
        return None
    entry = float(entry)
    exitp = float(exitp)
    c = int(leg.get("contracts") or 0)
    side = (leg.get("side") or "").upper()
    if c <= 0:
        return 0.0
    if side == "BUY":
        return (exitp - entry) * c * multiplier
    if side == "SELL":
        return (entry - exitp) * c * multiplier
    return None

def calc_trade_pnl_total(legs: List[Dict[str, Any]], fees: float = 0.0, multiplier: int = 100) -> Tuple[Optional[float], float, bool]:
    """
    Retorna:
      - pnl_if_closed: soma legs - fees, somente se TODOS legs tiverem exit_price
      - pnl_partial: soma apenas legs com exit_price - fees
      - is_fully_closed: True se todos legs tiverem exit_price
    """
    fees = float(fees or 0.0)
    pnls = [calc_leg_pnl(l, multiplier=multiplier) for l in legs]
    partial_sum = sum([p for p in pnls if p is not None]) - fees
    fully_closed = all(p is not None for p in pnls) and len(legs) > 0
    closed_sum = (sum(pnls) - fees) if fully_closed else None
    return closed_sum, partial_sum, fully_closed

def contract_base(legs: List[Dict[str, Any]]) -> int:
    """
    Define uma 'unidade de estratégia' em contratos para normalização.
    Regra prática: usa o mínimo de contracts entre os legs (>0).
    (Em spreads/condors normalmente é igual; se não for, ainda dá uma unidade consistente.)
    """
    cs = [int(l.get("contracts") or 0) for l in legs if int(l.get("contracts") or 0) > 0]
    return min(cs) if cs else 1

def pnl_per_100(pnl_total: float, legs: List[Dict[str, Any]]) -> float:
    """
    Normaliza P&L para 'por 100 ações' = por 1 unidade de estratégia (contract_base).
    """
    base = contract_base(legs)
    return pnl_total / float(base)

def pnl_pct_spot(pnl_total: float, legs: List[Dict[str, Any]], spot: Optional[float], multiplier: int = 100) -> Optional[float]:
    """
    Normaliza P&L como % do notional: spot * multiplier * contract_base
    """
    if spot is None or spot <= 0:
        return None
    base = contract_base(legs)
    denom = float(spot) * float(multiplier) * float(base)
    if denom == 0:
        return None
    return (pnl_total / denom) * 100.0


# ---------------- UI helpers ----------------
def fmt_opt(r: Dict[str, Any]) -> str:
    lp = r["last_price"]
    return (
        f'{r["option_symbol"]} | K={float(r["strike"]):.2f} | '
        f'last={lp if lp is not None else "NA"} | '
        f'vol={int(r["volume"] or 0)} | trd={int(r["trades"] or 0)}'
    )

def pick_atm_index(quotes: List[Dict[str, Any]], spot: Optional[float]) -> Optional[int]:
    """Escolhe a opção mais ATM (menor |K-spot|), dentre opções com negócios.
       Desempate: mais trades, depois mais volume.
    """
    if not quotes:
        return None
    if spot is None:
        return 0

    scored: List[Tuple[float, int, int, int]] = []
    for i, r in enumerate(quotes):
        k = float(r["strike"])
        dist = abs(k - float(spot))
        tr = int(r["trades"] or 0)
        vol = int(r["volume"] or 0)
        scored.append((dist, -tr, -vol, i))
    scored.sort()
    return scored[0][-1]


# ---------------- Page ----------------
st.set_page_config(page_title="Option Trade Log", layout="wide")
st.title("Log de Operações com Opções")

if "legs" not in st.session_state:
    st.session_state.legs = []

# ---------------- (1) Underlying ----------------
assets = q_all("""
SELECT id AS asset_id, ticker
FROM assets
WHERE is_active = 1
ORDER BY ticker
""")

if not assets:
    st.error("Nenhum ativo encontrado em assets(is_active=1).")
    st.stop()

asset_map = {a["ticker"]: a["asset_id"] for a in assets}
ticker = st.selectbox("1) Underlying", options=list(asset_map.keys()), index=0)
asset_id = asset_map[ticker]

# as-of trade date (último pregão coletado para opções)
asof_row = q_one("SELECT MAX(trade_date) AS asof_trade_date FROM option_quote WHERE asset_id=%s", (asset_id,))
asof_trade_date = asof_row["asof_trade_date"] if asof_row else None

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    st.caption("Pregão de referência (última data com option_quote)")
    st.write(asof_trade_date)

# spot no pregão de referência
spot = None
with colB:
    spot_row = None
    if asof_trade_date is not None:
        spot_row = q_one(
            "SELECT close FROM daily_bars WHERE asset_id=%s AND trade_date=%s",
            (asset_id, asof_trade_date),
        )
    spot = float(spot_row["close"]) if spot_row and spot_row["close"] is not None else None
    st.caption("Spot (close) no pregão de referência")
    st.write(spot)

with colC:
    st.caption("Legs no editor")
    st.write(len(st.session_state.legs))

if asof_trade_date is None:
    st.warning("Não há option_quote coletado para este ativo ainda.")
    st.stop()

st.divider()

# ---------------- Leg builder ----------------
st.subheader("Montar um LEG (ordem obrigatória)")

# (2) Expiry filtrado por negócios (SUM(trades) > 0)
expiries = q_all("""
SELECT q.expiry_date
FROM option_quote q
WHERE q.asset_id=%s
  AND q.trade_date=%s
GROUP BY q.expiry_date
HAVING COALESCE(SUM(q.trades), 0) > 0
ORDER BY q.expiry_date
""", (asset_id, asof_trade_date))

expiry_options = [r["expiry_date"] for r in expiries]
if not expiry_options:
    st.warning("Não há vencimentos com negócios (trades) neste pregão de referência.")
    st.stop()

expiry = st.selectbox("2) Expiry (somente com negócios)", options=expiry_options, index=0)

# (3) Tipo
opt_type = st.selectbox("3) Tipo", options=["CALL", "PUT"], index=0)

# (4) Opções do tipo+vencimento, somente com negócios (trades > 0)
quotes = q_all("""
SELECT option_symbol, strike, last_price, volume, trades, collected_at
FROM option_quote
WHERE asset_id=%s
  AND trade_date=%s
  AND expiry_date=%s
  AND option_type=%s
  AND COALESCE(trades, 0) > 0
ORDER BY strike
""", (asset_id, asof_trade_date, expiry, opt_type))

if not quotes:
    st.warning("Não há opções com negócios para este tipo/vencimento no pregão de referência.")
    st.stop()

opt_labels = [fmt_opt(r) for r in quotes]
atm_idx = pick_atm_index(quotes, spot)

opt_idx = st.selectbox(
    "4) Opção (somente com negócios) — default: mais próxima de ATM",
    options=list(range(len(opt_labels))),
    format_func=lambda i: opt_labels[i],
    index=atm_idx if atm_idx is not None else 0
)
selected = quotes[opt_idx]

# (5) prêmio e ajustes + qty
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    side = st.selectbox("Side", options=["BUY", "SELL"], index=0)

with col2:
    qty_shares = st.number_input(
        "Quantidade (ações) — múltiplos de 100",
        min_value=100,
        step=100,
        value=100,
    )
    contracts = int(qty_shares // 100)
    st.caption("Contratos (qty/100)")
    st.write(contracts)

with col3:
    last_price = selected["last_price"]
    st.caption("last_price (referência)")
    st.write(last_price)

with col4:
    default_entry = float(last_price) if last_price is not None else 0.0
    entry_price = st.number_input(
        "entry_price (efetivado)",
        min_value=0.0,
        step=0.01,
        value=default_entry
    )

with col5:
    st.caption("Strike / Expiry")
    st.write(f"K={float(selected['strike']):.2f} | exp={expiry}")

# (6) Data da operação
trade_date = st.date_input("6) Data da operação", value=date.today())

# (7) incluir
def validate_new_leg(new_leg, existing_legs):
    if existing_legs:
        if new_leg["asset_id"] != existing_legs[0]["asset_id"]:
            return False, "Todos os legs devem ter o MESMO underlying."
        if new_leg["expiry"] != existing_legs[0]["expiry"]:
            return False, "Todos os legs devem ter o MESMO expiry."
        if new_leg["trade_date"] != existing_legs[0]["trade_date"]:
            return False, "Todos os legs devem ter a MESMA data da operação (trade_date)."
    if new_leg["contracts"] <= 0:
        return False, "Quantidade inválida: contratos deve ser > 0."
    return True, ""

if st.button("7) Incluir LEG", type="primary"):
    new_leg = {
        "asset_id": asset_id,
        "ticker": ticker,
        "expiry": expiry,
        "opt_type": opt_type,
        "side": side,
        "contracts": int(contracts),
        "qty_shares": int(qty_shares),
        "option_symbol": selected["option_symbol"],
        "strike": float(selected["strike"]),
        "entry_price": float(entry_price),
        "ref_last_price": float(last_price) if last_price is not None else None,
        "ref_collected_at": selected["collected_at"],
        "trade_date": trade_date,
    }

    ok, msg = validate_new_leg(new_leg, st.session_state.legs)
    if not ok:
        st.error(msg)
    else:
        new_leg["leg_no"] = len(st.session_state.legs) + 1
        st.session_state.legs.append(new_leg)
        st.success(f"LEG {new_leg['leg_no']} incluído.")

st.divider()

# ---------------- Legs editor ----------------
st.subheader("Editor de LEGS")

if not st.session_state.legs:
    st.info("Nenhum leg incluído ainda.")
else:
    for leg in st.session_state.legs:
        st.write(
            f"LEG {leg['leg_no']}: {leg['side']} {leg['opt_type']} "
            f"{leg['option_symbol']} K={leg['strike']:.2f} exp={leg['expiry']} "
            f"qty={leg['qty_shares']} ({leg['contracts']} ctrt) entry={leg['entry_price']}"
        )

    # direction preview do editor (antes de salvar)
    dir_preview = direction_from_legs(st.session_state.legs, multiplier=100)
    st.caption("Direction calculado (preview pela entrada de prêmio)")
    st.write(dir_preview)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Remover último LEG"):
            st.session_state.legs.pop()
            st.rerun()
    with c2:
        if st.button("Limpar LEGS"):
            st.session_state.legs = []
            st.rerun()
    with c3:
        st.caption("Validações: mesma expiry/underlying/trade_date (versão atual).")

st.divider()

# ---------------- Save trade + legs ----------------
st.subheader("Salvar operação")

strategy = st.selectbox(
    "Strategy",
    options=["CUSTOM", "SELL_PUT", "BUY_DITM_CALL", "VERTICAL_SPREAD", "COVERED_CALL", "STRADDLE"],
    index=0
)
fees = st.number_input("Fees (total)", min_value=0.0, step=0.01, value=0.0)
notes = st.text_area("Notes", value="")

def save_trade_and_legs():
    if not st.session_state.legs:
        st.error("Inclua ao menos 1 leg antes de salvar.")
        return

    td = st.session_state.legs[0]["trade_date"]

    # spot snapshot (prefere o pregão de referência; se não houver, tenta pelo trade_date)
    spot_row2 = q_one(
        "SELECT close FROM daily_bars WHERE asset_id=%s AND trade_date=%s",
        (asset_id, asof_trade_date),
    )
    spot2 = float(spot_row2["close"]) if spot_row2 and spot_row2["close"] is not None else None
    if spot2 is None:
        spot_row3 = q_one(
            "SELECT close FROM daily_bars WHERE asset_id=%s AND trade_date=%s",
            (asset_id, td),
        )
        spot2 = float(spot_row3["close"]) if spot_row3 and spot_row3["close"] is not None else None

    # direction automático a partir do editor de legs
    direction_auto = direction_from_legs(st.session_state.legs, multiplier=100)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO option_trades
                  (trade_date, strategy, asset_id, direction, status, quantity_multiplier,
                   fees, notes, underlying_spot)
                VALUES
                  (%s, %s, %s, %s, 'OPEN', 100,
                   %s, %s, %s)
                """,
                (td, strategy, asset_id, direction_auto, float(fees), notes, spot2)
            )
            trade_id = cur.lastrowid

            now_ts = datetime.now()
            for leg in st.session_state.legs:
                cur.execute(
                    """
                    INSERT INTO option_trade_legs
                      (trade_id, leg_no, side, opt_type, strike, expiry,
                       contracts, entry_price, entry_dt, option_symbol)
                    VALUES
                      (%s, %s, %s, %s, %s, %s,
                       %s, %s, %s, %s)
                    """,
                    (
                        trade_id,
                        leg["leg_no"],
                        leg["side"],
                        leg["opt_type"],
                        leg["strike"],
                        leg["expiry"],
                        leg["contracts"],
                        leg["entry_price"],
                        now_ts,
                        leg["option_symbol"],
                    )
                )

        st.success(f"Operação salva! trade_id={trade_id} | direction={direction_auto} | legs={len(st.session_state.legs)}")
        st.session_state.legs = []
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
    finally:
        conn.close()

st.button("Salvar operação (trade + legs)", type="primary", on_click=save_trade_and_legs)


# ============================================================
#                      HISTÓRICO / EDIÇÃO
# ============================================================
st.divider()
st.header("Histórico de operações (ver / corrigir / fechar / duplicar / remover)")

# ---- Filtros
colf1, colf2, colf3, colf4 = st.columns([1, 1, 1, 1])
with colf1:
    only_status = st.selectbox("Filtrar status", options=["(todos)", "OPEN", "CLOSED", "CANCELLED"], index=0)
with colf2:
    only_ticker = st.selectbox("Filtrar underlying", options=["(todos)"] + list(asset_map.keys()), index=0)
with colf3:
    limit_n = st.number_input("Mostrar últimos N", min_value=10, max_value=500, step=10, value=50)
with colf4:
    show_only_with_legs = st.checkbox("Somente operações com legs", value=True)

where = []
params: List[Any] = []

if only_status != "(todos)":
    where.append("t.status=%s")
    params.append(only_status)

if only_ticker != "(todos)":
    where.append("a.ticker=%s")
    params.append(only_ticker)

if show_only_with_legs:
    where.append("EXISTS (SELECT 1 FROM option_trade_legs l2 WHERE l2.trade_id=t.id)")

where_sql = ("WHERE " + " AND ".join(where)) if where else ""

trades = q_all(f"""
SELECT
  t.id,
  t.trade_date,
  t.strategy,
  t.direction,
  t.status,
  t.fees,
  t.underlying_spot,
  a.ticker,
  COUNT(l.id) AS n_legs,
  MIN(l.expiry) AS exp_min,
  MAX(l.expiry) AS exp_max
FROM option_trades t
JOIN assets a ON a.id = t.asset_id
LEFT JOIN option_trade_legs l ON l.trade_id = t.id
{where_sql}
GROUP BY t.id, t.trade_date, t.strategy, t.direction, t.status, t.fees, t.underlying_spot, a.ticker
ORDER BY t.trade_date DESC, t.id DESC
LIMIT %s
""", tuple(params + [int(limit_n)]))

if not trades:
    st.info("Nenhuma operação encontrada com esses filtros.")
    st.stop()

def fmt_trade(r):
    exp = r["exp_min"] if r["exp_min"] == r["exp_max"] else f'{r["exp_min"]}..{r["exp_max"]}'
    return (
        f'#{r["id"]} | {r["trade_date"]} | {r["ticker"]} | {r["strategy"]} | '
        f'{r["direction"]} | {r["status"]} | legs={r["n_legs"]} | exp={exp} | fees={r["fees"]}'
    )

trade_labels = [fmt_trade(r) for r in trades]
idx = st.selectbox("Selecione uma operação", options=list(range(len(trade_labels))), format_func=lambda i: trade_labels[i])
trade = trades[idx]
trade_id = trade["id"]

# ---- carregar cabeçalho e legs
head_row = q_one("""
SELECT t.*, a.ticker
FROM option_trades t
JOIN assets a ON a.id=t.asset_id
WHERE t.id=%s
""", (trade_id,))

legs_db = q_all("""
SELECT *
FROM option_trade_legs
WHERE trade_id=%s
ORDER BY leg_no
""", (trade_id,))

mult = int(head_row.get("quantity_multiplier") or 100)
fees_db = float(head_row.get("fees") or 0.0)

# Spot para % (prioriza snapshot; se não, tenta daily_bars no trade_date)
spot_for_pct = head_row.get("underlying_spot")
if spot_for_pct is None:
    spot_row_hist = q_one(
        "SELECT close FROM daily_bars WHERE asset_id=%s AND trade_date=%s",
        (head_row["asset_id"], head_row["trade_date"]),
    )
    spot_for_pct = float(spot_row_hist["close"]) if spot_row_hist and spot_row_hist["close"] is not None else None

# ---- direction auto (para exibir + (opcional) corrigir no DB)
direction_auto_now = direction_from_legs(legs_db, multiplier=mult)

# ---- P&L totals
pnl_closed, pnl_partial, fully_closed = calc_trade_pnl_total(legs_db, fees=fees_db, multiplier=mult)

# ---- modo de exibição do P&L
pnl_mode = st.radio(
    "Modo de P&L",
    options=["Total (R$)", "Por 100 ações", "% do spot"],
    horizontal=True
)

def format_pnl(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}"

def pnl_in_mode(pnl_value: Optional[float]) -> Optional[float]:
    if pnl_value is None:
        return None
    if pnl_mode == "Total (R$)":
        return pnl_value
    if pnl_mode == "Por 100 ações":
        return pnl_per_100(pnl_value, legs_db)
    # % do spot
    return pnl_pct_spot(pnl_value, legs_db, spot=spot_for_pct, multiplier=mult)

st.subheader(f"Operação #{trade_id} — detalhes")

colh1, colh2, colh3, colh4, colh5, colh6 = st.columns([1, 1, 1, 1, 1, 1])
with colh1:
    st.caption("Underlying")
    st.write(head_row["ticker"])
with colh2:
    st.caption("Trade date")
    st.write(head_row["trade_date"])
with colh3:
    st.caption("Strategy")
    st.write(head_row["strategy"])
with colh4:
    st.caption("Status / Fees")
    st.write(f'{head_row["status"]} / {fees_db}')
with colh5:
    st.caption("Direction (DB / Auto)")
    st.write(f'{head_row["direction"]} / {direction_auto_now}')
with colh6:
    st.caption("Spot p/ %")
    st.write(spot_for_pct)

st.markdown("**P&L (entry/exit) - fees**")
if pnl_closed is not None:
    st.write(f"Fechado: {format_pnl(pnl_in_mode(pnl_closed))}")
else:
    st.write(f"Parcial: {format_pnl(pnl_in_mode(pnl_partial))}  (ainda aberto: faltam exits)")

st.markdown("**Legs**")
if not legs_db:
    st.info("Esta operação não possui legs.")
else:
    for l in legs_db:
        lpnl = calc_leg_pnl(l, multiplier=mult)
        # normaliza no modo (por leg)
        if pnl_mode == "Total (R$)":
            lpnl_disp = lpnl
        elif pnl_mode == "Por 100 ações":
            # por 100 ações: divide pela unidade-base (mesma do trade)
            lpnl_disp = (lpnl / float(contract_base(legs_db))) if lpnl is not None else None
        else:
            lpnl_disp = pnl_pct_spot(lpnl, legs_db, spot=spot_for_pct, multiplier=mult) if lpnl is not None else None

        st.write(
            f'LEG {l["leg_no"]}: {l["side"]} {l["opt_type"]} {l["option_symbol"]} '
            f'K={float(l["strike"]):.2f} exp={l["expiry"]} '
            f'ctrt={l["contracts"]} entry={float(l["entry_price"]):.4f} '
            f'exit={l["exit_price"] if l["exit_price"] is not None else "-"} '
            f'| P&L={format_pnl(lpnl_disp)}'
        )

st.divider()
st.subheader("Ações: editar / fechar / duplicar / remover")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Direction (auto)", "Editar cabeçalho", "Editar um LEG", "Fechar operação (auto)", "Duplicar operação", "Remover operação"]
)

# ---------- Direction (auto): corrigir DB se divergir ----------
with tab0:
    st.info("Direction é calculado pela entrada líquida de prêmio (cashflow).")
    if head_row["direction"] != direction_auto_now:
        st.warning(f"DB está {head_row['direction']} mas o cálculo dá {direction_auto_now}.")
        if st.button("Atualizar direction no DB para o valor automático", type="primary"):
            try:
                recompute_and_update_direction(trade_id)
                st.success("Direction atualizado no DB.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao atualizar direction: {e}")
    else:
        st.success("Direction no DB está consistente com o cálculo automático.")

# ---------- Editar cabeçalho ----------
with tab1:
    with st.form(key=f"edit_head_{trade_id}"):
        new_trade_date = st.date_input("trade_date", value=head_row["trade_date"])
        new_strategy = st.selectbox(
            "strategy",
            options=["CUSTOM", "SELL_PUT", "BUY_DITM_CALL", "VERTICAL_SPREAD", "COVERED_CALL", "STRADDLE"],
            index=0 if head_row["strategy"] not in ["CUSTOM", "SELL_PUT", "BUY_DITM_CALL", "VERTICAL_SPREAD", "COVERED_CALL", "STRADDLE"]
            else ["CUSTOM", "SELL_PUT", "BUY_DITM_CALL", "VERTICAL_SPREAD", "COVERED_CALL", "STRADDLE"].index(head_row["strategy"])
        )
        new_status = st.selectbox("status", options=["OPEN", "CLOSED", "CANCELLED"], index=["OPEN","CLOSED","CANCELLED"].index(head_row["status"]))
        new_fees = st.number_input("fees", min_value=0.0, step=0.01, value=float(head_row["fees"] or 0.0))
        new_notes = st.text_area("notes", value=head_row["notes"] or "")

        st.caption("direction é automático (não editável aqui).")

        submitted = st.form_submit_button("Salvar alterações do cabeçalho")
        if submitted:
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE option_trades
                        SET trade_date=%s, strategy=%s, status=%s, fees=%s, notes=%s
                        WHERE id=%s
                    """, (new_trade_date, new_strategy, new_status, new_fees, new_notes, trade_id))
                # direction pode ter mudado se editar legs depois; aqui só garante consistência
                recompute_and_update_direction(trade_id)
                st.success("Cabeçalho atualizado.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao atualizar cabeçalho: {e}")
            finally:
                conn.close()

# ---------- Editar um LEG ----------
with tab2:
    if not legs_db:
        st.warning("Esta operação não possui legs.")
    else:
        leg_choices = [
            f'LEG {l["leg_no"]} | {l["side"]} {l["opt_type"]} {l["option_symbol"]} exp={l["expiry"]}'
            for l in legs_db
        ]
        leg_idx = st.selectbox("Escolha o leg", options=list(range(len(leg_choices))), format_func=lambda i: leg_choices[i])
        leg = legs_db[leg_idx]

        with st.form(key=f"edit_leg_{trade_id}_{leg['id']}"):
            qty_shares_edit = st.number_input(
                "Quantidade (ações) — múltiplos de 100",
                min_value=100,
                step=100,
                value=int(leg["contracts"]) * 100
            )
            contracts_edit = int(qty_shares_edit // 100)

            entry_price_edit = st.number_input(
                "entry_price",
                min_value=0.0,
                step=0.01,
                value=float(leg["entry_price"])
            )

            exit_price_default = float(leg["exit_price"]) if leg["exit_price"] is not None else 0.0
            exit_price_edit = st.number_input(
                "exit_price (0 = vazio)",
                min_value=0.0,
                step=0.01,
                value=exit_price_default
            )
            set_exit_dt = st.checkbox("Definir exit_dt como agora", value=False)

            submitted = st.form_submit_button("Salvar alterações do LEG")
            if submitted:
                conn = get_conn()
                try:
                    with conn.cursor() as cur:
                        exit_price_sql = None if float(exit_price_edit) == 0.0 else float(exit_price_edit)
                        exit_dt_sql = datetime.now() if set_exit_dt else leg["exit_dt"]
                        cur.execute("""
                            UPDATE option_trade_legs
                            SET contracts=%s, entry_price=%s, exit_price=%s, exit_dt=%s
                            WHERE id=%s
                        """, (contracts_edit, float(entry_price_edit), exit_price_sql, exit_dt_sql, leg["id"]))

                    # direction pode mudar se entry_price mudar
                    recompute_and_update_direction(trade_id)

                    st.success("LEG atualizado.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao atualizar leg: {e}")
                finally:
                    conn.close()

# ---------- Fechar operação automaticamente ----------
with tab3:
    if not legs_db:
        st.warning("Sem legs para fechar.")
    else:
        st.info("Informe somente os exit_prices (>0). O sistema grava exit_dt=agora e seta status='CLOSED'.")
        with st.form(key=f"close_trade_{trade_id}"):
            exit_inputs = {}
            for l in legs_db:
                key = f"exit_{trade_id}_{l['id']}"
                default_val = float(l["exit_price"]) if l["exit_price"] is not None else 0.0
                exit_inputs[l["id"]] = st.number_input(
                    f'LEG {l["leg_no"]} — {l["side"]} {l["opt_type"]} {l["option_symbol"]} | exit_price',
                    min_value=0.0,
                    step=0.01,
                    value=default_val,
                    key=key
                )

            submitted = st.form_submit_button("Fechar operação agora")
            if submitted:
                missing = [lid for lid, v in exit_inputs.items() if float(v) <= 0.0]
                if missing:
                    st.error("Preencha exit_price (>0) para TODOS os legs antes de fechar.")
                else:
                    conn = get_conn()
                    try:
                        with conn.cursor() as cur:
                            now_ts = datetime.now()
                            for l in legs_db:
                                cur.execute("""
                                    UPDATE option_trade_legs
                                    SET exit_price=%s, exit_dt=%s
                                    WHERE id=%s
                                """, (float(exit_inputs[l["id"]]), now_ts, l["id"]))

                            cur.execute("""
                                UPDATE option_trades
                                SET status='CLOSED'
                                WHERE id=%s
                            """, (trade_id,))

                        # direction continua sendo o da entrada, mas mantemos consistente
                        recompute_and_update_direction(trade_id)

                        st.success("Operação fechada.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao fechar operação: {e}")
                    finally:
                        conn.close()

# ---------- Duplicar operação ----------
with tab4:
    st.info("Duplica o trade e os legs (mantém entry_price/side/etc.), zera exit e cria novo trade OPEN com trade_date=hoje.")
    new_trade_date = st.date_input("trade_date do novo trade", value=date.today(), key=f"dup_td_{trade_id}")
    new_notes = st.text_area("notes do novo trade (opcional)", value="", key=f"dup_notes_{trade_id}")

    confirm_dup = st.checkbox("Confirmo duplicar esta operação.", value=False, key=f"dup_confirm_{trade_id}")
    if st.button("Duplicar operação", type="primary", disabled=not confirm_dup):
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                # direction automático (entrada) baseado nos legs originais
                dir_dup = direction_from_legs(legs_db, multiplier=mult)

                # cria novo header (OPEN)
                cur.execute(
                    """
                    INSERT INTO option_trades
                      (trade_date, strategy, asset_id, direction, status, quantity_multiplier,
                       fees, notes, underlying_spot)
                    VALUES
                      (%s, %s, %s, %s, 'OPEN', %s,
                       %s, %s, %s)
                    """,
                    (
                        new_trade_date,
                        head_row["strategy"],
                        head_row["asset_id"],
                        dir_dup,
                        int(head_row.get("quantity_multiplier") or 100),
                        float(head_row.get("fees") or 0.0),
                        new_notes if new_notes.strip() else (head_row.get("notes") or None),
                        head_row.get("underlying_spot"),
                    )
                )
                new_trade_id = cur.lastrowid

                now_ts = datetime.now()
                for l in legs_db:
                    cur.execute(
                        """
                        INSERT INTO option_trade_legs
                          (trade_id, leg_no, side, opt_type, strike, expiry,
                           contracts, entry_price, entry_dt, option_symbol,
                           exit_price, exit_dt)
                        VALUES
                          (%s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s,
                           NULL, NULL)
                        """,
                        (
                            new_trade_id,
                            l["leg_no"],
                            l["side"],
                            l["opt_type"],
                            l["strike"],
                            l["expiry"],
                            l["contracts"],
                            l["entry_price"],
                            now_ts,
                            l["option_symbol"],
                        )
                    )

            # garante direction coerente (caso alguém mexa no futuro)
            recompute_and_update_direction(new_trade_id)

            st.success(f"Duplicado! novo trade_id={new_trade_id}")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao duplicar: {e}")
        finally:
            conn.close()

# ---------- Remover operação ----------
with tab5:
    st.warning("Remover apaga o trade e todos os legs (ON DELETE CASCADE).")
    confirm_del = st.checkbox("Confirmo que desejo remover esta operação.", value=False, key=f"del_confirm_{trade_id}")
    if st.button("REMOVER operação", type="primary", disabled=not confirm_del):
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM option_trades WHERE id=%s", (trade_id,))
            st.success("Operação removida.")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao remover operação: {e}")
        finally:
            conn.close()
