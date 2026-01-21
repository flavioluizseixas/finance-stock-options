from datetime import date
import pandas as pd
import streamlit as st
from .conn import get_conn

@st.cache_data(ttl=60)
def load_assets():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, ticker FROM assets WHERE is_active=1 ORDER BY ticker;")
            rows = cur.fetchall()
    return pd.DataFrame(rows)

@st.cache_data(ttl=60)
def load_latest_trade_date(asset_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(trade_date) AS trade_date FROM option_quote WHERE asset_id=%s", (asset_id,))
            row = cur.fetchone()
    return row["trade_date"] if row else None

@st.cache_data(ttl=60)
def load_daily_indicators(asset_id: int, trade_date: date):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT
                  trade_date,
                  close,
                  vol_annual,
                  sma_20, sma_50, sma_200,
                  macd, macd_signal, macd_hist,
                  rsi_14,
                  atr_14
                FROM daily_bars
                WHERE asset_id=%s AND trade_date=%s
                ''',
                (asset_id, trade_date),
            )
            row = cur.fetchone()
    return row

@st.cache_data(ttl=60)
def load_chain(asset_id: int, trade_date: date):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT
                  oq.option_symbol,
                  oq.option_type,
                  oq.expiry_date,
                  oq.strike,
                  oq.last_price,
                  oq.trades,
                  oq.volume,
                  oq.collected_at AS quote_collected_at,

                  om.spot,
                  om.rate_r,
                  om.dividend_q,
                  om.t_years,
                  om.iv,
                  om.bsm_price,
                  om.bsm_price_histvol,
                  om.mispricing,
                  om.mispricing_pct,
                  om.delta,
                  om.gamma,
                  om.vega,
                  om.theta,
                  om.rho,
                  om.hist_vol_annual,
                  om.collected_at AS model_collected_at
                FROM option_quote oq
                LEFT JOIN option_model om
                  ON om.asset_id=oq.asset_id
                 AND om.trade_date=oq.trade_date
                 AND om.option_symbol=oq.option_symbol
                WHERE oq.asset_id=%s
                  AND oq.trade_date=%s
                  AND oq.trades > 0
                  AND oq.last_price > 0
                ORDER BY oq.expiry_date, oq.option_type, oq.strike;
                ''',
                (asset_id, trade_date),
            )
            rows = cur.fetchall() or []
    return pd.DataFrame(rows)
