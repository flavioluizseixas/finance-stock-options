-- ============================================================
-- Sanity check (AGREGADO) - vencimento MAIS PRÓXIMO
-- MySQL 8+
-- ============================================================

WITH
-- ------------------------------------------------------------
-- Spot: último e penúltimo pregão (daily_bars)
-- ------------------------------------------------------------
bars_ranked AS (
  SELECT
    a.id AS asset_id,
    a.ticker,
    b.trade_date,
    b.close AS spot_close,
    ROW_NUMBER() OVER (
      PARTITION BY a.id
      ORDER BY b.trade_date DESC
    ) AS rn
  FROM assets a
  JOIN daily_bars b
    ON b.asset_id = a.id
  WHERE a.is_active = 1
),
bars_last2 AS (
  SELECT
    asset_id,
    ticker,
    MAX(CASE WHEN rn = 1 THEN trade_date END) AS bar_last_trade_date,
    MAX(CASE WHEN rn = 1 THEN spot_close END) AS bar_last_close,
    MAX(CASE WHEN rn = 2 THEN trade_date END) AS bar_prev_trade_date,
    MAX(CASE WHEN rn = 2 THEN spot_close END) AS bar_prev_close
  FROM bars_ranked
  WHERE rn IN (1, 2)
  GROUP BY asset_id, ticker
),

-- ------------------------------------------------------------
-- Último trade_date disponível no option_quote por ativo
-- (para limitar o "vencimento mais próximo" ao dia mais recente importado em opções)
-- ------------------------------------------------------------
opt_last_trade_date AS (
  SELECT
    asset_id,
    MAX(trade_date) AS last_opt_trade_date
  FROM option_quote
  GROUP BY asset_id
),

-- ------------------------------------------------------------
-- Vencimento MAIS PRÓXIMO (menor expiry_date >= last_opt_trade_date)
-- por ativo, considerando o último pregão de opções
-- ------------------------------------------------------------
nearest_expiry AS (
  SELECT
    oq.asset_id,
    MIN(oq.expiry_date) AS nearest_expiry_date
  FROM option_quote oq
  JOIN opt_last_trade_date ltd
    ON ltd.asset_id = oq.asset_id
   AND oq.trade_date = ltd.last_opt_trade_date
  WHERE oq.expiry_date >= oq.trade_date
  GROUP BY oq.asset_id
),

-- ------------------------------------------------------------
-- Agregado do vencimento mais próximo por dia (VWAP)
-- + collected_at (max) para sanity check de atualização
-- ------------------------------------------------------------
expiry_agg AS (
  SELECT
    a.ticker,
    oq.asset_id,
    oq.expiry_date,
    oq.trade_date,

    COUNT(DISTINCT oq.option_symbol) AS n_contracts,
    SUM(COALESCE(oq.volume, 0))      AS total_volume,
    SUM(COALESCE(oq.trades, 0))      AS total_trades,

    CASE
      WHEN SUM(COALESCE(oq.volume, 0)) = 0 THEN NULL
      ELSE
        SUM(COALESCE(oq.last_price, 0) * COALESCE(oq.volume, 0))
        / NULLIF(SUM(COALESCE(oq.volume, 0)), 0)
    END AS vwap_last_price,

    MAX(oq.collected_at) AS agg_collected_at,

    ROW_NUMBER() OVER (
      PARTITION BY oq.asset_id, oq.expiry_date
      ORDER BY oq.trade_date DESC
    ) AS rn_day
  FROM option_quote oq
  JOIN assets a
    ON a.id = oq.asset_id
  JOIN nearest_expiry ne
    ON ne.asset_id = oq.asset_id
   AND ne.nearest_expiry_date = oq.expiry_date
  WHERE a.is_active = 1
  GROUP BY
    a.ticker,
    oq.asset_id,
    oq.expiry_date,
    oq.trade_date
),

expiry_agg_last2 AS (
  SELECT
    asset_id,
    ticker,
    MAX(expiry_date) AS opt_nearest_expiry_date,

    MAX(CASE WHEN rn_day = 1 THEN trade_date END)        AS exp_last_trade_date,
    MAX(CASE WHEN rn_day = 1 THEN n_contracts END)      AS exp_last_n_contracts,
    MAX(CASE WHEN rn_day = 1 THEN total_volume END)     AS exp_last_total_volume,
    MAX(CASE WHEN rn_day = 1 THEN total_trades END)     AS exp_last_total_trades,
    MAX(CASE WHEN rn_day = 1 THEN vwap_last_price END)  AS exp_last_vwap_last_price,
    MAX(CASE WHEN rn_day = 1 THEN agg_collected_at END) AS exp_last_collected_at,

    MAX(CASE WHEN rn_day = 2 THEN trade_date END)        AS exp_prev_trade_date,
    MAX(CASE WHEN rn_day = 2 THEN n_contracts END)      AS exp_prev_n_contracts,
    MAX(CASE WHEN rn_day = 2 THEN total_volume END)     AS exp_prev_total_volume,
    MAX(CASE WHEN rn_day = 2 THEN total_trades END)     AS exp_prev_total_trades,
    MAX(CASE WHEN rn_day = 2 THEN vwap_last_price END)  AS exp_prev_vwap_last_price,
    MAX(CASE WHEN rn_day = 2 THEN agg_collected_at END) AS exp_prev_collected_at
  FROM expiry_agg
  WHERE rn_day IN (1, 2)
  GROUP BY asset_id, ticker
)

SELECT
  a.ticker,

  -- Spot
  b.bar_last_trade_date,
  b.bar_last_close,
  b.bar_prev_trade_date,
  b.bar_prev_close,

  -- Opções agregadas (vencimento mais próximo)
  e2.opt_nearest_expiry_date,

  e2.exp_last_trade_date,
  e2.exp_last_n_contracts,
  e2.exp_last_total_volume,
  e2.exp_last_total_trades,
  e2.exp_last_vwap_last_price,
  e2.exp_last_collected_at,

  e2.exp_prev_trade_date,
  e2.exp_prev_n_contracts,
  e2.exp_prev_total_volume,
  e2.exp_prev_total_trades,
  e2.exp_prev_vwap_last_price,
  e2.exp_prev_collected_at

FROM assets a
LEFT JOIN bars_last2 b
  ON b.asset_id = a.id
LEFT JOIN expiry_agg_last2 e2
  ON e2.asset_id = a.id
WHERE a.is_active = 1
ORDER BY a.ticker;
