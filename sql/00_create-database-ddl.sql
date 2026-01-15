-- 00_reset_finance_options.sql
DROP DATABASE IF EXISTS finance_options;

CREATE DATABASE finance_options
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE finance_options;

CREATE TABLE assets (
  id INT NOT NULL AUTO_INCREMENT,
  ticker VARCHAR(32) NOT NULL,
  is_active TINYINT NOT NULL DEFAULT 1,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_assets_ticker (ticker)
) ENGINE=InnoDB;

CREATE TABLE daily_bars (
  asset_id INT NOT NULL,
  trade_date DATE NOT NULL,

  open DOUBLE NULL,
  high DOUBLE NULL,
  low DOUBLE NULL,
  close DOUBLE NULL,
  adj_close DOUBLE NULL,
  volume BIGINT NULL,

  log_ret DOUBLE NULL,
  vol_annual DOUBLE NULL,

  sma_20 DOUBLE NULL,
  sma_50 DOUBLE NULL,
  sma_200 DOUBLE NULL,

  ema_12 DOUBLE NULL,
  ema_26 DOUBLE NULL,
  macd DOUBLE NULL,
  macd_signal DOUBLE NULL,
  macd_hist DOUBLE NULL,

  rsi_14 DOUBLE NULL,
  atr_14 DOUBLE NULL,

  collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (asset_id, trade_date),
  CONSTRAINT fk_daily_bars_asset FOREIGN KEY (asset_id) REFERENCES assets(id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  KEY ix_daily_bars_trade_date (trade_date)
) ENGINE=InnoDB;

CREATE TABLE option_chain (
  asset_id INT NOT NULL,
  trade_date DATE NOT NULL,
  expiry_date DATE NOT NULL,
  collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (asset_id, trade_date, expiry_date),
  CONSTRAINT fk_option_chain_asset FOREIGN KEY (asset_id) REFERENCES assets(id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  KEY ix_chain_expiry (expiry_date),
  KEY ix_chain_trade_date (trade_date)
) ENGINE=InnoDB;

CREATE TABLE option_quote (
  asset_id INT NOT NULL,
  trade_date DATE NOT NULL,

  expiry_date DATE NOT NULL,
  option_symbol VARCHAR(32) NOT NULL,

  option_type ENUM('CALL','PUT') NOT NULL,
  model_code VARCHAR(8) NULL,

  strike DOUBLE NOT NULL,

  last_price DOUBLE NULL,
  trades INT NULL,
  volume BIGINT NULL,

  collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (asset_id, trade_date, option_symbol),
  CONSTRAINT fk_option_quote_asset FOREIGN KEY (asset_id) REFERENCES assets(id)
    ON DELETE CASCADE ON UPDATE CASCADE,

  KEY ix_quote_expiry_type_strike (expiry_date, option_type, strike),
  KEY ix_quote_trade_date (trade_date),
  KEY ix_quote_liquidity (trades, volume)
) ENGINE=InnoDB;

-- Curva de juros por vértice (dias úteis) por pregão
CREATE TABLE yield_curve (
  trade_date DATE NOT NULL,
  vertex_bd INT NOT NULL,        -- dias úteis até o vértice
  rate DOUBLE NOT NULL,          -- taxa a.a em decimal (ex.: 0.1123)
  source VARCHAR(16) NOT NULL,   -- 'BCB' | 'B3_FILE'
  collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (trade_date, vertex_bd),
  KEY ix_yield_curve_trade_date (trade_date)
) ENGINE=InnoDB;

CREATE TABLE option_model (
  asset_id INT NOT NULL,
  trade_date DATE NOT NULL,
  option_symbol VARCHAR(32) NOT NULL,

  spot DOUBLE NULL,
  rate_r DOUBLE NULL,        -- agora é r interpolado pela curva para esse vencimento
  dividend_q DOUBLE NULL,
  t_years DOUBLE NULL,

  iv DOUBLE NULL,

  bsm_price DOUBLE NULL,
  bsm_price_histvol DOUBLE NULL,

  mispricing DOUBLE NULL,
  mispricing_pct DOUBLE NULL,

  delta DOUBLE NULL,
  gamma DOUBLE NULL,
  vega DOUBLE NULL,
  theta DOUBLE NULL,         -- THETA ANUAL no DB
  rho DOUBLE NULL,

  hist_vol_annual DOUBLE NULL,

  collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (asset_id, trade_date, option_symbol),
  CONSTRAINT fk_option_model_asset FOREIGN KEY (asset_id) REFERENCES assets(id)
    ON DELETE CASCADE ON UPDATE CASCADE,

  KEY ix_model_trade_date (trade_date),
  KEY ix_model_iv (iv),
  KEY ix_model_mispricing (mispricing),
  KEY ix_model_mispricing_pct (mispricing_pct)
) ENGINE=InnoDB;
