USE finance_options;

CREATE TABLE IF NOT EXISTS option_trades (
  id BIGINT NOT NULL AUTO_INCREMENT,

  trade_date DATE NOT NULL,                 -- data da operação (entrada)
  asof_trade_date DATE NULL,                -- data do pregão usado p/ listar opções (snapshot)

  strategy VARCHAR(32) NOT NULL,            -- SELL_PUT, VERTICAL_SPREAD, ...
  asset_id INT NOT NULL,                    -- FK assets.id
  direction VARCHAR(8) NOT NULL,            -- DEBIT | CREDIT (ou compute depois)
  status VARCHAR(16) NOT NULL DEFAULT 'OPEN',

  quantity_multiplier INT NOT NULL DEFAULT 100,
  fees DECIMAL(18,6) NOT NULL DEFAULT 0,
  notes TEXT NULL,

  underlying_spot DECIMAL(18,6) NULL,
  thesis VARCHAR(255) NULL,
  tags VARCHAR(255) NULL,

  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (id),
  CONSTRAINT fk_trades_asset FOREIGN KEY (asset_id) REFERENCES assets(id)
    ON DELETE RESTRICT ON UPDATE CASCADE,

  INDEX ix_trades_asset_date (asset_id, trade_date),
  INDEX ix_trades_status (status),
  INDEX ix_trades_strategy (strategy)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS option_trade_legs (
  id BIGINT NOT NULL AUTO_INCREMENT,
  trade_id BIGINT NOT NULL,

  leg_no INT NOT NULL,
  side VARCHAR(4) NOT NULL,     -- BUY | SELL
  opt_type VARCHAR(4) NOT NULL, -- CALL | PUT

  expiry DATE NOT NULL,
  strike DECIMAL(18,6) NOT NULL,
  option_symbol VARCHAR(64) NOT NULL,       -- torna obrigatório (recomendado)

  contracts INT NOT NULL,
  entry_price DECIMAL(18,6) NOT NULL,
  exit_price DECIMAL(18,6) NULL,

  entry_dt TIMESTAMP NULL,
  exit_dt TIMESTAMP NULL,

  ref_last_price DECIMAL(18,6) NULL,        -- snapshot opcional do last_price exibido
  ref_collected_at TIMESTAMP NULL,          -- snapshot opcional do collected_at exibido

  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (id),

  CONSTRAINT fk_trade_legs_trade
    FOREIGN KEY (trade_id) REFERENCES option_trades(id)
    ON DELETE CASCADE,

  UNIQUE KEY uq_trade_legno (trade_id, leg_no),
  INDEX ix_legs_expiry (expiry),
  INDEX ix_legs_symbol (option_symbol)
) ENGINE=InnoDB;