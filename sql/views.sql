USE finance_options;

CREATE OR REPLACE VIEW v_spot_by_date AS
SELECT
  a.ticker AS underlying,
  db.trade_date,
  db.close AS spot_close
FROM daily_bars db
JOIN assets a ON a.id = db.asset_id;

CREATE OR REPLACE VIEW v_option_lookup AS
SELECT
  oc.underlying AS underlying,
  oc.opt_type AS opt_type,           -- CALL/PUT
  oc.strike AS strike,
  oc.expiry AS expiry,
  oc.option_symbol AS option_symbol
FROM option_chain oc;

CREATE OR REPLACE VIEW v_option_lookup AS
SELECT
  a.ticker AS underlying,
  oc.opt_type,
  oc.strike,
  oc.expiry,
  oc.option_symbol
FROM option_chain oc
JOIN assets a ON a.id = oc.asset_id;

