DEFAULT_TOP_N = 5

DEFAULT_CFG = {
    # --- deltas / bandas ---
    "deep_itm_delta_up": 0.75,
    "deep_itm_delta_neutral": 0.80,
    "deep_itm_delta_down": 0.85,

    "put_delta_lo": -0.35,
    "put_delta_hi": -0.10,
    "put_delta_lo_bear": -0.25,
    "put_delta_hi_bear": -0.08,
    "put_min_otm_dist_bear": 0.015,

    "cc_delta_up": (0.12, 0.25),
    "cc_delta_neutral": (0.15, 0.30),
    "cc_delta_down": (0.20, 0.40),

    # --- credit spreads delta bands ---
    "bps_put_sell_band": (-0.30, -0.15),
    "bps_put_buy_band":  (-0.15, -0.05),
    "bps_put_sell_band_down": (-0.25, -0.12),
    "bps_put_buy_band_down":  (-0.12, -0.04),

    "bcs_call_sell_band": (0.15, 0.30),
    "bcs_call_buy_band":  (0.05, 0.15),
    "bcs_call_sell_band_up": (0.12, 0.22),
    "bcs_call_buy_band_up":  (0.04, 0.10),

    # --- liquidez (perna única) ---
    "liq_min_trades": 5,
    "liq_min_volume": 50,
    "liq_min_ok": 8.0,
    "liq_min_alert": 5.0,
    "liq_penalty_alert": 1.0,
    "liq_penalty_bad": 3.0,
    "liq_single_filter_hard": True,

    # --- liquidez (par) ---
    "liq_pair_min_ok": 6.0,
    "liq_pair_min_alert": 4.0,
    "liq_pair_ratio_ok": 0.40,
    "liq_pair_ratio_alert": 0.25,
    "liq_pair_hard_min": 3.0,
    "liq_pair_hard_ratio": 0.20,

    # --- Booster horizontal ---
    # 2 ou 3 vencimentos à frente
    "booster_long_steps": 2,     # altere para 3 se quiser
    "booster_put_delta_target": -0.25,
    "booster_put_delta_band": 0.10,   # aceita deltas em [target-band, target+band]
    "booster_use_same_strike": True,  # se False, permite pequena diferença de strike
    "booster_max_strike_gap_pct": 0.02,
}
