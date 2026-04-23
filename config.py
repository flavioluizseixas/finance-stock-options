import os

from parquet_store import env_paths as parquet_env_paths

DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "3"))

DEFAULT_CFG = {
    # Liquidez — single
    "liq_min_trades": float(os.getenv("LIQ_MIN_TRADES", "3")),
    "liq_min_volume": float(os.getenv("LIQ_MIN_VOLUME", "100")),
    "liq_min_ok": float(os.getenv("LIQ_MIN_OK", "4.0")),
    "liq_min_alert": float(os.getenv("LIQ_MIN_ALERT", "2.5")),
    "liq_penalty_alert": float(os.getenv("LIQ_PENALTY_ALERT", "0.25")),
    "liq_penalty_bad": float(os.getenv("LIQ_PENALTY_BAD", "0.75")),
    "liq_single_filter_hard": False,

    # Liquidez — pair
    "liq_pair_min_ok": float(os.getenv("LIQ_PAIR_MIN_OK", "6.0")),
    "liq_pair_min_alert": float(os.getenv("LIQ_PAIR_MIN_ALERT", "4.0")),
    "liq_pair_ratio_ok": float(os.getenv("LIQ_PAIR_RATIO_OK", "0.40")),
    "liq_pair_ratio_alert": float(os.getenv("LIQ_PAIR_RATIO_ALERT", "0.20")),
    "liq_pair_filter_hard": False,
    "liq_pair_hard_min": 0.0,
    "liq_pair_hard_ratio": 0.0,

    # Payoff
    "payoff_default_lo_mult": float(os.getenv("PAYOFF_LO_MULT", "0.5")),
    "payoff_default_hi_mult": float(os.getenv("PAYOFF_HI_MULT", "1.5")),
}

def env_paths(base_dir: str):
    return parquet_env_paths(base_dir)
