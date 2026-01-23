import numpy as np
import pandas as pd

def liquidity_score_df(d: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(d.get("trades", 0), errors="coerce").fillna(0)
    v = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)
    return t + np.log1p(v)
