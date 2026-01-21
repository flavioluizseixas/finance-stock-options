import numpy as np
import pandas as pd

def to_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None

def ensure_date(x):
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return x
