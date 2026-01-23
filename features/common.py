import numpy as np

def to_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None
