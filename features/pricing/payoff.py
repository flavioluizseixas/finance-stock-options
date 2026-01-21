import numpy as np

def payoff_long_call(ST, K, premium):  return np.maximum(ST - K, 0.0) - premium
def payoff_short_call(ST, K, premium): return premium - np.maximum(ST - K, 0.0)
def payoff_long_put(ST, K, premium):   return np.maximum(K - ST, 0.0) - premium
def payoff_short_put(ST, K, premium):  return premium - np.maximum(K - ST, 0.0)
def payoff_stock(ST, S0):              return ST - S0

def payoff_strategy(op: dict, ST: np.ndarray, spot: float) -> np.ndarray:
    legs = op.get("legs", [])
    include_stock = bool(op.get("include_stock", False))
    stock_qty = float(op.get("stock_qty", 1.0))

    total = np.zeros_like(ST, dtype=float)
    if include_stock:
        total += stock_qty * payoff_stock(ST, spot)

    for lg in legs:
        opt_type = lg["type"]
        side = lg["side"]
        qty = float(lg.get("qty", 1.0))
        K = float(lg["K"])
        prem = float(lg["premium"])

        if opt_type == "CALL" and side == "LONG":
            total += qty * payoff_long_call(ST, K, prem)
        elif opt_type == "CALL" and side == "SHORT":
            total += qty * payoff_short_call(ST, K, prem)
        elif opt_type == "PUT" and side == "LONG":
            total += qty * payoff_long_put(ST, K, prem)
        elif opt_type == "PUT" and side == "SHORT":
            total += qty * payoff_short_put(ST, K, prem)

    return total

def break_evens(ST: np.ndarray, pnl: np.ndarray, tol: float = 1e-8):
    be = []
    y = pnl
    x = ST

    near = np.where(np.abs(y) < tol)[0]
    for i in near:
        be.append(float(x[i]))

    s = np.sign(y)
    for i in range(len(x) - 1):
        if s[i] == 0:
            continue
        if s[i] * s[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            xb = x0 - y0 * (x1 - x0) / (y1 - y0)
            be.append(float(xb))

    be = sorted(set([round(v, 6) for v in be]))
    return be

def minmax(pnl: np.ndarray):
    return float(np.min(pnl)), float(np.max(pnl))
