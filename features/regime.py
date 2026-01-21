from .common import to_float

def infer_regime(ind: dict, rsi_hi: float = 55, rsi_lo: float = 45):
    if not ind:
        return "N/A", 0, 0, {}

    close = to_float(ind.get("close"))
    sma20 = to_float(ind.get("sma_20"))
    sma50 = to_float(ind.get("sma_50"))
    sma200 = to_float(ind.get("sma_200"))
    macdh = to_float(ind.get("macd_hist"))
    rsi = to_float(ind.get("rsi_14"))
    atr = to_float(ind.get("atr_14"))

    up = 0
    down = 0
    details = {}

    if (sma20 is not None) and (sma50 is not None) and (sma200 is not None):
        if sma20 > sma50 > sma200:
            up += 2
            details["SMA_stack"] = "Alta (SMA20>SMA50>SMA200)"
        elif sma20 < sma50 < sma200:
            down += 2
            details["SMA_stack"] = "Baixa (SMA20<SMA50<SMA200)"
        else:
            details["SMA_stack"] = "Misto"
    else:
        details["SMA_stack"] = "N/A"

    if (close is not None) and (sma50 is not None):
        if close > sma50:
            up += 1
            details["Close_vs_SMA50"] = "Acima"
        elif close < sma50:
            down += 1
            details["Close_vs_SMA50"] = "Abaixo"
        else:
            details["Close_vs_SMA50"] = "Em cima"
    else:
        details["Close_vs_SMA50"] = "N/A"

    if macdh is not None:
        if macdh > 0:
            up += 1
            details["MACD_hist"] = "Positivo"
        elif macdh < 0:
            down += 1
            details["MACD_hist"] = "Negativo"
        else:
            details["MACD_hist"] = "Zero"
    else:
        details["MACD_hist"] = "N/A"

    if rsi is not None:
        if rsi >= rsi_hi:
            up += 1
            details["RSI"] = f"Forte (≥{rsi_hi})"
        elif rsi <= rsi_lo:
            down += 1
            details["RSI"] = f"Fraco (≤{rsi_lo})"
        else:
            details["RSI"] = "Neutro"
    else:
        details["RSI"] = "N/A"

    details["ATR14"] = atr

    if up >= down + 2:
        return "Alta", up, down, details
    if down >= up + 2:
        return "Baixa", up, down, details
    return "Neutra", up, down, details
