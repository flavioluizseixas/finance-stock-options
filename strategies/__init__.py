from .single_legs import BuyDeepITMCall, SellPut, CoveredCall
from .straddle import LongStraddle
from .spreads import DebitSpreads, CreditSpreads
from .booster_puts import BoosterHorizontalPuts
from .volatility_trap import VolatilityTrapRatio

STRATEGIES = [
    BuyDeepITMCall(),
    SellPut(),
    CoveredCall(),
    LongStraddle(),
    DebitSpreads(),
    CreditSpreads(),
    BoosterHorizontalPuts(),
    VolatilityTrapRatio(),
]

def get_strategies():
    return STRATEGIES
