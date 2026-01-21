from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from datetime import date

@dataclass
class StrategyResult:
    table: pd.DataFrame
    meta: Dict[str, Any] | None = None

class Strategy:
    key: str = "base"
    name: str = "Base"
    kind: str = "single"

    def candidates(self, df: pd.DataFrame, expiry_sel: Optional[date], cfg: Dict[str, Any], top_n: int) -> StrategyResult:
        raise NotImplementedError

    def payoff_spec(self, row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
