from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd

@dataclass
class StrategyResult:
    """Result object returned by a strategy candidate generator.

    Older versions of this project returned only the candidate table.
    We extend it to also carry criteria bullets/notes, but keep backwards
    compatibility so existing code can still do StrategyResult(table).
    """

    table: pd.DataFrame
    criteria: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

@dataclass
class Strategy:
    key: str
    name: str
    candidates: Callable
    payoff_spec: Callable
