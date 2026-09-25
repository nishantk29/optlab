"""Performance metrics. All equity inputs are pd.Series of account value over time."""
import math
import numpy as np
import pandas as pd


def per_1k(equity: pd.Series) -> pd.Series:
    return equity / equity.iloc[0] * 1000.0


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def max_drawdown(equity: pd.Series) -> float:
    return float(-(equity / equity.cummax() - 1).min())


def sortino(equity: pd.Series, periods: int = 252) -> float:
    r = equity.pct_change().dropna()
    if r.empty or (r == 0).all():
        return 0.0
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    mean = r.mean()
    if downside == 0:
        return math.inf if mean > 0 else 0.0
    return float(mean * periods / (downside * math.sqrt(periods)))


def win_rate(trades) -> float:
    trades = list(trades)
    return sum(t.pnl > 0 for t in trades) / len(trades) if trades else 0.0


def worst_trade_per_1k(trades) -> float:
    trades = list(trades)
    return min(t.pnl / t.collateral * 1000 for t in trades) if trades else 0.0
