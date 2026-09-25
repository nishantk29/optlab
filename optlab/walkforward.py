"""Walk-forward evaluation.

A variant is a fixed parameter record, so it is simulated once over the full history and its
equity/trades are sliced into n sequential out-of-sample windows. "In-sample fitting" is the
selection among variants done by the weekly proposal loop (Plan 2).
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from . import metrics
from .backtest import run_backtest


@dataclass
class WalkForwardReport:
    window_returns: list[float]
    sortino: float
    bh_sortino: float
    mdd: float
    bh_mdd: float
    n_trades: int
    win_rate: float
    worst_per_1k: float
    total_return: float
    trade_returns_per_1k: list[float]


def split_windows(index: pd.DatetimeIndex, n: int = 4):
    parts = np.array_split(np.arange(len(index)), n)
    return [(index[p[0]], index[p[-1]]) for p in parts]


def evaluate(strategy, close, dividends=None, earnings=(), r=0.04) -> WalkForwardReport:
    res = run_backtest(strategy, close, dividends, earnings, r)
    eq = res.equity
    bh = close.loc[eq.index].astype(float)
    wins = []
    for a, b in split_windows(eq.index, 4):
        seg = eq.loc[a:b]
        wins.append(float(seg.iloc[-1] / seg.iloc[0] - 1))
    return WalkForwardReport(
        window_returns=wins,
        sortino=metrics.sortino(eq), bh_sortino=metrics.sortino(bh),
        mdd=metrics.max_drawdown(eq), bh_mdd=metrics.max_drawdown(bh),
        n_trades=len(res.trades), win_rate=metrics.win_rate(res.trades),
        worst_per_1k=metrics.worst_trade_per_1k(res.trades),
        total_return=metrics.total_return(eq),
        trade_returns_per_1k=[float(t.pnl / t.collateral * 1000) for t in res.trades],
    )


def gate(rep: WalkForwardReport) -> tuple[bool, list[str]]:
    reasons = []
    if sum(w > 0 for w in rep.window_returns) < 3:
        reasons.append("profitable in fewer than 3 of 4 windows")
    if not rep.sortino > rep.bh_sortino:
        reasons.append("sortino not above buy-and-hold")
    if rep.mdd > 1.25 * rep.bh_mdd:
        reasons.append("max drawdown above 1.25x underlying")
    if rep.n_trades < 30:
        reasons.append("fewer than 30 trades")
    return (not reasons, reasons)


def report_to_dict(rep: WalkForwardReport) -> dict:
    d = asdict(rep)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}
