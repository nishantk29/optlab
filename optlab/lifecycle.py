"""Strategy status state machine: candidate -> backtest_pass -> paper -> promoted; benched; retired."""
from dataclasses import dataclass
import numpy as np

STATUSES = ("candidate", "backtest_pass", "paper", "promoted", "benched", "retired")


@dataclass
class PaperStats:
    weeks: int
    closed_trades: int
    realized_per_trade_1k: float
    paper_mdd: float


@dataclass
class BacktestBand:
    p10: float
    p90: float
    mdd: float


def after_backtest(status: str, passed: bool) -> str:
    if status != "candidate":
        raise ValueError(f"after_backtest from {status!r}")
    return "backtest_pass" if passed else "retired"


def start_paper(status: str) -> str:
    if status != "backtest_pass":
        raise ValueError(f"start_paper from {status!r}")
    return "paper"


def review(status: str, s: PaperStats, band: BacktestBand, benched_weeks: int = 0) -> str:
    if status == "benched":
        return "retired" if benched_weeks >= 8 else "benched"
    if status not in ("paper", "promoted"):
        raise ValueError(f"review from {status!r}")
    if s.paper_mdd > band.mdd or (s.weeks >= 8 and s.realized_per_trade_1k < band.p10):
        return "benched"
    if status == "paper" and s.weeks >= 8 and s.closed_trades >= 10 \
            and band.p10 <= s.realized_per_trade_1k <= band.p90:
        return "promoted"
    return status


def band_from_trade_returns(trade_returns_per_1k, mdd: float) -> BacktestBand:
    arr = np.asarray(trade_returns_per_1k, dtype=float)
    return BacktestBand(float(np.percentile(arr, 10)), float(np.percentile(arr, 90)), float(mdd))
