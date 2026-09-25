import math
from types import SimpleNamespace as T
import pandas as pd
import pytest
from optlab.metrics import sortino, max_drawdown, total_return, win_rate, worst_trade_per_1k, per_1k

IDX = pd.bdate_range("2024-01-01", periods=5)


def test_max_drawdown_and_total_return():
    eq = pd.Series([100, 120, 90, 110, 130], index=IDX, dtype=float)
    assert max_drawdown(eq) == pytest.approx(0.25)
    assert total_return(eq) == pytest.approx(0.30)


def test_sortino_all_up_is_inf_and_flat_is_zero():
    assert math.isinf(sortino(pd.Series([100, 101, 102, 103, 104], index=IDX, dtype=float)))
    assert sortino(pd.Series([100] * 5, index=IDX, dtype=float)) == 0.0


def test_sortino_sign():
    down = pd.Series([100, 99, 97, 98, 95], index=IDX, dtype=float)
    assert sortino(down) < 0


def test_trade_stats():
    trades = [T(pnl=50, collateral=5000), T(pnl=-200, collateral=5000), T(pnl=10, collateral=1000)]
    assert win_rate(trades) == pytest.approx(2 / 3)
    assert worst_trade_per_1k(trades) == pytest.approx(-40.0)
    assert win_rate([]) == 0.0


def test_per_1k():
    eq = pd.Series([2500.0, 2750.0], index=IDX[:2])
    assert list(per_1k(eq)) == [1000.0, 1100.0]
