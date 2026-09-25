import pandas as pd
import pytest
from optlab.walkforward import split_windows, gate, WalkForwardReport, evaluate, report_to_dict
from optlab.strategy import Strategy
from tests.test_backtest import gbm


def rep(**kw):
    base = dict(window_returns=[0.02, 0.01, 0.03, -0.01], sortino=1.5, bh_sortino=1.0, mdd=0.10,
                bh_mdd=0.20, n_trades=40, win_rate=0.8, worst_per_1k=-60.0, total_return=0.05,
                trade_returns_per_1k=[5.0] * 40)
    base.update(kw)
    return WalkForwardReport(**base)


def test_split_windows_contiguous():
    idx = pd.bdate_range("2020-01-01", periods=100)
    w = split_windows(idx, 4)
    assert len(w) == 4 and w[0][0] == idx[0] and w[-1][1] == idx[-1]
    assert all(w[k][1] < w[k + 1][0] for k in range(3))


def test_gate_pass():
    assert gate(rep()) == (True, [])


@pytest.mark.parametrize("kw,needle", [
    (dict(window_returns=[0.01, -0.01, -0.02, 0.03]), "windows"),
    (dict(sortino=0.9), "sortino"),
    (dict(mdd=0.26), "drawdown"),
    (dict(n_trades=29), "trades"),
])
def test_gate_failures(kw, needle):
    ok, reasons = gate(rep(**kw))
    assert not ok and any(needle in r for r in reasons)


def test_evaluate_end_to_end():
    r = evaluate(Strategy("CSP", 0.25, 30), gbm(n=252 + 1000))
    assert len(r.window_returns) == 4 and r.n_trades > 0
    d = report_to_dict(r)
    assert set(d) >= {"window_returns", "sortino", "bh_sortino", "n_trades"}
