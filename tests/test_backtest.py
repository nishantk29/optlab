import numpy as np
import pandas as pd
import pytest
from optlab.backtest import run_backtest, vol_proxy, vol_rank, WARMUP
from optlab.strategy import Strategy


def gbm(n=WARMUP + 400, mu=0.08, sigma=0.2, seed=7, s0=100.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal((mu - 0.5 * sigma**2) / 252, sigma / np.sqrt(252), n)
    return pd.Series(s0 * np.exp(np.cumsum(rets)), index=pd.bdate_range("2015-01-01", periods=n))


def test_vol_proxy_is_1_1x_realized():
    c = gbm()
    rv = np.log(c).diff().rolling(30).std() * np.sqrt(252)
    pd.testing.assert_series_equal(vol_proxy(c), 1.1 * rv)


def test_vol_rank_bounds():
    vr = vol_rank(gbm()).dropna()
    assert vr.between(0, 100).all()


def test_csp_runs_and_trades():
    res = run_backtest(Strategy("CSP", 0.25, 30), gbm())
    assert len(res.trades) >= 5
    assert res.equity.index[0] == gbm().index[WARMUP]
    assert all(t.kind == "put" and t.premium > 0 for t in res.trades)


def test_csp_assignment_loss_on_crash():
    c = gbm(sigma=0.15, mu=0.05)
    c.iloc[WARMUP + 10:] = c.iloc[WARMUP + 10:] * 0.6   # 40% gap down after entry
    res = run_backtest(Strategy("CSP", 0.30, 30, take_profit=None, roll_at_dte=None, hold_to_expiry=True), c)
    assert res.trades[0].reason == "assigned" and res.trades[0].pnl < 0


def test_take_profit_closes_early():
    c = pd.Series(100.0, index=pd.bdate_range("2015-01-01", periods=WARMUP + 200))
    c += np.random.default_rng(1).normal(0, 0.8, len(c))   # vol present, no trend
    res = run_backtest(Strategy("CSP", 0.25, 45, take_profit=0.5, roll_at_dte=None), c)
    assert any(t.reason == "take_profit" for t in res.trades)


def test_roll_at_dte():
    res = run_backtest(Strategy("CSP", 0.25, 45, take_profit=None, roll_at_dte=21), gbm())
    assert any(t.reason == "roll" for t in res.trades)


def test_cc_called_away_on_rally():
    c = gbm(mu=0.05)
    c.iloc[WARMUP + 5:] = c.iloc[WARMUP + 5:] * 1.3
    res = run_backtest(Strategy("CC", 0.20, 30, take_profit=None, roll_at_dte=None, hold_to_expiry=True), c)
    assert res.trades[0].reason == "called_away"


def test_cc_early_assignment_before_dividend():
    c = gbm(mu=0.05)
    c.iloc[WARMUP + 3:] = c.iloc[WARMUP + 3:] * 1.25
    ex = c.index[WARMUP + 8]
    divs = pd.Series({ex: 5.0})
    res = run_backtest(Strategy("CC", 0.30, 30, take_profit=None, roll_at_dte=None, hold_to_expiry=True), c, dividends=divs)
    assert res.trades[0].reason == "early_assigned"


def test_earnings_blocks_entries_spanning_earnings():
    c = gbm()
    earn = [c.index[WARMUP + 12]]
    res = run_backtest(Strategy("CSP", 0.25, 30, take_profit=None, roll_at_dte=None), c, earnings=earn)
    assert res.trades[0].open_date >= earn[0]
    assert all(not (t.open_date < earn[0] <= t.close_date) for t in res.trades)


def test_earnings_ignored_when_flag_off():
    c = gbm()
    earn = [c.index[WARMUP + 12]]
    res = run_backtest(Strategy("CSP", 0.25, 30, take_profit=None, roll_at_dte=None,
                                close_before_earnings=False), c, earnings=earn)
    assert res.trades[0].open_date == c.index[WARMUP]


def test_cc_dividends_are_paid_to_shareholders():
    c = gbm(sigma=0.1, mu=0.0, seed=21)
    ex = c.index[WARMUP + 5]
    s = Strategy("CC", 0.15, 45, take_profit=None, roll_at_dte=None, hold_to_expiry=True)
    without = run_backtest(s, c)
    with_div = run_backtest(s, c, dividends=pd.Series({ex: 1.0}))
    assert with_div.trades[0].reason != "early_assigned"
    assert with_div.equity[ex] - without.equity[ex] == pytest.approx(100.0)   # 100 shares x $1


def test_expiry_settles_on_or_before_nominal_date():
    s = Strategy("CSP", 0.25, 30, take_profit=None, roll_at_dte=None, hold_to_expiry=True)
    res = run_backtest(s, gbm(n=WARMUP + 800))
    settled = [t for t in res.trades if t.reason in ("expired", "assigned")]
    assert settled
    assert all(t.close_date <= t.open_date + pd.Timedelta(days=30) for t in settled)
    assert all(t.close_date.weekday() < 5 for t in settled)


def test_trend_filter_blocks_csp_in_downtrend():
    c = gbm(mu=-0.6, sigma=0.15)
    res = run_backtest(Strategy("CSP", 0.25, 30, trend_filter=True), c)
    assert len(res.trades) <= 2


def test_ivr_min_blocks_entries():
    # IV rank hits 100 on genuine new vol highs, so ivr_min=100 still allows rare entries;
    # the filter must cut entries sharply versus no filter.
    unfiltered = len(run_backtest(Strategy("CSP", 0.25, 30), gbm(n=WARMUP + 1000)).trades)
    filtered = len(run_backtest(Strategy("CSP", 0.25, 30, ivr_min=100), gbm(n=WARMUP + 1000)).trades)
    assert filtered < unfiltered / 3


def test_commission_charged():
    res = run_backtest(Strategy("CSP", 0.25, 30, take_profit=None, roll_at_dte=None, hold_to_expiry=True), gbm())
    t = res.trades[0]
    if t.reason == "expired":
        assert t.pnl == pytest.approx(t.qty * (t.premium * 100 - 0.65))


def test_csp_keeps_trading_as_underlying_triples():
    # Regression: fixed one-contract sizing starved CSPs of collateral once the stock rose.
    c = gbm(n=WARMUP + 2000, mu=0.25, sigma=0.2, seed=11)
    res = run_backtest(Strategy("CSP", 0.25, 30), c)
    late = [t for t in res.trades if t.open_date > c.index[WARMUP + 1500]]
    assert len(late) >= 5


def test_cc_rebuys_after_called_away_in_rally():
    c = gbm(n=WARMUP + 1500, mu=0.3, sigma=0.2, seed=12)
    res = run_backtest(Strategy("CC", 0.30, 30, take_profit=None, roll_at_dte=None, hold_to_expiry=True), c)
    assert any(t.reason == "called_away" for t in res.trades)
    late = [t for t in res.trades if t.open_date > c.index[WARMUP + 1000]]
    assert len(late) >= 5


def test_position_scales_with_capital():
    res = run_backtest(Strategy("CSP", 0.25, 30), gbm())
    assert all(t.qty > 0 for t in res.trades)
    assert all(t.collateral == pytest.approx(t.strike * 100 * t.qty) for t in res.trades)
