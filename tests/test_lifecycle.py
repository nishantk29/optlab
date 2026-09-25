import pytest
from optlab.lifecycle import (after_backtest, start_paper, review, PaperStats, BacktestBand,
                              band_from_trade_returns, STATUSES)

BAND = BacktestBand(p10=-5.0, p90=15.0, mdd=0.10)


def test_backtest_transitions():
    assert after_backtest("candidate", True) == "backtest_pass"
    assert after_backtest("candidate", False) == "retired"
    with pytest.raises(ValueError):
        after_backtest("paper", True)


def test_start_paper():
    assert start_paper("backtest_pass") == "paper"
    with pytest.raises(ValueError):
        start_paper("candidate")


def test_promote():
    assert review("paper", PaperStats(8, 10, 5.0, 0.05), BAND) == "promoted"


def test_not_enough_evidence_stays_paper():
    assert review("paper", PaperStats(7, 20, 5.0, 0.05), BAND) == "paper"
    assert review("paper", PaperStats(10, 9, 5.0, 0.05), BAND) == "paper"


def test_demotions():
    assert review("paper", PaperStats(3, 4, 5.0, 0.11), BAND) == "benched"
    assert review("promoted", PaperStats(8, 12, -6.0, 0.05), BAND) == "benched"
    assert review("promoted", PaperStats(12, 20, 5.0, 0.05), BAND) == "promoted"


def test_retire_after_8_benched_weeks():
    assert review("benched", PaperStats(0, 0, 0, 0), BAND, benched_weeks=8) == "retired"
    assert review("benched", PaperStats(0, 0, 0, 0), BAND, benched_weeks=7) == "benched"


def test_band():
    b = band_from_trade_returns(list(range(101)), 0.2, window=1)
    assert (b.p10, b.p90, b.mdd) == (10.0, 90.0, 0.2)


def test_band_uses_rolling_averages_so_it_is_narrower():
    import numpy as np
    rets = list(np.random.default_rng(0).normal(5, 40, 300))
    single = band_from_trade_returns(rets, 0.2, window=1)
    avg10 = band_from_trade_returns(rets, 0.2)
    assert avg10.p90 - avg10.p10 < (single.p90 - single.p10) / 2
    assert set(STATUSES) == {"candidate", "backtest_pass", "paper", "promoted", "benched", "retired"}
