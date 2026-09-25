import pytest
from optlab.fills import synthetic_quote, fill_price, COMMISSION_PER_CONTRACT


def test_fill_prices_pay_quarter_spread():
    assert fill_price(1.00, 1.20, "sell") == pytest.approx(1.05)
    assert fill_price(1.00, 1.20, "buy") == pytest.approx(1.15)


def test_fill_rejects_bad_side():
    with pytest.raises(ValueError):
        fill_price(1, 2, "hold")


def test_synthetic_quote_5pct_spread_and_floor():
    bid, ask = synthetic_quote(2.00)
    assert (bid, ask) == pytest.approx((1.95, 2.05))
    bid, ask = synthetic_quote(0.10)
    assert ask - bid == pytest.approx(0.02)
    bid, _ = synthetic_quote(0.005)
    assert bid == 0.0


def test_commission():
    assert COMMISSION_PER_CONTRACT == 0.65
