import math
import pytest
from optlab.pricing import bs_price, bs_delta, strike_step, strike_for_delta


def test_bs_textbook_values():
    assert bs_price(100, 100, 1.0, 0.05, 0.2, "call") == pytest.approx(10.4506, abs=1e-3)
    assert bs_price(100, 100, 1.0, 0.05, 0.2, "put") == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    c = bs_price(120, 110, 0.5, 0.03, 0.3, "call")
    p = bs_price(120, 110, 0.5, 0.03, 0.3, "put")
    assert c - p == pytest.approx(120 - 110 * math.exp(-0.03 * 0.5), abs=1e-9)


def test_expired_is_intrinsic():
    assert bs_price(105, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(5.0)
    assert bs_price(95, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(5.0)
    assert bs_price(95, 100, 0.0, 0.05, 0.2, "call") == 0.0


def test_delta_values_and_signs():
    assert bs_delta(100, 100, 1.0, 0.05, 0.2, "call") == pytest.approx(0.6368, abs=1e-3)
    assert bs_delta(100, 100, 1.0, 0.05, 0.2, "put") == pytest.approx(0.6368 - 1, abs=1e-3)


def test_strike_step():
    assert strike_step(40) == 1.0
    assert strike_step(150) == 2.5
    assert strike_step(450) == 5.0


def test_strike_for_delta_is_otm_and_close_to_target():
    k_put = strike_for_delta(100, 30 / 365, 0.04, 0.25, "put", 0.25)
    k_call = strike_for_delta(100, 30 / 365, 0.04, 0.25, "call", 0.25)
    assert k_put < 100 < k_call
    assert abs(abs(bs_delta(100, k_put, 30 / 365, 0.04, 0.25, "put")) - 0.25) < 0.06
    assert abs(bs_delta(100, k_call, 30 / 365, 0.04, 0.25, "call") - 0.25) < 0.06
