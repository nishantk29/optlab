import pytest
from optlab.strategy import Strategy


def test_valid_and_stable_id():
    a = Strategy("CSP", 0.25, 30)
    b = Strategy.from_dict(a.to_dict())
    assert a == b and a.id == b.id and len(a.id) == 10


def test_id_changes_with_params():
    assert Strategy("CSP", 0.25, 30).id != Strategy("CSP", 0.30, 30).id


@pytest.mark.parametrize("kwargs", [
    dict(type="PUT", delta_target=0.25, dte=30),
    dict(type="CSP", delta_target=0.50, dte=30),
    dict(type="CSP", delta_target=0.25, dte=60),
    dict(type="CSP", delta_target=0.25, dte=30, take_profit=1.5),
    dict(type="CSP", delta_target=0.25, dte=30, roll_at_dte=40),
    dict(type="CSP", delta_target=0.25, dte=30, ivr_min=150),
    dict(type="CSP", delta_target=0.25, dte=30, hold_to_expiry=True),  # roll_at_dte must be None
])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        Strategy(**kwargs)


def test_hold_to_expiry_valid_without_roll():
    s = Strategy("CC", 0.2, 45, take_profit=None, roll_at_dte=None, hold_to_expiry=True)
    assert s.hold_to_expiry


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Strategy.from_dict({"type": "CSP", "delta_target": 0.25, "dte": 30, "leverage": 3})
