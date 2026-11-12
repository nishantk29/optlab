"""Standard starting variants (22). Later proposals add more by changing values only."""
from .strategy import Strategy

TP_ROLL = dict(take_profit=0.5, roll_at_dte=21, hold_to_expiry=False)
HOLD = dict(take_profit=None, roll_at_dte=None, hold_to_expiry=True)


def standard_variants() -> list[Strategy]:
    out = []
    for d in (0.20, 0.30):
        for dte in (30, 45):
            for mgmt in (TP_ROLL, HOLD):
                out.append(Strategy("CSP", d, dte, trend_filter=True, **mgmt))
    out += [Strategy("CSP", 0.25, 30, ivr_min=30, trend_filter=True, **TP_ROLL),
            Strategy("CSP", 0.25, 45, ivr_min=30, trend_filter=True, **TP_ROLL)]
    for d in (0.15, 0.25, 0.30):
        for dte in (30, 45):
            for mgmt in (TP_ROLL, HOLD):
                out.append(Strategy("CC", d, dte, **mgmt))
    return out
