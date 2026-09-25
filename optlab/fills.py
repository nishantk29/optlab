"""Conservative fill model: sellers get mid - 25% of spread, buyers pay mid + 25%."""
COMMISSION_PER_CONTRACT = 0.65
SYNTHETIC_SPREAD_PCT = 0.05


def synthetic_quote(theo: float) -> tuple[float, float]:
    half = max(theo * SYNTHETIC_SPREAD_PCT / 2, 0.01)
    return max(theo - half, 0.0), theo + half


def fill_price(bid: float, ask: float, side: str) -> float:
    mid, spread = (bid + ask) / 2, ask - bid
    if side == "sell":
        return mid - 0.25 * spread
    if side == "buy":
        return mid + 0.25 * spread
    raise ValueError(f"side must be sell/buy, got {side!r}")
