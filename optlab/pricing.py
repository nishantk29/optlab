"""Black-Scholes pricing helpers (European, continuous dividend yield q)."""
import math
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> float:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be call/put, got {kind!r}")
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        itm = S > K if kind == "call" else S < K
        return (1.0 if kind == "call" else -1.0) if itm else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return math.exp(-q * T) * norm.cdf(d1)
    return -math.exp(-q * T) * norm.cdf(-d1)


def strike_step(S: float) -> float:
    """Listed-strike grid approximation."""
    if S < 100:
        return 1.0
    if S < 250:
        return 2.5
    return 5.0


def strike_for_delta(S: float, T: float, r: float, sigma: float, kind: str, target: float) -> float:
    """Grid strike whose |delta| is closest to `target`, restricted to OTM strikes."""
    step = strike_step(S)
    base = round(S / step) * step
    if kind == "put":
        grid = [base - i * step for i in range(0, int(0.5 * S / step) + 1) if base - i * step < S]
    else:
        grid = [base + i * step for i in range(0, int(0.5 * S / step) + 1) if base + i * step > S]
    grid = [k for k in grid if k > 0]
    return min(grid, key=lambda k: abs(abs(bs_delta(S, k, T, r, sigma, kind)) - target))
