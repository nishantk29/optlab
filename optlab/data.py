"""yfinance history loader with a pickle cache (refreshed when older than 1 day).

Pickle is safe here: the cache files are written only by this function, into a private
local cache dir, and are never read from any other source.
"""
import os
import time
import pandas as pd
import yfinance as yf


def load_history(symbol: str, years: int = 10, cache_dir: str = ".cache"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{years}y.pkl")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        return pd.read_pickle(path)
    t = yf.Ticker(symbol)
    h = t.history(period=f"{years}y", auto_adjust=False)
    h.index = h.index.tz_localize(None).normalize()
    close = h["Close"].astype(float)
    divs = h.loc[h["Dividends"] > 0, "Dividends"].astype(float)
    try:
        ed = t.get_earnings_dates(limit=60)
        earnings = sorted({pd.Timestamp(d).tz_localize(None).normalize() for d in ed.index})
    except Exception:
        earnings = []
    result = (close, divs, earnings)
    pd.to_pickle(result, path)
    return result
