"""Daily simulation engine for covered calls (CC) and cash-secured puts (CSP), synthetic pricing.

Simplifications (per plan): CSP assignment is settled as shares sold at that day's close;
CC holds 100 shares and re-buys them at the next eligible entry after being called away.
"""
from dataclasses import dataclass
from typing import Iterable
import math
import numpy as np
import pandas as pd

from .fills import COMMISSION_PER_CONTRACT as COMM, fill_price, synthetic_quote
from .pricing import bs_price, strike_for_delta
from .strategy import Strategy

WARMUP = 252
EARNINGS_BUFFER_DAYS = 3
MIN_PREMIUM = 0.05


@dataclass
class Trade:
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    kind: str
    strike: float
    premium: float
    close_cost: float
    pnl: float
    collateral: float
    reason: str


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series


def vol_proxy(close: pd.Series) -> pd.Series:
    return 1.1 * (np.log(close).diff().rolling(30).std() * np.sqrt(252))


def vol_rank(close: pd.Series) -> pd.Series:
    rv = np.log(close).diff().rolling(30).std() * np.sqrt(252)
    lo, hi = rv.rolling(252, min_periods=60).min(), rv.rolling(252, min_periods=60).max()
    return ((rv - lo) / (hi - lo).replace(0, np.nan) * 100).clip(0, 100)


def _earnings_within(date, horizon_days, earnings) -> bool:
    end = date + pd.Timedelta(days=horizon_days)
    return any(date < e <= end for e in earnings)


def run_backtest(strategy: Strategy, close: pd.Series, dividends: pd.Series | None = None,
                 earnings: Iterable = (), r: float = 0.04) -> BacktestResult:
    close = close.astype(float)
    kind = "put" if strategy.type == "CSP" else "call"
    vol, ivr = vol_proxy(close), vol_rank(close)
    sma200 = close.rolling(200).mean()
    earnings = sorted(pd.Timestamp(e) for e in earnings)
    divs = dividends if dividends is not None else pd.Series(dtype=float)
    dates = close.index

    cash = 100.0 * close.iloc[WARMUP]          # capital for one contract at the start
    shares = 0
    opt = None                                 # dict(K, expiry, premium, open_date, collateral)
    trades: list[Trade] = []
    equity = {}

    def close_opt(date, cost, reason, extra_loss=0.0):
        # P&L = premium - opening commission - buyback (+ closing commission if bought back) - assignment loss
        nonlocal opt
        pnl = opt["premium"] * 100 - COMM - cost * 100 - (COMM if cost > 0 else 0.0) - extra_loss
        trades.append(Trade(opt["open_date"], date, kind, opt["K"], opt["premium"], cost, pnl,
                            opt["collateral"], reason))
        opt = None

    for i in range(WARMUP, len(dates)):
        date, S, sigma = dates[i], float(close.iloc[i]), float(vol.iloc[i])
        if not math.isfinite(sigma) or sigma <= 0:
            sigma = 0.2

        if opt is not None:
            K = opt["K"]
            days_left = (opt["expiry"] - date).days
            if days_left <= 0:
                if kind == "put" and S < K:
                    loss = (K - S) * 100
                    cash -= loss
                    close_opt(date, 0.0, "assigned", extra_loss=loss)
                elif kind == "call" and S > K:
                    cash += K * 100
                    shares = 0
                    close_opt(date, 0.0, "called_away", extra_loss=(S - K) * 100)
                else:
                    close_opt(date, 0.0, "expired")
            else:
                T = days_left / 365
                nxt = dates[i + 1] if i + 1 < len(dates) else None
                div_next = float(divs.get(nxt, 0.0)) if nxt is not None else 0.0
                theo = bs_price(S, K, T, r, sigma, kind)
                if kind == "call" and div_next > 0 and S > K and theo - (S - K) < div_next:
                    cash += K * 100
                    shares = 0
                    close_opt(date, 0.0, "early_assigned", extra_loss=(S - K) * 100)
                else:
                    bid, ask = synthetic_quote(theo)
                    buy = fill_price(bid, ask, "buy")
                    reason = None
                    if strategy.take_profit and buy <= opt["premium"] * (1 - strategy.take_profit):
                        reason = "take_profit"
                    elif strategy.roll_at_dte and not strategy.hold_to_expiry and days_left <= strategy.roll_at_dte:
                        reason = "roll"
                    elif strategy.close_before_earnings and _earnings_within(date, EARNINGS_BUFFER_DAYS, earnings) \
                            and any(e <= opt["expiry"] for e in earnings if e > date):
                        reason = "earnings"
                    if reason:
                        cash -= buy * 100 + COMM
                        close_opt(date, buy, reason)

        if opt is None and i < len(dates) - 1:
            if kind == "call" and shares == 0 and cash >= S * 100:
                cash -= S * 100
                shares = 100
            can_enter = (kind == "put" or shares == 100)
            if can_enter and float(np.nan_to_num(ivr.iloc[i], nan=0.0)) < strategy.ivr_min:
                can_enter = False
            if can_enter and strategy.type == "CSP" and strategy.trend_filter and not S > sma200.iloc[i]:
                can_enter = False
            if can_enter and strategy.close_before_earnings and _earnings_within(date, strategy.dte, earnings) \
                    and _earnings_within(date, EARNINGS_BUFFER_DAYS, earnings):
                can_enter = False
            if can_enter:
                T = strategy.dte / 365
                K = strike_for_delta(S, T, r, sigma, kind, strategy.delta_target)
                bid, ask = synthetic_quote(bs_price(S, K, T, r, sigma, kind))
                prem = fill_price(bid, ask, "sell")
                collateral = K * 100 if kind == "put" else S * 100
                if prem >= MIN_PREMIUM and (kind == "call" or cash >= K * 100):
                    cash += prem * 100 - COMM
                    opt = dict(K=K, expiry=date + pd.Timedelta(days=strategy.dte), premium=prem,
                               open_date=date, collateral=collateral)

        liability = 0.0
        if opt is not None:
            T = max((opt["expiry"] - date).days, 0) / 365
            liability = bs_price(S, opt["K"], T, r, sigma, kind) * 100
        equity[date] = cash + shares * S - liability

    return BacktestResult(trades, pd.Series(equity, dtype=float))
