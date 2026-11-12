# optlab

A research lab for two income options strategies: covered calls (CC) and cash-secured puts (CSP). It backtests each strategy variant on daily price history with Black-Scholes pricing and a conservative fill model, scores it with walk-forward windows, and only advances variants that beat buy-and-hold on a risk-adjusted basis.

It is research only. Nothing in this repo connects to a broker or places trades.

## How it works
- **Strategies are data, not code.** Each variant is a parameter record (`optlab/strategy.py`): delta or moneyness target, days to expiry, take-profit, roll timing. `optlab/seed.py` defines 22 starting variants.
- **Daily simulation** (`optlab/backtest.py`): sells the option, marks it to market every day, handles take-profit, rolls, assignment and call-away. Contracts are fractional, so every variant uses all of its capital and commissions scale with size.
- **Pricing and fills:** Black-Scholes with a continuous dividend yield (`optlab/pricing.py`). Sellers get mid minus 25% of the spread, buyers pay mid plus 25%, plus $0.65 per contract (`optlab/fills.py`).
- **Walk-forward gate** (`optlab/walkforward.py`): each run is split into 4 out-of-sample windows. A variant passes only if it is profitable in at least 3 of the 4 windows, its Sortino ratio beats buy-and-hold, its max drawdown is no worse than 1.25x the underlying's, and it made at least 30 trades.
- **Lifecycle** (`optlab/lifecycle.py`): candidate → backtest_pass → paper → promoted, with benched and retired states.
- **Storage:** DuckDB (`optlab/store.py`) keeps strategies, backtest results and run logs. Price history comes from yfinance with a one-day local cache (`optlab/data.py`).

## Stack
Python 3.11+, NumPy, pandas, SciPy, DuckDB, yfinance, pytest. Managed with [uv](https://docs.astral.sh/uv/).

## Run
```bash
uv run python -m optlab.cli lab-run --symbols SPY,QQQ,AAPL,MSFT --db data/options.duckdb --out out/optlab_lab.json
```
This evaluates every candidate variant on each symbol, records the results in DuckDB, and writes a ranked JSON report. `data/` and `out/` are git-ignored.

## Tests
```bash
uv run --extra dev pytest -q   # 70 tests
```
The tests cover pricing against known values, fills, the backtest engine (with golden results), metrics, walk-forward windows and the gate, the lifecycle, and the store.
