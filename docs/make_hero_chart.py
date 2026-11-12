"""README chart: SPY, buy-and-hold vs two seed variants, equity per $1k, with the four walk-forward windows shaded.

uv run --with matplotlib python docs/make_hero_chart.py [out.png]
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optlab import data
from optlab.backtest import run_backtest, WARMUP
from optlab.strategy import Strategy
from optlab.walkforward import buy_and_hold, split_windows
out = sys.argv[1] if len(sys.argv) > 1 else "docs/hero-chart.png"
close, divs, earn = data.load_history("SPY", cache_dir=".cache")
bh = buy_and_hold(close, divs).iloc[WARMUP:]
bh = bh / bh.iloc[0] * 1000
series = {"Buy and hold SPY": (bh, "#1f2937", 2.2)}
for label, s, col in [("Covered call, 0.30 delta, 45 days, take profit 50%", Strategy("CC", 0.30, 45), "#2563eb"),
                      ("Cash-secured put, 0.20 delta, 30 days, trend filter", Strategy("CSP", 0.20, 30, trend_filter=True), "#d97706")]:
    eq = run_backtest(s, close, divs, earn).equity.dropna()
    series[label] = (eq / eq.iloc[0] * 1000, col, 1.8)
fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
for i, (a, b) in enumerate(split_windows(bh.index, 4)):
    ax.axvspan(a, b, color="#f3f4f6" if i % 2 == 0 else "#e5e7eb", lw=0, zorder=0)
    ax.text(a, 1.0, f" window {i+1}", transform=ax.get_xaxis_transform(), va="top", fontsize=10, color="#6b7280")
for label, (s, col, lw) in series.items():
    ax.plot(s.index, s.values, label=label, color=col, lw=lw)
ax.set_title("SPY daily backtest: equity per \\$1,000, four walk-forward windows", loc="left", fontsize=15, color="#111827", pad=14)
ax.set_ylabel(r"\$ per \$1,000 invested", color="#374151")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#d1d5db", lw=.6); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, .93), fontsize=10.5)
fig.text(0.99, 0.01, "Black-Scholes pricing, fills at mid ± 25% of spread, \\$0.65/contract. Research only.", ha="right", fontsize=9, color="#6b7280")
fig.tight_layout(); fig.savefig(out); print(out, {k: round(float(v[0].iloc[-1]), 1) for k, v in series.items()})
