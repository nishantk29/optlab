"""`python -m optlab.cli lab-run` — evaluate every 'candidate' strategy and gate it."""
import argparse
import datetime as dt
import json
import os

from . import data, seed, store
from .backtest import WARMUP
from .lifecycle import after_backtest
from .walkforward import evaluate, gate, report_to_dict


def lab_run(db: str, symbols: list[str], out: str, loader=data.load_history) -> dict:
    started = dt.datetime.now()
    con = store.connect(db)
    for s in seed.standard_variants():
        store.upsert_strategy(con, s)
    hist, failed = {}, []
    for sym in symbols:
        try:
            close, divs, earn = loader(sym)
            if close is None or len(close) <= WARMUP + 1:
                raise ValueError(f"only {0 if close is None else len(close)} rows")
            hist[sym] = (close, divs, earn)
        except Exception as e:                       # one bad symbol must not sink the run
            failed.append(sym)
            print(f"skipping {sym}: {e}")
    if not hist:
        store.log_run(con, "lab", False, f"all symbols failed: {failed}", started, dt.datetime.now())
        return {"evaluated": 0, "symbols": symbols, "failed_symbols": failed, "strategies": []}
    evaluated, rows = 0, []
    for strat, status in store.get_strategies(con, ("candidate",)):
        passes, per_symbol = 0, {}
        for sym, (close, divs, earn) in hist.items():
            rep = evaluate(strat, close, divs, earn)
            ok, reasons = gate(rep)
            passes += ok
            store.record_backtest(con, strat.id, sym, ok, reasons, report_to_dict(rep))
            per_symbol[sym] = {"passed": ok, "reasons": reasons,
                               "per_1k_total_return": round(rep.total_return * 1000, 2),
                               "win_rate": round(rep.win_rate, 3), "worst_per_1k": round(rep.worst_per_1k, 2),
                               "n_trades": rep.n_trades}
        new = after_backtest(status, passes >= len(hist) / 2)
        store.set_status(con, strat.id, new)
        evaluated += 1
        rows.append({"id": strat.id, "spec": strat.to_dict(), "status": new, "symbols": per_symbol,
                     "per_1k_total_return": round(sum(v["per_1k_total_return"] for v in per_symbol.values())
                                                  / max(len(per_symbol), 1), 2)})
    all_rows = rows or [{"id": s.id, "spec": s.to_dict(), "status": st, "symbols": {}, "per_1k_total_return": None}
                        for s, st in store.get_strategies(con)]
    result = {"evaluated": evaluated, "symbols": symbols, "failed_symbols": failed,
              "strategies": sorted(all_rows, key=lambda r: (r["per_1k_total_return"] is None,
                                                            -(r["per_1k_total_return"] or 0), r["id"]))}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    store.log_run(con, "lab", True, f"evaluated={evaluated} failed_symbols={failed}", started, dt.datetime.now())
    return result


def main(argv=None):
    p = argparse.ArgumentParser(prog="optlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    lr = sub.add_parser("lab-run")
    lr.add_argument("--db", default="data/options.duckdb")
    lr.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT")
    lr.add_argument("--out", default="out/optlab_lab.json")
    a = p.parse_args(argv)
    os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
    res = lab_run(a.db, a.symbols.split(","), a.out)
    passed = [r for r in res["strategies"] if r["status"] == "backtest_pass"]
    print(f"evaluated {res['evaluated']} strategies; {len(passed)} passed the backtest gate")
    for r in passed[:10]:
        print(f"  {r['id']} {r['spec']['type']} d={r['spec']['delta_target']} dte={r['spec']['dte']} "
              f"per$1k={r['per_1k_total_return']}")


if __name__ == "__main__":
    main()
