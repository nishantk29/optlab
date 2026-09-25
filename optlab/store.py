"""DuckDB persistence for the lab. One file: ~/invest/data/options.duckdb on the Spectre."""
import datetime as dt
import json
import duckdb

from .lifecycle import STATUSES
from .strategy import Strategy

SCHEMA = """
create table if not exists strategies(id varchar primary key, spec json, status varchar, updated_at timestamp);
create table if not exists backtests(strategy_id varchar, symbol varchar, run_at timestamp, passed boolean,
                                     reasons json, metrics json);
create table if not exists runs(job varchar, started timestamp, finished timestamp, ok boolean, detail varchar);
create table if not exists chains(snap_date date, symbol varchar, expiry date, strike double, type varchar,
                                  bid double, ask double, last double, volume bigint, oi bigint, iv double);
create table if not exists prices(symbol varchar, date date, close double);
create table if not exists events(symbol varchar, date date, kind varchar, value double);
"""


def connect(path: str):
    con = duckdb.connect(path)
    con.execute(SCHEMA)
    return con


def upsert_strategy(con, s: Strategy, status: str = "candidate") -> None:
    if con.execute("select 1 from strategies where id = ?", [s.id]).fetchone():
        return
    con.execute("insert into strategies values (?, ?, ?, ?)",
                [s.id, json.dumps(s.to_dict()), status, dt.datetime.now()])


def get_strategies(con, statuses=None):
    q, args = "select spec, status from strategies", []
    if statuses:
        q += " where status in (" + ",".join("?" * len(statuses)) + ")"
        args = list(statuses)
    q += " order by id"
    return [(Strategy.from_dict(json.loads(spec)), st) for spec, st in con.execute(q, args).fetchall()]


def set_status(con, strategy_id: str, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    con.execute("update strategies set status = ?, updated_at = ? where id = ?",
                [status, dt.datetime.now(), strategy_id])


def record_backtest(con, strategy_id, symbol, passed, reasons, metrics) -> None:
    con.execute("insert into backtests values (?, ?, ?, ?, ?, ?)",
                [strategy_id, symbol, dt.datetime.now(), passed, json.dumps(reasons), json.dumps(metrics)])


def log_run(con, job, ok, detail, started, finished) -> None:
    con.execute("insert into runs values (?, ?, ?, ?, ?)", [job, started, finished, ok, detail])
