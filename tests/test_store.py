import datetime as dt
import pytest
from optlab import store
from optlab.strategy import Strategy


@pytest.fixture
def con(tmp_path):
    return store.connect(str(tmp_path / "t.duckdb"))


def test_schema(con):
    names = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}
    assert {"strategies", "backtests", "runs", "chains", "prices", "events"} <= names


def test_upsert_keeps_status_and_roundtrips(con):
    s = Strategy("CSP", 0.25, 30)
    store.upsert_strategy(con, s)
    store.set_status(con, s.id, "paper")
    store.upsert_strategy(con, s)
    assert store.get_strategies(con) == [(s, "paper")]
    assert store.get_strategies(con, ("candidate",)) == []


def test_set_status_rejects_unknown(con):
    with pytest.raises(ValueError):
        store.set_status(con, "x", "moon")


def test_record_backtest_and_run(con):
    store.record_backtest(con, "abc", "SPY", False, ["fewer than 30 trades"], {"n_trades": 3})
    row = con.execute("select strategy_id, symbol, passed, reasons, metrics from backtests").fetchone()
    assert row[0:3] == ("abc", "SPY", False)
    now = dt.datetime.now()
    store.log_run(con, "lab", True, "ok", now, now)
    assert con.execute("select count(*) from runs").fetchone()[0] == 1
