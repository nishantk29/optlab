import json
from optlab import cli, seed, store
from tests.test_backtest import gbm


def fake_loader(symbol, years=10, cache_dir=".cache"):
    import pandas as pd
    seed_by_sym = {"AAA": 3, "BBB": 4}
    return gbm(n=252 + 1500, seed=seed_by_sym[symbol]), pd.Series(dtype=float), []


def test_standard_variants():
    v = seed.standard_variants()
    assert len(v) == 22 and len({s.id for s in v}) == 22
    assert {s.type for s in v} == {"CC", "CSP"}


def test_lab_run_deterministic_and_updates_status(tmp_path):
    db, out = str(tmp_path / "a.duckdb"), str(tmp_path / "lab.json")
    r1 = cli.lab_run(db, ["AAA", "BBB"], out, loader=fake_loader)
    r2 = cli.lab_run(str(tmp_path / "b.duckdb"), ["AAA", "BBB"], str(tmp_path / "lab2.json"), loader=fake_loader)
    assert r1 == r2                                   # golden: fully deterministic
    con = store.connect(db)
    statuses = {st for _, st in store.get_strategies(con)}
    assert statuses <= {"backtest_pass", "retired"} and statuses
    doc = json.load(open(out))
    assert len(doc["strategies"]) == 22
    assert all("per_1k_total_return" in s for s in doc["strategies"])


def test_lab_run_is_incremental(tmp_path):
    db = str(tmp_path / "c.duckdb")
    cli.lab_run(db, ["AAA"], str(tmp_path / "x.json"), loader=fake_loader)
    r = cli.lab_run(db, ["AAA"], str(tmp_path / "y.json"), loader=fake_loader)
    assert r["evaluated"] == 0                        # nothing left in 'candidate'


def test_lab_run_survives_bad_symbols(tmp_path):
    import pandas as pd

    def loader(symbol, years=10, cache_dir=".cache"):
        if symbol == "BAD":
            raise RuntimeError("no data")
        if symbol == "SHORT":
            return gbm(n=100), pd.Series(dtype=float), []
        return fake_loader(symbol)

    r = cli.lab_run(str(tmp_path / "d.duckdb"), ["AAA", "BAD", "SHORT"], str(tmp_path / "z.json"), loader=loader)
    assert r["evaluated"] == 22 and sorted(r["failed_symbols"]) == ["BAD", "SHORT"]


def test_lab_run_all_symbols_fail_leaves_candidates(tmp_path):
    def loader(symbol, years=10, cache_dir=".cache"):
        raise RuntimeError("down")

    db = str(tmp_path / "e.duckdb")
    r = cli.lab_run(db, ["X"], str(tmp_path / "e.json"), loader=loader)
    assert r["evaluated"] == 0
    con = store.connect(db)
    assert {st for _, st in store.get_strategies(con)} == {"candidate"}
    assert con.execute("select ok from runs").fetchone()[0] is False
