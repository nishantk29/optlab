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
