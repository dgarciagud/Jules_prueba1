import json

import numpy as np
import pandas as pd
import pytest

from factor_monitor.common.runlog import RunLog
from factor_monitor.daily.data import (
    RefWeights, download_all, drifted_weights, estimate_weights, mark_stale, nnls, reference_weights,
    sector_returns, tracking, wide,
)  # fmt: skip
from factor_monitor.daily.run import index_calendar, run_data
from factor_monitor.daily.sources import SourceError, parse_chart, parse_stooq
from factor_monitor.daily.universe import Index, Series, load_daily_universe

# ----------------------------------------------------------------------------- fuentes


def chart_json(ts, closes, tz="America/New_York", adj=None):
    return json.dumps(
        {"chart": {"result": [{"meta": {"exchangeTimezoneName": tz}, "timestamp": ts,
                               "indicators": {"quote": [{"close": closes}], "adjclose": [{"adjclose": adj or closes}]}}], "error": None}}  # fmt: skip
    ).encode()


def test_parse_chart_uses_exchange_local_date():
    # 2024-03-05 09:30 Nueva York = 14:30 UTC; 2024-03-06 00:30 París = 2024-03-05 23:30 UTC
    df = parse_chart(chart_json([1709649000, 1709735400], [100.0, None]), "X")
    assert df["date"].tolist() == [pd.Timestamp("2024-03-05")] and df["close"].tolist() == [100.0]
    df = parse_chart(chart_json([1709681400], [7.0], tz="Europe/Paris"), "Y")
    assert df["date"].iloc[0] == pd.Timestamp("2024-03-06")


def test_parse_chart_errors():
    with pytest.raises(SourceError):
        parse_chart(json.dumps({"chart": {"result": None, "error": {"description": "No data found"}}}).encode(), "Z")
    with pytest.raises(SourceError):
        parse_chart(chart_json([1709649000], [None]), "Z")


def test_parse_stooq():
    df = parse_stooq(b"Date,Open,High,Low,Close,Volume\n2024-03-04,1,2,0.5,1.5,10\n2024-03-05,1,2,0.5,1.6,10\n", "x")
    assert df["close"].tolist() == [1.5, 1.6] and (df["adjclose"] == df["close"]).all()
    with pytest.raises(SourceError):
        parse_stooq(b"No data", "x")


# ----------------------------------------------------------------------------- pesos


def test_nnls_matches_known_solution():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 4))
    x_true = np.array([0.5, 0.0, 1.5, 0.0])
    x = nnls(A, A @ x_true - 0.01 * A[:, 1])
    assert np.allclose(x, x_true, atol=1e-2) and (x >= 0).all()


def synthetic_index(n=300, seed=1):
    """Índice = cesta de acciones fijas; devuelve (precios, nivel, pesos reales en la última fecha)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    rets = rng.normal(0.0003, 0.015, size=(n, 5)) + rng.normal(0, 0.01, size=(n, 1))
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=list("ABCDE"))
    shares = np.array([3.0, 1.0, 0.0, 2.0, 0.5])
    level = (px * shares).sum(axis=1)
    w_true = px.iloc[-1] * shares / level.iloc[-1]
    return px, level, w_true


def test_estimate_weights_recovers_basket():
    px, level, w_true = synthetic_index()
    w, info = estimate_weights(level, px)
    assert info["r2"] > 0.9999
    assert np.allclose(pd.Series(w)[w_true.index], w_true, atol=1e-6)


def test_estimate_weights_shrinks_to_prior_when_data_uninformative():
    px, level, _ = synthetic_index()
    px["F"] = px["A"]  # F es indistinguible de A
    prior = {"A": 0.0, "F": 1.0}
    w, _ = estimate_weights(level, px, prior=prior, shrink=1e-3)
    assert w["F"] > w["A"]


def test_drifted_weights_follow_prices():
    close = pd.DataFrame({"A": [50.0, 100.0], "B": [100.0, 100.0]}, index=pd.bdate_range("2025-01-01", periods=2))
    w = drifted_weights(close, {"A": 0.5, "B": 0.5})
    assert np.allclose(w.iloc[0], [1 / 3, 2 / 3]) and np.allclose(w.iloc[-1], [0.5, 0.5])


def test_sector_returns_reproduce_index_exactly():
    px, level, w_true = synthetic_index()
    universe = load_daily_universe()
    ix = universe.indices["CAC"]
    close = px.assign(CAC=level)
    ref = RefWeights("CAC", "x", "components", {}, {"A": {"sector": "s1", "weight": w_true["A"]}, "B": {"sector": "s1", "weight": w_true["B"]},
                     "D": {"sector": "s2", "weight": w_true["D"]}, "E": {"sector": "s2", "weight": w_true["E"]}}, "test")  # fmt: skip
    rets, w = sector_returns(ix, close, ref)
    assert list(rets.columns) == ["s1", "s2"] and np.allclose(w.sum(axis=1), 1)
    t = tracking(level, rets, w)
    assert t["r2"] > 0.999999 and t["te_annual"] < 1e-6


def test_reference_weights_components_and_calendar():
    px, level, w_true = synthetic_index()
    ix = Index(id="CAC", name="CAC", region="eu", weights_from_fund=None, composition_since="2025-03-01",
               components=[Series(id=k, name=k, role="component", yahoo=k, sector="s") for k in "ABCDE"])  # fmt: skip
    close = px.assign(CAC=level)
    close.iloc[10, 0] = np.nan  # festivo local de A: se arrastra en el calendario del índice
    cal = index_calendar(close, "CAC")
    ref = reference_weights(ix, cal, None, None, RunLog())
    assert ref.fit["from"] >= "2025-03-01" and ref.fit["r2"] > 0.9999
    assert abs(ref.members["A"]["weight"] - w_true["A"]) < 1e-6


# ----------------------------------------------------------------------------- descarga


class FakeYahoo:
    def __init__(self, fail=()):
        self.fail = set(fail)

    def history(self, symbol, start):
        if symbol in self.fail:
            raise SourceError("429")
        dates = pd.bdate_range("2025-01-01", periods=80)
        h = abs(hash(symbol)) % 100
        return pd.DataFrame({"date": dates, "close": 100 + h + np.arange(80) * 0.1, "adjclose": 100 + h + np.arange(80) * 0.1})

    def fund_sectors(self, symbol):
        return {"technology": 0.5, "energy": 0.5}


class FakeStooq:
    def __init__(self, ok=()):
        self.ok = set(ok)

    def history(self, symbol):
        if symbol not in self.ok:
            raise SourceError("reset")
        return FakeYahoo().history(symbol, None)


def test_download_all_fallbacks():
    universe = load_daily_universe()
    prev = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "id": ["AI"], "close": [1.0], "adjclose": [1.0]})
    runlog = RunLog()
    prices, cov = download_all(universe, FakeYahoo(fail={"^GSPC", "SMH", "XLK"}), FakeStooq(ok={"^spx"}), prev, runlog)
    src = cov.set_index("id")["source"]
    assert src["SPX"] == "stooq" and src["AI"] == "anterior" and pd.isna(src["SPX:technology"])
    assert src["CAC"] == "yahoo" and set(prices["id"]) >= {"SPX", "AI", "CAC"}
    cov = mark_stale(cov, universe, runlog)
    assert cov.set_index("id").loc["AI", "stale"]  # la copia anterior es de hace 80 sesiones
    assert any("AI" in w["message"] for w in runlog.warnings)


def test_universe_config_is_consistent():
    u = load_daily_universe()
    assert set(u.indices) == {"SPX", "CAC"}
    assert len(u.indices["CAC"].components) == 40 and all(c.sector for c in u.indices["CAC"].components)
    assert set(u.indices["SPX"].sectors) == set(u.sector_names)


def test_run_data_end_to_end(tmp_path):
    universe = load_daily_universe()
    code = run_data(universe, tmp_path, FakeYahoo(), None, None)
    assert code == 0
    meta = json.loads((tmp_path / "data_meta.json").read_text())
    assert meta["data_end"] and set(meta["tracking"]) == {"SPX", "CAC"}
    sectors = pd.read_parquet(tmp_path / "sectors.parquet")
    assert set(sectors["index"]) == {"SPX", "CAC"}
    last = sectors[sectors["date"] == sectors["date"].max()]
    assert np.allclose(last.groupby("index")["weight"].sum(), 1)
    assert wide(pd.read_parquet(tmp_path / "prices.parquet")).shape[1] == len(universe.all_series())


# ----------------------------------------------------------------------------- atribución

from factor_monitor.daily.attribution import attribution_table, daily_sector_contributions, linked, relative_strength  # noqa: E402


def basket_ref(w_true, sectors):
    return RefWeights("CAC", "x", "components", {}, {m: {"sector": sectors[m], "weight": float(w_true[m])} for m in sectors}, "test")


def test_attribution_adds_up_to_index_for_every_period():
    px, level, w_true = synthetic_index(n=400)
    sectors = {"A": "s1", "B": "s1", "D": "s2", "E": "s2"}
    cal = px.assign(CAC=level)
    t = attribution_table("CAC", cal, basket_ref(w_true, sectors), {"s1": "Uno", "s2": "Dos"}, {})
    for period, g in t.groupby("period"):
        total = g.loc[g.level == "index", "contribution"].iloc[0]
        assert np.isclose(g.loc[g.level == "sector", "contribution"].sum(), total, atol=1e-10), period
        assert np.isclose(g.loc[g.level == "member", "contribution"].sum(), total, atol=1e-10), period
        assert abs(g.loc[g.level == "residual", "contribution"].iloc[0]) < 1e-10
    one_day = t[(t.period == "1D") & (t.level == "index")]["contribution"].iloc[0]
    assert np.isclose(one_day, level.iloc[-1] / level.iloc[-2] - 1)
    ytd = t[(t.period == "YTD") & (t.level == "index")].iloc[0]
    first_2026 = level[level.index.year == level.index[-1].year].index[0]
    assert ytd["from"] == first_2026.date().isoformat()
    # rentabilidad de un sector = la de su cesta de acciones fijas
    y1 = t[(t.period == "1A") & (t.id == "s1")].iloc[0]
    shares = w_true[["A", "B"]] / px.iloc[-1][["A", "B"]]
    basket = (px[["A", "B"]] * shares).sum(axis=1)
    assert np.isclose(y1["ret"], basket.iloc[-1] / basket.iloc[-253] - 1)


def test_linked_contributions_compound():
    dates = pd.bdate_range("2025-01-01", periods=3)
    c = pd.DataFrame({"x": [0.10, -0.05, 0.02]}, index=dates)
    r = pd.Series([0.10, -0.05, 0.02], index=dates)
    contrib, total = linked(c, r, 0)
    assert np.isclose(total, 1.10 * 0.95 * 1.02 - 1) and np.isclose(contrib["x"], total)


def test_relative_strength_and_daily_contributions():
    px, level, w_true = synthetic_index(n=400)
    px["A"] = px["A"] * np.exp(np.linspace(0, 3.0, len(px)))  # A sube mucho más que el resto
    cal = px.assign(CAC=(px * (w_true / px.iloc[-1])).sum(axis=1))
    ref = basket_ref(w_true, {"A": "fuerte", "B": "flojo", "D": "flojo", "E": "flojo"})
    rs = relative_strength("CAC", cal, ref, {})
    r3 = rs[rs.horizon == "3M"].set_index("sector")["rs"]
    assert r3["fuerte"] > 0 > r3["flojo"]
    d = daily_sector_contributions("CAC", cal, ref, sessions=50)
    per_day = d.groupby("date").agg(c=("contribution", "sum"), r=("index_ret", "first"))
    assert len(per_day) == 50 and np.allclose(per_day["c"], per_day["r"], atol=1e-12)


# ----------------------------------------------------------------------------- factores

from factor_monitor.daily import factors as fx  # noqa: E402


def test_orthogonalize_removes_market():
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(size=200))
    F = pd.DataFrame({"f": 0.8 * m + rng.normal(size=200)})
    Fp = fx.orthogonalize(F, m)
    assert abs(np.corrcoef(Fp["f"], m)[0, 1]) < 1e-10


def test_factor_changes_by_kind():
    universe = load_daily_universe()
    dates = pd.bdate_range("2025-01-01", periods=3)
    close = pd.DataFrame({"OIL": [100.0, 110.0, 99.0], "US10Y": [4.0, 4.1, np.nan]}, index=dates)
    ch = fx.factor_changes(close, universe, dates)
    assert np.isclose(ch["OIL"].iloc[1], np.log(1.1)) and np.isclose(ch["US10Y"].iloc[1], 0.1)
    assert ch["US10Y"].iloc[2] == 0.0  # sin dato: se arrastra el nivel


def test_snapshot_dates_month_ends():
    dates = pd.bdate_range("2019-01-01", "2026-09-25")
    snaps = fx.snapshot_dates(dates, years=1)
    assert snaps[-1] == dates[-1] and pd.Timestamp("2026-08-31") in snaps and len(snaps) in (12, 13)


def test_sector_driven_by_factor_gets_the_macro_share():
    rng = np.random.default_rng(4)
    n = 300
    dates = pd.bdate_range("2025-01-01", periods=n)
    m = rng.normal(0, 0.01, n)
    oil = rng.normal(0, 0.02, n)
    usd = rng.normal(0, 0.005, n)
    energy = 0.8 * m + 0.5 * oil + rng.normal(0, 0.003, n)
    other = 1.0 * m + rng.normal(0, 0.004, n)
    lv = lambda r: 100 * np.exp(np.cumsum(r))  # noqa: E731
    cal = pd.DataFrame({"E": lv(energy), "O": lv(other)}, index=dates)
    cal["CAC"] = 0.5 * cal["E"] / cal["E"].iloc[-1] + 0.5 * cal["O"] / cal["O"].iloc[-1]
    close = cal.assign(OIL=lv(oil), USD=lv(usd))
    universe = load_daily_universe()
    ref = RefWeights("CAC", "x", "components", {}, {"E": {"sector": "energy", "weight": 0.5}, "O": {"sector": "industrials", "weight": 0.5}}, "t")
    d = fx.sector_factor_shapley("CAC", "us", cal, close, ref, universe, snapshots=False)
    e = d[d.target == "energy"].set_index("factor")
    assert np.isclose(e["shapley"].sum(), e["r2"].iloc[0])
    assert e.loc["OIL", "share"] > 0.9 and e.loc["OIL", "std_beta"] > 0
    assert set(d.target) == {"energy", "industrials", fx.INDEX_TARGET}
    assert "MKT" not in set(d[d.target == fx.INDEX_TARGET].factor)
