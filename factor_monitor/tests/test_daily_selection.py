import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factor_monitor.common.runlog import RunLog
from factor_monitor.common.universe import parse_universe
from factor_monitor.nightly import daily_layer, run as nightly
from factor_monitor.nightly.daily_layer import MKT, run_daily_layer, session_closes
from factor_monitor.nightly.selection import is_week_end, next_week_bounds, run_selection, select_target
from factor_monitor.nightly.store import BarStore, LocalBackend

RAW = {
    "instruments": {
        "DAX": {"kind": "target", "dukascopy_id": "deuidxeur", "session": "eu", "exchange": "XETR"},
        "SX5E": {"kind": "target", "dukascopy_id": "eusidxeur", "session": "eu", "exchange": "XETR"},
        "CAC": {"kind": "target", "dukascopy_id": "fraidxeur", "session": "eu", "exchange": "XPAR"},
        "SPX": {"kind": "target", "dukascopy_id": "usa500idxusd", "session": "us", "exchange": "XNYS"},
        "BRENT": {"kind": "factor", "dukascopy_id": "brentcmdusd"},
        "EURUSD": {"kind": "factor", "dukascopy_id": "eurusd"},
    },
    "factors": {"BRENT": {"instrument": "BRENT"}, "USD": {"instrument": "EURUSD", "sign": -1}},
    "targets": {
        "DAX": {"session": "eu", "exchange": "XETR", "candidates": ["BRENT", "USD"], "market": {"type": "mean", "members": ["SX5E", "CAC"]}},
        "SPX": {"session": "us", "exchange": "XNYS", "candidates": ["BRENT", "USD"], "market": {"type": "none"}},
    },
}


@pytest.fixture(scope="module")
def universe():
    return parse_universe(RAW)


def synthetic_bars(start="2024-01-02", end="2024-06-28", seed=7):
    """Velas de 5 min de 07:00 a 21:00 UTC con estructura de factores conocida."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    idx = pd.DatetimeIndex(
        [t for d in days for t in pd.date_range(d + pd.Timedelta(hours=7), d + pd.Timedelta(hours=21), freq="5min", inclusive="left")],
        tz="UTC",
    )
    n = len(idx)
    s = 0.001
    m, brent, usd = (rng.normal(0, s, n) for _ in range(3))
    inc = {
        "SX5E": m + rng.normal(0, s / 4, n),
        "CAC": m + rng.normal(0, s / 4, n),
        "DAX": m + 0.6 * brent + rng.normal(0, s / 3, n),
        "SPX": 0.7 * usd + rng.normal(0, s / 2, n),
        "BRENT": brent,
        "EURUSD": -usd,
    }
    bars = {}
    for k, r in inc.items():
        close = 100 * np.exp(np.cumsum(r))
        bars[k] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "spread": 0.01, "n": 5.0},
            index=pd.DatetimeIndex(idx, name="ts"),
        )
    return bars


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars()


def test_session_closes_dst_and_holidays(bars, universe):
    panel = daily_layer.close_panel(bars)
    closes = session_closes(panel, universe.targets["DAX"])
    assert pd.Timestamp("2024-05-01") not in closes.index  # Xetra cerrado el 1 de mayo
    assert pd.Timestamp("2024-03-29") not in closes.index  # Viernes Santo
    winter, summer = pd.Timestamp("2024-01-15 16:25", tz="UTC"), pd.Timestamp("2024-04-02 15:25", tz="UTC")
    assert closes.loc["2024-01-15", "DAX"] == pytest.approx(panel.loc[winter, "DAX"])
    assert closes.loc["2024-04-02", "DAX"] == pytest.approx(panel.loc[summer, "DAX"])
    spx = session_closes(panel, universe.targets["SPX"])
    assert pd.Timestamp("2024-05-01") in spx.index
    assert spx.loc["2024-01-16", "SPX"] == pytest.approx(panel.loc[pd.Timestamp("2024-01-16 20:55", tz="UTC"), "SPX"])


def test_daily_layer_recovers_drivers(bars, universe):
    res = run_daily_layer(bars, universe, RunLog())
    last = res.factors[res.factors["date"] == res.factors["date"].max()]
    dax = last[last["target"] == "DAX"].set_index("factor")
    assert dax.loc["BRENT", "share"] > 0.8 and dax.loc["USD", "share"] < 0.1
    assert dax.loc["BRENT", "std_beta"] > 0
    spx = last[last["target"] == "SPX"].set_index("factor")
    assert MKT not in spx.index
    assert spx.loc["USD", "share"] > 0.8
    r2 = res.r2.groupby("target")["r2"].last()
    assert (r2 > 0.5).all()
    g = res.factors.groupby(["date", "target"])["shapley"].sum()
    merged = res.r2.set_index(["date", "target"])["r2"]
    assert np.allclose(g.loc[merged.index], merged)


def test_selection_end_to_end(bars, universe):
    res = run_daily_layer(bars, universe, RunLog())
    sel, rows = run_selection(res.factors, universe, date(2024, 6, 28))
    assert sel["valid_from"] == "2024-07-01" and sel["valid_to"] == "2024-07-05"
    assert sel["targets"]["DAX"]["factors"] == ["BRENT"] and sel["targets"]["DAX"]["market"]
    assert sel["targets"]["SPX"]["factors"] == ["USD"] and not sel["targets"]["SPX"]["market"]
    assert set(rows["target"]) == {"DAX", "SPX"}


def fake_factors(weekly):
    """weekly: {factor: [(share, std_beta) por semana]} → filas de viernes consecutivos."""
    rows = []
    fridays = pd.date_range("2024-05-03", periods=len(next(iter(weekly.values()))), freq="7D")
    for f, vals in weekly.items():
        for d, (share, sb) in zip(fridays, vals):
            rows.append({"date": d, "target": "X", "factor": f, "share": share, "std_beta": sb})
    return pd.DataFrame(rows), fridays[-1].date()


def test_selection_rules():
    df, as_of = fake_factors(
        {
            "A": [(0.5, 1)] * 4,
            "B": [(0.3, -1)] * 4,
            "C": [(0.2, 1), (0.2, -1), (0.2, 1), (0.2, 1)],   # signo inestable
            "D": [(0.16, 1)] * 4,
            "E": [(0.10, 1)] * 4,                              # cuota < 15 %
            "F": [(0.17, 1)] * 4,
        }
    )
    sel = select_target(df, "X", as_of, has_market=True)
    assert sel["factors"] == ["A", "B", "F"]  # máximo 3, por cuota
    assert sel["signs"]["B"] == -1 and sel["signs"]["C"] == 0
    df2, as_of2 = fake_factors({"A": [(0.1, 1)] * 4})
    assert select_target(df2, "X", as_of2, has_market=False)["status"] == "sin_driver_macro"
    df3, as_of3 = fake_factors({"A": [(0.9, 1)] * 3})  # menos de 4 semanas
    assert select_target(df3, "X", as_of3, has_market=False)["factors"] == []


def test_week_end_with_holidays():
    assert is_week_end(date(2024, 6, 28))
    assert not is_week_end(date(2024, 6, 27))
    assert is_week_end(date(2024, 3, 28))  # Viernes Santo cerrado en Xetra
    assert next_week_bounds(date(2024, 3, 28)) == (date(2024, 4, 1), date(2024, 4, 5))


def make_args(tmp_path, **kw):
    base = dict(
        results_dir=str(tmp_path / "results"), skip_download=True, until=None, engine="node",
        years=3, full=False, force_selection=False,
    )  # fmt: skip
    base.update(kw)
    return SimpleNamespace(**base)


def test_pipeline_writes_and_isolates_failures(tmp_path, bars, universe, monkeypatch):
    store = BarStore(LocalBackend(tmp_path / "remote"), tmp_path / "cache")
    store.write({(k, 2024): v for k, v in bars.items()})
    monkeypatch.setattr(nightly, "yesterday_utc", lambda: date(2024, 6, 29))

    assert nightly.run(make_args(tmp_path), universe, store) == 0
    res = tmp_path / "results"
    for f in ("shapley.parquet", "daily_r2.parquet", "selection.json", "selection_history.parquet", "recent_bars.parquet",
              "run_meta.json", "thresholds.json", "intraday_r2.parquet", "track_record_alerts.parquet"):
        assert (res / f).exists(), f
    meta = json.loads((res / "run_meta.json").read_text())
    assert meta["data_end"] == "2024-06-28" and meta["steps"]["daily"]["status"] == "ok"
    assert meta["steps"]["track_record"]["status"] == "ok", meta["steps"]["track_record"]
    thr = json.loads((res / "thresholds.json").read_text())
    assert set(thr["targets"]) == {"DAX", "SPX"}
    before = (res / "shapley.parquet").read_bytes()
    recent = pd.read_parquet(res / "recent_bars.parquet")
    assert set(recent["instrument"]) == set(bars) and recent["ts"].min() >= pd.Timestamp("2024-06-07", tz="UTC")

    def boom(*a, **k):
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(nightly, "run_daily_layer", boom)
    assert nightly.run(make_args(tmp_path), universe, store) == 1
    assert (res / "shapley.parquet").read_bytes() == before  # resultados anteriores intactos
    meta2 = json.loads((res / "run_meta.json").read_text())
    assert meta2["steps"]["daily"]["status"] == "failed" and "fallo simulado" in meta2["steps"]["daily"]["error"]
    assert meta2["last_success"]["daily"] == meta["steps"]["daily"]["at"]
    assert meta2["steps"]["selection"]["status"] == "skipped"  # ya calculada con estos datos
