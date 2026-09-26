import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_intraday_alerts import RAW, make_bars  # noqa: E402
from test_track_record import selection_history  # noqa: E402

from factor_monitor.common.events import COLUMNS as EVENT_COLUMNS  # noqa: E402
from factor_monitor.common.runlog import RunLog  # noqa: E402
from factor_monitor.common.universe import parse_universe  # noqa: E402
from factor_monitor.live import sync_results  # noqa: E402
from factor_monitor.live.alerts import append_alerts  # noqa: E402
from factor_monitor.live.consistency import compare, status_for  # noqa: E402
from factor_monitor.live.engine import LiveEngine  # noqa: E402
from factor_monitor.live.runner import seconds_to_next_step  # noqa: E402
from factor_monitor.nightly import track_record as tr  # noqa: E402
from factor_monitor.sources.mt5 import MT5Source  # noqa: E402

RAW_MT5 = {**RAW, "instruments": {k: {**v, "mt5_symbol": f"{k}.dwx"} for k, v in RAW["instruments"].items()}}
U = parse_universe(RAW_MT5)


class FakeMT5:
    """Mock mínimo del paquete MetaTrader5: el servidor va `offset` horas por delante de UTC."""

    TIMEFRAME_M5 = 5

    def __init__(self, bars=None, offset_hours=2, now=1_700_000_000.0, tick_age=5.0, symbols=None):
        self.bars = bars or {}
        self.offset = offset_hours
        self.now = now
        self.tick_age = tick_age
        self.symbols = symbols if symbols is not None else [f"{k}.dwx" for k in RAW["instruments"]]

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (0, "ok")

    def symbols_get(self):
        return [SimpleNamespace(name=s) for s in self.symbols]

    def symbol_info(self, symbol):
        return SimpleNamespace(point=0.01)

    def symbol_info_tick(self, symbol):
        if self.tick_age is None:
            return None
        return SimpleNamespace(time=int(self.now - self.tick_age + self.offset * 3600))

    def copy_rates_range(self, symbol, timeframe, start, end):
        iid = symbol.replace(".dwx", "")
        df = self.bars.get(iid)
        if df is None:
            return None
        srv_idx = df.index + pd.Timedelta(hours=self.offset)
        lo = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        hi = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        m = (srv_idx >= lo) & (srv_idx <= hi)
        sub = df[m]
        arr = np.zeros(len(sub), dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4")])
        arr["time"] = (srv_idx[m].as_unit("s").asi8)
        for c in ("open", "high", "low", "close"):
            arr[c] = sub[c].to_numpy()
        arr["tick_volume"] = 10
        arr["spread"] = 2
        return arr


@pytest.mark.parametrize("offset", [2, 3])
def test_server_offset_detection_and_conversion(tmp_path, offset):
    ts = pd.date_range("2024-02-15 08:00", periods=12, freq="5min", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.arange(12.0), "spread": 0.0, "n": 1.0}, index=ts)
    now = pd.Timestamp("2024-02-15 09:02", tz="UTC").timestamp()
    fake = FakeMT5({"DAX": bars}, offset_hours=offset, now=now)
    src = MT5Source(fake, state_path=tmp_path / "s.json", now=lambda: now)
    res = src.detect_offset()
    assert (res.hours, res.source) == (offset, "detected")
    got = src.bars("DAX.dwx", ts[0], pd.Timestamp("2024-02-15 09:00", tz="UTC"))
    assert got.index[0] == ts[0] and got["close"].iloc[0] == 0.0  # marca = apertura en UTC
    assert got.index[-1] < pd.Timestamp("2024-02-15 09:00", tz="UTC")  # la vela en curso no entra
    assert got["spread"].iloc[0] == pytest.approx(0.02)


def test_offset_falls_back_to_stored_without_recent_tick(tmp_path):
    now = 1_700_000_000.0
    src = MT5Source(FakeMT5(offset_hours=3, now=now), state_path=tmp_path / "s.json", now=lambda: now)
    assert src.detect_offset().source == "detected"
    stale = MT5Source(FakeMT5(offset_hours=3, now=now, tick_age=3 * 86400), state_path=tmp_path / "s.json", now=lambda: now)
    res = stale.detect_offset()
    assert (res.hours, res.source) == (3, "stored")
    fresh = MT5Source(FakeMT5(now=now, tick_age=None), state_path=tmp_path / "otro.json", now=lambda: now)
    assert fresh.detect_offset().source == "default"


def test_inventory_flags_missing_symbols(tmp_path):
    fake = FakeMT5(symbols=["DAX.dwx", "SX5E.dwx"])
    inv = MT5Source(fake, state_path=tmp_path / "s.json").inventory(U).set_index("instrument")
    assert inv.loc["DAX", "available"] and not inv.loc["BRENT", "available"]
    assert inv.loc["BRENT", "reason"] == "símbolo no encontrado en el terminal"


def test_consistency_and_clock_warning():
    idx = pd.date_range("2024-02-01 08:00", periods=600, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 600))), index=idx)
    duk = pd.DataFrame({"close": close})
    good = pd.DataFrame({"close": close * (1 + rng.normal(0, 2e-5, 600))})
    res = compare(duk, good, "DAX")
    assert res["status"] == "ok" and res["best_lag"] == 0 and not res["clock_warning"]
    shifted = pd.DataFrame({"close": close.shift(1)}).dropna()   # MT5 con una vela de desfase
    res2 = compare(duk, shifted, "DAX")
    assert res2["status"] == "no_consistente" and res2["clock_warning"] and res2["best_lag"] != 0
    assert status_for(0.93) == "aviso"


def test_append_alerts_deduplicates(tmp_path):
    a = pd.DataFrame({"ts": [pd.Timestamp("2024-02-15 10:00", tz="UTC")], "target": ["DAX"], "z": [2.5]})
    path = tmp_path / "alerts.parquet"
    assert len(append_alerts(path, a)) == 1
    assert append_alerts(path, a).empty
    assert len(pd.read_parquet(path)) == 1


def test_seconds_to_next_step():
    base = pd.Timestamp("2024-02-15 10:00:00", tz="UTC").timestamp()
    assert seconds_to_next_step(base + 5) == 5
    assert seconds_to_next_step(base + 12) == pytest.approx(298)


def test_sync_results_from_git(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    git = lambda *a, cwd=work: subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", *a], cwd=cwd, check=True, capture_output=True)
    git("checkout", "-q", "--orphan", "results")
    (work / "run_meta.json").write_text(json.dumps({"data_end": "2024-06-27", "steps": {"daily": {"status": "ok"}}}))
    git("add", "-A")
    git("commit", "-q", "-m", "r")
    git("push", "-q", "origin", "results")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True)
    st = sync_results.sync(clone, tmp_path / "res", today=date(2024, 6, 28))
    assert st.ok and st.data_end == date(2024, 6, 27) and not st.stale
    st2 = sync_results.read_status(tmp_path / "res", today=date(2024, 7, 2))
    assert st2.stale  # el lunes 1 de julio falta
    bad = sync_results.sync(tmp_path / "no_repo", tmp_path / "res", today=date(2024, 6, 28))
    assert bad.stale and "Sin conexión" in bad.message


def test_live_engine_matches_track_record():
    """Paridad: con los mismos datos, el motor en vivo y el historial dan el mismo z y las mismas alertas."""
    shocks = [("DAX", f"2024-04-15 11:{m:02d}", 0.004) for m in (0, 5, 10, 15, 20, 25)]
    bars = make_bars(end="2024-04-16", seed=21, shocks=shocks)
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    hist = selection_history(["DAX", "SPX"], factors={"BRENT": True, "USD": True})
    session = pd.Timestamp("2024-04-15")

    full = tr.run_track_record(bars, U, hist, events, RunLog())
    prior = full.intraday_r2[full.intraday_r2["session"] < session]
    thresholds = {"targets": {}}
    for t in ("DAX", "SPX"):
        r2p = prior[prior["target"] == t]["r2"].tail(250)
        thresholds["targets"][t] = {"z_star": 2.3, "r2_median": float(r2p.median())}
    hist_alerts, _, _ = tr.run_target(bars, U, "DAX", hist, events, since=session, prior_r2=prior[prior["target"] == "DAX"].set_index("session")["r2"], z_fixed=2.3)
    hist_alerts = hist_alerts[hist_alerts["session"] == session]

    sel_row = hist[(hist["valid_from"] <= session)]
    last = sel_row[sel_row["as_of"] == sel_row["as_of"].max()]
    selection = {"targets": {t: {"factors": last[(last["target"] == t) & last["selected"]]["factor"].tolist(), "status": "ok"} for t in ("DAX", "SPX")}}
    engine = LiveEngine(U, selection, thresholds, events)
    now = pd.Timestamp("2024-04-15 15:31", tz="UTC")   # tras el cierre europeo (verano: 15:30 UTC)
    live_bars = {k: v[v.index < now.floor("5min")] for k, v in bars.items()}
    evals, live_alerts = engine.step(live_bars, now)
    live_alerts = live_alerts[live_alerts["target"] == "DAX"]

    assert len(hist_alerts) >= 1
    cols = ["ts", "L", "z", "gap", "main_factor", "origin"]
    pd.testing.assert_frame_equal(
        hist_alerts[cols].reset_index(drop=True), live_alerts[cols].reset_index(drop=True), check_dtype=False
    )
