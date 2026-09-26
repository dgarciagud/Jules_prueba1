import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_intraday_alerts import U, make_bars  # noqa: E402

from factor_monitor.common.events import COLUMNS as EVENT_COLUMNS  # noqa: E402
from factor_monitor.common.runlog import RunLog  # noqa: E402
from factor_monitor.nightly import track_record as tr  # noqa: E402


def iid_session(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-02-15 08:00", periods=n, freq="5min", tz="UTC")
    resid = rng.normal(0, 1, n)
    ev = pd.DataFrame({"session": pd.Timestamp("2024-02-15"), "resid": resid, "y": resid, "implied": 0.0}, index=idx)
    ev["gap_6"] = pd.Series(resid, index=idx).rolling(6).sum()
    return ev


def test_anchored_gap_does_not_fake_convergence():
    rates_anchored, rates_rolling = [], []
    for seed in range(30):
        ev = iid_session(seed=seed)
        thr = 2.0 * np.sqrt(6)
        for pos in np.where(ev["gap_6"].abs() >= thr)[0]:
            if pos + 6 >= len(ev):
                continue
            al = pd.Series({"ts": ev.index[pos], "gap": ev["gap_6"].iloc[pos]})
            rates_anchored.append(tr.outcomes(al, ev, cost_bps=0.0)["converged_6"])
            # Métrica de la v2: la suma móvil 6 velas después ya no contiene los residuos de la alerta.
            rates_rolling.append(float(abs(ev["gap_6"].iloc[pos + 6]) <= 0.5 * abs(al["gap"])))
    assert np.mean(rates_anchored) < 0.25   # azar: ≈ 11 %
    assert np.mean(rates_rolling) > 0.6     # la métrica antigua da convergencia falsa


def test_outcomes_who_adjusts():
    idx = pd.date_range("2024-02-15 10:00", periods=8, freq="5min", tz="UTC")
    ev = pd.DataFrame({"session": pd.Timestamp("2024-02-15"), "y": 0.0, "implied": 0.0, "resid": 0.0}, index=idx)
    # Gap positivo de 10 pb; después el objetivo cae 8 pb (ajusta el objetivo).
    ev.loc[idx[1:7], "y"] = -0.0008 / 6
    ev["resid"] = ev["y"] - ev["implied"]
    al = pd.Series({"ts": idx[0], "gap": 0.001})
    out = tr.outcomes(al, ev, cost_bps=2.0)
    assert out["converged_6"] == 1.0 and out["target_share_6"] == pytest.approx(1.0)
    assert out["net_bps_6"] == pytest.approx(8.0 - 2.0)
    assert np.isnan(out["converged_12"])  # la sesión acaba antes


def selection_history(targets, start="2024-01-05", end="2024-04-26", factors=None, status="ok"):
    rows = []
    for as_of in pd.date_range(start, end, freq="W-FRI"):
        for t in targets:
            for f, sel in (factors or {"BRENT": True, "USD": False}).items():
                rows.append({"as_of": as_of, "valid_from": as_of + pd.Timedelta(days=3), "target": t, "factor": f,
                             "share": 0.5, "sign": 1, "selected": sel, "status": status})  # fmt: skip
    return pd.DataFrame(rows)


def test_selection_for_point_in_time_and_expiry():
    h = selection_history(["DAX"], start="2024-01-05", end="2024-01-19")
    assert tr.selection_for(h, "DAX", pd.Timestamp("2024-01-10")) == (["BRENT"], True)
    assert tr.selection_for(h, "DAX", pd.Timestamp("2024-01-05")) is None      # antes de la primera
    assert tr.selection_for(h, "DAX", pd.Timestamp("2024-02-05")) is None      # caducada
    assert tr.selection_for(h, "SPX", pd.Timestamp("2024-01-10")) is None


def test_calibration_hits_target_rate():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2024-01-02", periods=100)
    idx = pd.DatetimeIndex([t for d in days for t in pd.date_range(d + pd.Timedelta(hours=8), periods=96, freq="5min")], tz="UTC")
    z6 = pd.Series(rng.normal(size=len(idx)), index=idx).rolling(6).sum() / np.sqrt(6)
    ev = pd.DataFrame({"session": idx.tz_localize(None).normalize(), "z_6": z6, "z_12": np.nan}, index=idx)
    base = np.ones(len(ev), dtype=bool)
    z_star = tr.calibrate_z(ev, base, 0.4)
    n = tr.count_alerts(ev[["z_6", "z_12"]].to_numpy(), base, ev["session"].to_numpy(), idx.as_unit("ns").asi8, z_star)
    assert z_star > 2.0 and abs(n - 40) <= 3
    assert tr.calibrate_z(ev.iloc[:96 * 5], base[: 96 * 5], 0.4) == tr.Z_DEFAULT  # poco historial


def fake_alerts():
    rng = np.random.default_rng(2)
    rows = []
    for year in (2022, 2023, 2024):
        for i in range(40):
            rows.append({"ts": pd.Timestamp(f"{year}-03-01", tz="UTC") + pd.Timedelta(hours=i), "target": "DAX",
                         "main_factor": "BRENT", "origin": "driver_moved", "year": year,
                         "converged_6": 1.0, "converged_12": 1.0, "ret_conv_6": 0.001, "ret_conv_12": 0.001,
                         "target_share_6": 0.7, "target_share_12": 0.7, "net_bps_6": 5.0, "net_bps_12": 5.0 if year != 2023 else -1.0})  # fmt: skip
    for i in range(10):
        rows.append({**rows[0], "ts": rows[0]["ts"] + pd.Timedelta(days=i + 1), "main_factor": "USD", "origin": "driver_moved"})
    return pd.DataFrame(rows)


def test_aggregate_hierarchy_and_persistence():
    agg = tr.aggregate(fake_alerts())
    brent = agg[(agg["level"] == "target×factor×origin") & (agg["main_factor"] == "BRENT")].iloc[0]
    assert brent["n"] == 120 and brent["sufficient"] and brent["persistent"] and brent["years_positive"] == 2
    cell = tr.cell_for(agg, "DAX", "USD", "driver_moved")   # USD: 10 alertas → sube de nivel
    assert cell["level"] == "target×origin" and cell["n"] == 130
    assert tr.cell_for(agg, "SPX", "USD", "target_moved") is None


def test_end_to_end_without_look_ahead():
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    hist = selection_history(["DAX", "SPX"], factors={"BRENT": True, "USD": True})
    full_bars = make_bars(end="2024-04-30", seed=11)
    res = tr.run_track_record(full_bars, U, hist, events, RunLog())
    assert not res.intraday_r2.empty and set(res.thresholds["targets"]) == {"DAX", "SPX"}
    assert all(v["z_star"] >= 2.0 for v in res.thresholds["targets"].values())
    assert {"converged_6", "net_bps_12", "main_factor", "origin"} <= set(res.alerts.columns)

    cut = pd.Timestamp("2024-03-28", tz="UTC")
    short_bars = {k: v[v.index < cut] for k, v in full_bars.items()}
    short = tr.run_track_record(short_bars, U, hist, events, RunLog())
    early = lambda a: a[a["ts"] < cut - pd.Timedelta(days=1)][["ts", "target", "z", "z_star", "gap"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(early(res.alerts), early(short.alerts))
    r2_full = res.intraday_r2[res.intraday_r2["session"] < pd.Timestamp("2024-03-27")].reset_index(drop=True)
    r2_short = short.intraday_r2[short.intraday_r2["session"] < pd.Timestamp("2024-03-27")].reset_index(drop=True)
    pd.testing.assert_frame_equal(r2_full, r2_short)


def test_cost_without_spread_is_commission_only():
    idx = pd.date_range("2024-02-01 10:00", periods=50, freq="5min", tz="UTC")
    b = pd.DataFrame({"close": 100.0, "spread": np.nan}, index=idx)
    assert tr.cost_for(b, idx[-1], 1.5) == 1.5
    b["spread"] = 0.02
    assert tr.cost_for(b, idx[-1], 1.5) == pytest.approx(2.0 + 1.5)
