from datetime import date

import numpy as np
import pandas as pd
import pytest

from factor_monitor.common.alerts import find_alerts
from factor_monitor.common.events import block_mask, eia_events, load_events, validate
from factor_monitor.common.intraday import MKT, IntradayModel, build_frame, last_sessions
from factor_monitor.common.universe import parse_universe

RAW = {
    "instruments": {
        "DAX": {"kind": "target", "session": "eu", "exchange": "XETR"},
        "SX5E": {"kind": "target", "session": "eu", "exchange": "XETR"},
        "CAC": {"kind": "target", "session": "eu", "exchange": "XPAR"},
        "SPX": {"kind": "target", "session": "us", "exchange": "XNYS"},
        "BRENT": {"kind": "factor"},
        "EURUSD": {"kind": "factor"},
    },
    "factors": {"BRENT": {"instrument": "BRENT"}, "USD": {"instrument": "EURUSD", "sign": -1}},
    "targets": {
        "DAX": {"session": "eu", "exchange": "XETR", "candidates": ["BRENT", "USD"], "market": {"type": "mean", "members": ["SX5E", "CAC"]}},
        "SPX": {"session": "us", "exchange": "XNYS", "candidates": ["BRENT", "USD"], "market": {"type": "none"}},
    },
}
U = parse_universe(RAW)


def make_bars(start="2024-01-02", end="2024-04-30", seed=3, shocks=None):
    """Velas de 07:00 a 21:00 UTC. DAX = MKT + 0.8·BRENT + ruido AR(1) con patrón intradía.

    `shocks`: lista de (instrumento, marca UTC, retorno añadido en esa vela).
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    idx = pd.DatetimeIndex(
        [t for d in days for t in pd.date_range(d + pd.Timedelta(hours=7), d + pd.Timedelta(hours=21), freq="5min", inclusive="left")],
        tz="UTC",
    )
    n = len(idx)
    s = 0.0005
    hour = idx.hour + idx.minute / 60
    scale = np.where((hour >= 13.5) & (hour < 15), 2.0, 1.0)  # más volatilidad a la apertura de EE.UU.
    m = rng.normal(0, s, n) * scale
    brent = rng.normal(0, s, n)
    usd = rng.normal(0, s, n)
    e = rng.normal(0, s / 2, n) * scale
    noise = np.empty(n)
    noise[0] = e[0]
    for i in range(1, n):
        noise[i] = 0.2 * noise[i - 1] + e[i]
    inc = {
        "SX5E": m + rng.normal(0, s / 5, n),
        "CAC": m + rng.normal(0, s / 5, n),
        "DAX": m + 0.8 * brent + noise,
        "SPX": 0.6 * usd + rng.normal(0, s / 2, n),
        "BRENT": brent,
        "EURUSD": -usd,
    }
    for inst, ts, r in shocks or []:
        inc[inst][idx.get_loc(pd.Timestamp(ts, tz="UTC"))] += r
    out = {}
    for k, r in inc.items():
        c = 100 * np.exp(np.cumsum(r))
        out[k] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "spread": 0.01, "n": 5.0}, index=pd.DatetimeIndex(idx, name="ts"))
    return out


@pytest.fixture(scope="module")
def frame():
    return build_frame(make_bars(), U.targets["DAX"], U, ["BRENT", "USD"])


def fit_dax(frame, before="2024-02-01"):
    train = last_sessions(frame, pd.Timestamp(before), 10)
    return IntradayModel.fit(train, "DAX", ["BRENT", "USD"], has_market=True)


def test_frame_only_in_session_without_holidays(frame):
    assert frame["minute"].min() == 0 and frame["minute"].max() == 505
    assert pd.Timestamp("2024-03-29") not in set(frame["session"])  # Viernes Santo
    assert set(frame.columns) >= {"y", MKT, "BRENT", "USD", "session", "bucket"}


def test_fit_recovers_betas_and_bucket_sigma(frame):
    model = fit_dax(frame)
    assert len(model.sessions) == 10
    betas = model.ridge.betas
    assert betas[MKT] == pytest.approx(1.0, abs=0.1)
    assert betas["BRENT"] == pytest.approx(0.8, abs=0.1)
    assert abs(betas["USD"]) < 0.1
    assert model.rho[0] == pytest.approx(0.2, abs=0.1)
    # la franja de 14:30-15:00 hora de París (13:30-14:00 UTC en invierno) tiene el doble de σ
    assert model.sigma_bucket.loc[11] > 1.6 * model.sigma_bucket.loc[2]


def test_z_frequency_matches_theory(frame):
    # Como en producción: cada sesión se evalúa con un modelo estimado en las 10 anteriores.
    parts = []
    for sess in sorted(s for s in frame["session"].unique() if s >= pd.Timestamp("2024-02-01")):
        model = fit_dax(frame, before=sess)
        parts.append(model.evaluate(frame[frame["session"] == sess]))
    ev = pd.concat(parts)
    for L in (6, 12):
        z = ev[f"z_{L}"].dropna()
        assert len(z) > 3000
        assert abs((z.abs() >= 2).mean() - 0.0455) < 0.015, L


def test_windows_do_not_cross_sessions(frame):
    model = fit_dax(frame)
    ev = model.evaluate(frame[frame["session"].isin([pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-02")])])
    first = ev[ev["session"] == pd.Timestamp("2024-02-02")]
    assert first["gap_6"].iloc[:5].isna().all() and np.isfinite(first["gap_6"].iloc[5])
    assert not first["eligible"].iloc[:6].any() and first["eligible"].iloc[6]


def evaluate_with_shock(shocks, session="2024-02-15"):
    f = build_frame(make_bars(shocks=shocks), U.targets["DAX"], U, ["BRENT", "USD"])
    model = fit_dax(f, before=session)
    ev = model.evaluate(f[f["session"] == pd.Timestamp(session)])
    return model, ev


def test_injected_target_dislocation():
    shocks = [("DAX", f"2024-02-15 11:{m:02d}", 0.005) for m in (0, 5, 10, 15, 20, 25)]
    model, ev = evaluate_with_shock(shocks)
    alerts = find_alerts(ev, "DAX", 2.0, model.regressors)
    hit = alerts[(alerts["ts"] >= pd.Timestamp("2024-02-15 11:00", tz="UTC")) & (alerts["ts"] <= pd.Timestamp("2024-02-15 12:00", tz="UTC"))]
    assert len(hit) >= 1
    assert hit.iloc[0]["z"] > 0 and hit.iloc[0]["origin"] == "target_moved"


def test_injected_driver_dislocation_and_attribution():
    # Brent sube con fuerza y el DAX no le sigue (los shocks se añaden después de
    # construir el DAX, así que no se transmiten).
    shocks = [("BRENT", f"2024-02-15 11:{m:02d}", 0.005) for m in (0, 5, 10, 15, 20, 25)]
    model, ev = evaluate_with_shock(shocks)
    alerts = find_alerts(ev, "DAX", 2.0, model.regressors)
    hit = alerts[(alerts["ts"] >= pd.Timestamp("2024-02-15 11:00", tz="UTC")) & (alerts["ts"] <= pd.Timestamp("2024-02-15 12:00", tz="UTC"))]
    assert len(hit) >= 1
    first = hit.iloc[0]
    assert first["z"] < 0 and first["origin"] == "driver_moved" and first["main_factor"] == "BRENT"


def synthetic_evals(z_values, start="2024-02-15 09:00"):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(z_values), freq="5min")
    z = np.asarray(z_values, float)
    return pd.DataFrame(
        {
            "session": pd.Timestamp("2024-02-15"), "minute": 60 + 5 * np.arange(len(z)), "eligible": True,
            "z_6": z, "z_12": np.nan, "gap_6": z * 1e-3, "implied_6": 0.0, "real_6": z * 1e-3,
            "origin_6": "mixed", "contrib_6_MKT": 0.0,
        },
        index=idx,
    )  # fmt: skip


def test_cooldown_needs_time_and_zero_cross():
    # Alerta en 0; z sigue alto 12 velas (60 min) sin cruzar 0 → no hay segunda alerta.
    ev = synthetic_evals([2.5] * 13)
    assert len(find_alerts(ev, "DAX", 2.0, [MKT])) == 1
    # Cruza 0 enseguida pero no han pasado 30 min: la segunda llega a los 30 min.
    ev = synthetic_evals([2.5, -0.1, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    a = find_alerts(ev, "DAX", 2.0, [MKT])
    assert len(a) == 2 and (a["ts"].iloc[1] - a["ts"].iloc[0]) == pd.Timedelta(minutes=30)


def test_alert_filters():
    ev = synthetic_evals([2.5, 0, 0, 0, 0, 0, 0, 0])
    assert find_alerts(ev, "DAX", 2.0, [MKT], driver_ok=False).empty
    assert find_alerts(ev, "DAX", 2.0, [MKT], r2_ok=pd.Series({pd.Timestamp("2024-02-15"): False})).empty
    assert find_alerts(ev, "DAX", 3.0, [MKT]).empty
    blocked = pd.Series(True, index=ev.index)
    assert find_alerts(ev, "DAX", 2.0, [MKT], blocked=blocked).empty
    ev2 = ev.copy()
    ev2["eligible"] = False
    assert find_alerts(ev2, "DAX", 2.0, [MKT]).empty


def test_event_blocks():
    events = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(["2024-01-31 19:00", "2024-02-01 12:00"], utc=True),
            "event_type": ["FOMC", "OPEC"], "region": ["US", "INT"],
            "block_before_min": [np.nan, np.nan], "block_after_min": [np.nan, np.nan], "source": ["fed", "opec"],
        }
    )  # fmt: skip
    idx = pd.date_range("2024-01-31 18:40", "2024-01-31 20:20", freq="5min", tz="UTC")
    m = block_mask(idx, events, "SPX", U)
    blocked = idx[m.to_numpy()]
    assert blocked.min() == pd.Timestamp("2024-01-31 18:45", tz="UTC")
    assert blocked.max() == pd.Timestamp("2024-02-01 20:10", tz="UTC") - pd.Timedelta(days=1)
    day = pd.date_range("2024-02-01 08:00", periods=3, freq="5min", tz="UTC")
    assert block_mask(day, events, "DAX", U).all()          # DAX tiene candidato Brent
    no_oil = parse_universe({**RAW, "targets": {"SPX": {"session": "us", "candidates": ["USD"], "market": {"type": "none"}}}})
    assert not block_mask(day, events, "SPX", no_oil).any()


def test_eia_rule_dst_and_validation(tmp_path):
    eia = eia_events(date(2024, 3, 1), date(2024, 3, 31))
    times = eia["datetime_utc"].dt.strftime("%m-%d %H:%M").tolist()
    assert times[0] == "03-06 15:30" and times[1] == "03-13 14:30"  # EE.UU. cambia de hora el 10 de marzo
    assert validate(load_events(tmp_path / "no.csv"), date(2024, 1, 1), date(2024, 12, 31))[0].startswith("events.csv vacío")
    ev = pd.DataFrame(
        {"datetime_utc": pd.to_datetime(["2024-01-31 19:00"] * 2, utc=True), "event_type": ["FOMC"] * 2,
         "region": ["US"] * 2, "block_before_min": np.nan, "block_after_min": np.nan, "source": ["fed", None]}
    )  # fmt: skip
    w = validate(ev, date(2024, 1, 1), date(2024, 12, 31))
    assert any("duplicados" in x for x in w) and any("sin fuente" in x for x in w) and any("FOMC 2024" in x for x in w)
