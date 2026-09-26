import lzma
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_monitor.common.bars import m1_to_bars, merge_bars, quality_report
from factor_monitor.sources import dukascopy as dk


def make_bi5(rows, factor):
    """rows: (segundos, open, close, low, high, volumen) en precio real."""
    arr = np.array(
        [(t, round(o * factor), round(c * factor), round(l * factor), round(h * factor), v) for t, o, c, l, h, v in rows],
        dtype=dk.CANDLE_DTYPE,
    )
    return lzma.compress(arr.tobytes(), format=lzma.FORMAT_ALONE)


def test_candle_url_month_zero_indexed():
    url = dk.candle_url("usa500idxusd", date(2024, 1, 3), "bid")
    assert url.endswith("/USA500IDXUSD/2024/00/03/BID_candles_min_1.bi5")
    assert "/2024/11/31/" in dk.candle_url("eurusd", date(2024, 12, 31), "ask")


def test_parse_candles_bi5_scale_and_flats():
    raw = make_bi5([(0, 1.1, 1.2, 1.0, 1.3, 5.0), (60, 1.2, 1.2, 1.2, 1.2, 0.0), (120, 1.2, 1.25, 1.19, 1.26, 2.0)], 1e5)
    df = dk.parse_candles_bi5(raw, date(2024, 1, 3), 1e5)
    assert len(df) == 2  # la vela plana de volumen 0 se descarta
    assert df.index[0] == pd.Timestamp("2024-01-03 00:00", tz="UTC")
    assert df.index[1] == pd.Timestamp("2024-01-03 00:02", tz="UTC")
    first = df.iloc[0]
    assert (first.open, first.high, first.low, first.close) == pytest.approx((1.1, 1.3, 1.0, 1.2))
    assert dk.parse_candles_bi5(b"", date(2024, 1, 3), 1e5).empty


def test_fetch_m1_bi5_joins_sides_and_skips_saturday():
    calls = []

    def getter(url):
        calls.append(url)
        side = 1.0 if "BID" in url else 1.001
        return make_bi5([(0, 100 * side, 100 * side, 100 * side, 100 * side, 1.0)], 1000)

    m1 = dk.fetch_m1_bi5("usa500idxusd", date(2024, 1, 5), date(2024, 1, 7), 1000, getter=getter)
    assert len(calls) == 4  # viernes y domingo, bid + ask
    assert len(m1) == 2
    assert (m1["ask_close"] > m1["bid_close"]).all()


def test_m1_to_bars_convention_and_filters():
    idx = pd.date_range("2024-01-15 08:00", periods=10, freq="1min", tz="UTC")
    bid = np.arange(10, dtype=float) + 100
    m1 = pd.DataFrame({f"bid_{f}": bid for f in ("open", "high", "low", "close")}, index=idx)
    for f in ("open", "high", "low", "close"):
        m1[f"ask_{f}"] = bid + 0.2
    m1.loc[idx[7], "ask_close"] = m1.loc[idx[7], "bid_close"]  # ask <= bid: se descarta
    bars = m1_to_bars(m1)
    assert list(bars.index) == [pd.Timestamp("2024-01-15 08:00", tz="UTC"), pd.Timestamp("2024-01-15 08:05", tz="UTC")]
    b0, b1 = bars.iloc[0], bars.iloc[1]
    assert b0.open == pytest.approx(100.1) and b0.close == pytest.approx(104.1)
    assert b0.high == pytest.approx(104.1) and b0.low == pytest.approx(100.1)
    assert b0.spread == pytest.approx(0.2) and b0.n == 5
    assert b1.n == 4


def test_merge_bars_replaces_from():
    idx = pd.date_range("2024-01-15", periods=4, freq="1D", tz="UTC")
    old = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    new = pd.DataFrame({"close": [30.0, 50.0]}, index=[idx[2], idx[3] + pd.Timedelta(days=1)])
    out = merge_bars(old, new, replace_from=idx[2])
    assert out["close"].tolist() == [1.0, 2.0, 30.0, 50.0]  # la fila del día 4 se sustituye


def test_node_command_and_fallback(tmp_path):
    cmd = dk.node_command("eurusd", date(2024, 1, 1), date(2024, 1, 31), "bid", tmp_path, "x")
    assert cmd[cmd.index("--date-to") + 1] == "2024-02-01"

    def failing_runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    def getter(url):
        return make_bi5([(0, 1.1, 1.1, 1.1, 1.1, 1.0)], 1e5) if "BID" in url else make_bi5([(0, 1.2, 1.2, 1.2, 1.2, 1.0)], 1e5)

    m1 = dk.download_m1("eurusd", date(2024, 1, 2), date(2024, 1, 2), 1e5, engine="node", runner=failing_runner, getter=getter)
    assert len(m1) == 1
    with pytest.raises(dk.DownloadError):
        dk.download_m1("eurusd", date(2024, 1, 2), date(2024, 1, 2), 1e5, engine="node", runner=failing_runner, fallback=False)


def test_node_csv_parsing_and_range(tmp_path):
    def runner(cmd, **kw):
        d = Path(cmd[cmd.index("--directory") + 1])
        name = cmd[cmd.index("--file-name") + 1]
        side = cmd[cmd.index("--price-type") + 1]
        p = 1.1 if side == "bid" else 1.2
        ts = [pd.Timestamp("2024-01-02 00:00", tz="UTC"), pd.Timestamp("2024-01-03 00:00", tz="UTC")]
        rows = [f"{int(t.timestamp() * 1000)},{p},{p},{p},{p},1" for t in ts]
        (d / f"{name}.csv").write_text("timestamp,open,high,low,close,volume\n" + "\n".join(rows))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    m1 = dk.fetch_m1_node("eurusd", date(2024, 1, 2), date(2024, 1, 2), runner=runner)
    assert len(m1) == 1  # el 3 de enero queda fuera del rango pedido
    assert m1["ask_close"].iloc[0] == pytest.approx(1.2)


def test_quality_report_detects_jump():
    idx = pd.date_range("2024-01-15 08:00", periods=300, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 300)))
    close[150:] *= 1.02
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "spread": 0.01, "n": 5.0}, index=idx)
    q = quality_report(bars, "X")
    assert q["jumps_10mad"].sum() == 1
    assert q["minute_coverage"].iloc[0] == pytest.approx(1.0)


def test_node_retries_on_rate_limit(tmp_path):
    calls, waits = [], []

    def runner(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, "Something went wrong:\n > Request failed with status 429", "(node) Warning: EnvHttpProxyAgent is experimental")
        d = Path(cmd[cmd.index("--directory") + 1])
        name = cmd[cmd.index("--file-name") + 1]
        ms = int(pd.Timestamp("2024-01-02", tz="UTC").timestamp() * 1000)
        p = 1.1 if "bid" in cmd else 1.2
        (d / f"{name}.csv").write_text(f"timestamp,open,high,low,close,volume\n{ms},{p},{p},{p},{p},1")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    m1 = dk.fetch_m1_node("eurusd", date(2024, 1, 2), date(2024, 1, 2), runner=runner, sleep=waits.append)
    assert len(m1) == 1 and waits == [60.0] and len(calls) == 3


def test_node_empty_day_is_empty_frame(tmp_path):
    def runner(cmd, **kw):
        d = Path(cmd[cmd.index("--directory") + 1])
        (d / f"{cmd[cmd.index('--file-name') + 1]}.csv").write_text("")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert dk.fetch_m1_node("eurusd", date(2024, 1, 1), date(2024, 1, 1), runner=runner).empty


def test_http_get_backs_off_on_429(monkeypatch):
    import urllib.error
    import urllib.request

    attempts, waits = [], []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(url, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert dk.http_get("https://x", sleep=waits.append) == b"ok"
    assert waits == [15.0, 30.0]


DATA = Path(__file__).parent / "data"


def test_jetta_decoder_matches_dukascopy_node():
    raw = (DATA / "jetta_usa500_20260924_bid.json").read_bytes()
    ours = dk.decode_jetta_candles(raw)
    ref = pd.read_csv(DATA / "node_usa500_20260924_bid.csv")
    ref.index = pd.to_datetime(ref["timestamp"], unit="ms", utc=True)
    assert len(ours) == len(ref) == 1335
    assert (ours.index == ref.index).all()
    for c in ("open", "high", "low", "close"):
        np.testing.assert_allclose(ours[c].to_numpy(), ref[c].to_numpy(), rtol=0, atol=1e-9)
    assert ours.index[0] == pd.Timestamp("2026-09-24 00:00", tz="UTC")
    assert dk.decode_jetta_candles(b"").empty
    assert dk.decode_jetta_candles(b'{"timestamp": 0, "times": []}').empty


def test_jetta_url_and_code():
    assert dk.jetta_code("USA500.IDX/USD") == "USA500.IDX-USD"
    assert dk.jetta_url("BRENT.CMD-USD", date(2024, 1, 3), "bid").endswith("/BRENT.CMD-USD/BID/2024/1/3")


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def now(self):
        return self.t


def test_pacer_spacing_and_backoff(monkeypatch):
    import urllib.error
    import urllib.request

    clock = FakeClock()
    pacer = dk.Pacer(interval=20.0, rate_limit_pause=60.0, sleep=clock.sleep, clock=clock.now)
    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(url, timeout):
        calls.append((url, clock.t))
        if len(calls) == 2:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dk.http_get_paced("u1", pacer)
    dk.http_get_paced("u2", pacer)
    times = [t for _, t in calls]
    assert times[1] - times[0] == 20.0          # ritmo fijo
    assert times[2] - times[1] >= 60.0          # espera larga tras el 429
    assert pacer.rate_limited == 1 and pacer.requests == 3


def test_fetch_jetta_skips_saturday_and_bid_only():
    raw = (DATA / "jetta_usa500_20260924_bid.json").read_bytes()
    urls = []

    def getter(url):
        urls.append(url)
        return raw

    m1 = dk.fetch_m1_jetta("USA500.IDX-USD", date(2026, 9, 25), date(2026, 9, 27), sides=("bid",), getter=getter)
    assert len(urls) == 2 and all("/BID/" in u for u in urls)   # viernes y domingo, sin sábado
    assert (m1["ask_close"] == m1["bid_close"]).all()
    bars5 = m1_to_bars(m1, bid_only=True)
    assert len(bars5) > 0 and bars5["spread"].isna().all()
    assert m1_to_bars(m1).empty   # sin el modo bid_only, ask == bid se descarta
