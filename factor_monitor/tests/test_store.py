import json

import numpy as np
import pandas as pd
import pytest

from factor_monitor.nightly import build_bars
from factor_monitor.nightly.store import MANIFEST, BarStore, LocalBackend


def bars(start, n, value=1.0):
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC", name="ts")
    return pd.DataFrame(
        {"open": value, "high": value, "low": value, "close": value, "spread": 0.1, "n": 5.0}, index=idx
    )


def test_write_read_and_garbage(tmp_path):
    backend = LocalBackend(tmp_path / "remote")
    store = BarStore(backend, tmp_path / "cache")
    store.write({("SPX", 2024): bars("2024-01-15 14:30", 10)})
    store.write({("SPX", 2024): bars("2024-01-15 14:30", 12, 2.0)})
    assets = backend.list_assets()
    assert MANIFEST in assets and len([a for a in assets if a.startswith("SPX_2024")]) == 1

    fresh = BarStore(backend, tmp_path / "cache2")
    df = fresh.read("SPX", 2024)
    assert len(df) == 12 and df["close"].iloc[0] == 2.0
    assert fresh.last_timestamp("SPX") == df.index.max()


class FlakyBackend(LocalBackend):
    def __init__(self, root, fail_on):
        super().__init__(root)
        self.fail_on = fail_on

    def upload(self, path, name):
        if self.fail_on(name):
            raise RuntimeError(f"fallo simulado subiendo {name}")
        super().upload(path, name)


def test_failure_mid_write_keeps_previous_manifest_valid(tmp_path):
    root = tmp_path / "remote"
    BarStore(LocalBackend(root), tmp_path / "c0").write(
        {("SPX", 2024): bars("2024-01-15 14:30", 10), ("DAX", 2024): bars("2024-01-15 08:00", 10)}
    )
    # Falla la subida del segundo fichero: el manifiesto anterior sigue siendo válido.
    flaky = FlakyBackend(root, fail_on=lambda n: n.startswith("SPX_"))
    store = BarStore(flaky, tmp_path / "c1")
    with pytest.raises(RuntimeError):
        store.write({("DAX", 2024): bars("2024-01-15 08:00", 20, 3.0), ("SPX", 2024): bars("2024-01-15 14:30", 20, 3.0)})
    after = BarStore(LocalBackend(root), tmp_path / "c2")
    assert len(after.read("DAX", 2024)) == 10 and len(after.read("SPX", 2024)) == 10
    # Un escritor posterior limpia el fichero huérfano de DAX.
    after.write({("SPX", 2024): bars("2024-01-15 14:30", 11)})
    assert len([a for a in LocalBackend(root).list_assets() if a.startswith("DAX_")]) == 1


def test_failure_on_manifest_upload_rebuilds_from_latest(tmp_path):
    root = tmp_path / "remote"
    BarStore(LocalBackend(root), tmp_path / "c0").write({("SPX", 2024): bars("2024-01-15 14:30", 10)})
    (root / MANIFEST).unlink()  # manifiesto perdido
    flaky = FlakyBackend(root, fail_on=lambda n: n == MANIFEST)
    with pytest.raises(RuntimeError):
        BarStore(flaky, tmp_path / "c1").write({("SPX", 2024): bars("2024-01-15 14:30", 15)})
    rebuilt = BarStore(LocalBackend(root), tmp_path / "c2")
    assert len(rebuilt.read("SPX", 2024)) == 15  # la versión más reciente


def test_matrix_chunks_years():
    m = build_bars.build_matrix([f"I{i}" for i in range(30)], 2017, 2026)
    assert len(m) <= build_bars.MAX_MATRIX
    assert {e["instrument"] for e in m} == {f"I{i}" for i in range(30)}
    small = build_bars.build_matrix(["SPX"], 2023, 2024)
    assert small == [{"instrument": "SPX", "years": "2023-2023"}, {"instrument": "SPX", "years": "2024-2024"}]


def test_incremental_refreshes_last_days(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    store = BarStore(LocalBackend(remote), tmp_path / "c0")
    store.write({("SPX", 2024): bars("2024-01-10 14:30", 5), ("DAX", 2024): bars("2024-01-10 08:00", 5)})
    requested = {}

    def fake_download(inst, start, end, engine):
        requested[inst.id] = (start, end)
        if inst.id == "DAX":
            raise build_bars.DownloadError("caído")
        return pd.concat([bars("2024-01-10 14:30", 3, 9.0), bars("2024-01-16 14:30", 4, 9.0)])

    monkeypatch.setattr(build_bars, "download_bars", fake_download)
    meta = tmp_path / "run_meta.json"
    build_bars.main(
        ["incremental", "--instruments", "SPX,DAX", "--until", "2024-01-16", "--store-dir", str(remote),
         "--cache", str(tmp_path / "c1"), "--meta", str(meta)]
    )  # fmt: skip
    # 10 ene 2024 (miércoles) menos 3 días hábiles = 5 ene.
    assert requested["SPX"][0].isoformat() == "2024-01-05"
    after = BarStore(LocalBackend(remote), tmp_path / "c2")
    spx = after.read("SPX", 2024)
    assert len(spx) == 3 + 4 and (spx["close"] == 9.0).all()  # las velas antiguas se sustituyen
    assert len(after.read("DAX", 2024)) == 5  # el fallo de DAX no toca sus datos
    data = json.loads(meta.read_text())
    assert data["steps"]["bars"]["status"] == "partial" and data["steps"]["bars"]["failed"] == ["DAX"]


def test_calibrate_factor():
    idx = pd.date_range("2024-01-02 08:00", periods=5, freq="1min", tz="UTC")
    node = pd.DataFrame({"bid_close": [4700.1, 4700.2, 4700.3, 4700.4, 4700.5]}, index=idx)
    raw = pd.DataFrame({"bid_close": node["bid_close"] * 1000}, index=idx)
    ratio, suggested = build_bars.calibrate_factor(node, raw)
    assert ratio == pytest.approx(1000) and suggested == 1000


def test_backfill_fails_when_a_year_cannot_be_downloaded(tmp_path, monkeypatch):
    def fake_download(inst, start, end, engine):
        if start.year == 2023:
            raise build_bars.DownloadError("429")
        return bars(f"{start.year}-01-03 14:30", 3)

    monkeypatch.setattr(build_bars, "download_bars", fake_download)
    monkeypatch.setattr(build_bars, "yesterday_utc", lambda: __import__("datetime").date(2024, 6, 1))
    with pytest.raises(SystemExit) as exc:
        build_bars.main(["backfill", "--instrument", "SPX", "--years", "2023-2024", "--out", str(tmp_path)])
    assert exc.value.code == 1
    assert (tmp_path / "SPX_2024.parquet").exists() and not (tmp_path / "SPX_2023.parquet").exists()
