import os
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo_dashboard import build  # noqa: E402

from factor_monitor.live.dashboard_data import Sources, cumulative_paths, data_status, decomposition, monitor_table, regime_table  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "factor_monitor" / "live" / "app.py")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return build(tmp_path_factory.mktemp("demo"))


def test_data_builders(demo):
    src = Sources(demo / "results", demo / "state")
    reg = regime_table(src.shapley(), src.daily_r2(), src.selection())
    assert set(reg["objetivo"]) == {"DAX", "SPX"} and reg["R²"].notna().all()
    now = pd.Timestamp("2024-04-15 11:36", tz="UTC")
    evals = {t: src.live_eval(t) for t in ("DAX", "SPX")}
    mon = monitor_table(["DAX", "SPX"], evals, src.status(), src.alerts(), src.track_record(), src.inventory(), now)
    assert list(mon.columns[:3]) == ["objetivo", "en MT5", "z*"]
    ratios = mon["|z|/z*"].dropna().tolist()
    assert ratios == sorted(ratios, reverse=True)   # ordenada por |z|/z*
    assert mon.set_index("objetivo").loc["SPX", "estado"] == "sesión cerrada"   # 11:36 UTC
    paths = cumulative_paths(evals["DAX"])
    assert set(paths["serie"]) == {"Real", "Implícito"}
    assert not decomposition(evals["DAX"], 6).empty
    levels = [lvl for lvl, _ in data_status(src.run_meta(), src.status(), now)]
    assert "error" not in levels


def test_data_status_flags_missing_runner():
    msgs = data_status(None, None, pd.Timestamp("2024-04-15 12:00", tz="UTC"))
    assert [lvl for lvl, _ in msgs] == ["error", "error"]


def test_app_renders_all_views(demo, monkeypatch):
    monkeypatch.setenv("FM_RESULTS", str(demo / "results"))
    monkeypatch.setenv("FM_STATE", str(demo / "state"))
    monkeypatch.setenv("FM_UNIVERSE", str(demo / "universe.yaml"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Régimen", "Monitor", "Detalle", "Estado del sistema"]
    assert len(at.dataframe) >= 4


def test_app_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("FM_RESULTS", str(tmp_path / "r"))
    monkeypatch.setenv("FM_STATE", str(tmp_path / "s"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("runner" in e.value for e in at.error)
