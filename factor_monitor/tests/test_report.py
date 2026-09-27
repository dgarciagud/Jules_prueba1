import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo_dashboard import build  # noqa: E402

from factor_monitor.nightly.report import build_report, write_report  # noqa: E402


def test_report_from_demo(tmp_path):
    results = build(tmp_path / "demo") / "results"
    path = write_report(results)
    html = path.read_text(encoding="utf-8")
    assert (results / ".nojekyll").exists()
    for h in ("Estado", "Selección de la semana", "Régimen de factores", "Umbrales e historial de alertas"):
        assert f"<h2>{h}</h2>" in html
    assert html.count("<svg") >= 2 and "DAX" in html and "None" not in html
    assert "prefers-color-scheme:dark" in html


def test_report_without_data(tmp_path):
    html = build_report(tmp_path, datetime(2026, 9, 27, tzinfo=timezone.utc))
    assert "<h1>Monitor de factores</h1>" in html and "Traceback" not in html
