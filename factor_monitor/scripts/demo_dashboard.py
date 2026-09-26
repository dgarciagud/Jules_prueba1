"""Genera un escenario de demostración para el dashboard con datos sintéticos.

    python scripts/demo_dashboard.py demo/        # crea demo/results, demo/state y demo/universe.yaml
    FM_RESULTS=demo/results FM_STATE=demo/state FM_UNIVERSE=demo/universe.yaml streamlit run src/factor_monitor/live/app.py
"""

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_intraday_alerts import RAW, make_bars  # noqa: E402

from factor_monitor.common.universe import parse_universe  # noqa: E402
from factor_monitor.live.alerts import append_alerts  # noqa: E402
from factor_monitor.live.engine import LiveEngine  # noqa: E402
from factor_monitor.nightly import run as nightly  # noqa: E402
from factor_monitor.nightly.store import BarStore, LocalBackend  # noqa: E402


def demo_raw() -> dict:
    """Universo de los tests con IDs de Dukascopy y símbolos MT5 ficticios (el nocturno solo lee descargables)."""
    raw = {**RAW, "instruments": {k: {**v, "dukascopy_id": k.lower(), "mt5_symbol": f"{k}.dwx"} for k, v in RAW["instruments"].items()}}
    return raw


def build(out: Path, now: pd.Timestamp = pd.Timestamp("2024-04-15 11:36", tz="UTC")) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    raw = demo_raw()
    (out / "universe.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    universe = parse_universe(raw)
    shocks = [("DAX", f"2024-04-15 11:{m:02d}", 0.004) for m in (0, 5, 10, 15, 20, 25)]
    bars = make_bars(start="2023-07-03", end="2024-04-16", seed=21, shocks=shocks)
    store = BarStore(LocalBackend(out / "remote"), out / "cache")
    store.write({(k, y): g for k, v in bars.items() for y, g in v.groupby(v.index.year)})
    nightly.yesterday_utc = lambda: date(2024, 4, 13)
    args = SimpleNamespace(results_dir=str(out / "results"), skip_download=True, until=None, engine="jetta",
                           years=2, full=True, force_selection=True, universe=str(out / "universe.yaml"))  # fmt: skip
    nightly.run(args, universe, store)

    state = out / "state"
    state.mkdir(exist_ok=True)
    events = pd.DataFrame(columns=["datetime_utc", "event_type", "region", "block_before_min", "block_after_min", "source"])
    engine = LiveEngine.from_results(universe, out / "results", events)
    live_bars = {k: v[v.index < now.floor("5min")] for k, v in bars.items()}
    evals, alerts = engine.step(live_bars, now)
    for t, ev in evals.items():
        ev.to_parquet(state / f"live_{t}.parquet")
    append_alerts(state / "alerts_live.parquet", alerts)
    status = {
        "updated_utc": now.isoformat(), "open_targets": sorted(evals), "server_offset_hours": 3, "offset_source": "detected",
        "results": {"ok": True, "data_end": "2024-04-12", "stale": False, "message": "Datos hasta 2024-04-12"},
        "excluded": engine.excluded,
        "models": {t: {"r2": s.model.r2 if s.model else None, "r2_ok": s.r2_ok, "z_star": s.z_star, "driver_ok": s.driver_ok,
                       "factors": s.model.factors if s.model else [], "note": s.note} for t, s in engine.states.items()},
        "errors": [],
    }  # fmt: skip
    (state / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{"instrument": k, "kind": v["kind"], "mt5_symbol": f"{k}.dwx", "available": True, "reason": ""}
                  for k, v in raw["instruments"].items()]).to_csv(state / "mt5_inventory.csv", index=False)  # fmt: skip
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "demo")))
