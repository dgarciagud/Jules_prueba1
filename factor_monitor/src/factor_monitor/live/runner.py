"""Runner local (SPEC v3 §10.2): único proceso con conexión a MT5.

Al arrancar y en cada apertura de sesión: sincroniza `results`, genera el
inventario MT5, detecta el desfase horario y el test de consistencia.
Cada 5 minutos (10 s después del cierre de la vela): descarga velas cerradas,
evalúa, registra alertas nuevas y escribe el estado para el dashboard.
Streamlit solo lee lo que escribe este proceso.

Uso:  python -m factor_monitor.live.runner [--repo .] [--results results_local] [--state state]
"""

from __future__ import annotations

import argparse
import json
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..common.events import eia_events, load_events
from ..common.universe import Universe, load_universe
from ..sources.mt5 import MT5Error, MT5Source
from .alerts import append_alerts, notify
from .consistency import run_consistency
from .engine import LiveEngine
from .sync_results import SyncStatus, sync

log = logging.getLogger(__name__)

STEP_SECONDS = 300
AFTER_CLOSE_SECONDS = 10


def seconds_to_next_step(now: float) -> float:
    """Segundos hasta 10 s después del próximo cierre de vela de 5 minutos."""
    nxt = (int(now) // STEP_SECONDS + 1) * STEP_SECONDS + AFTER_CLOSE_SECONDS
    wait = nxt - now
    return wait - STEP_SECONDS if wait > STEP_SECONDS else wait


@dataclass
class Runner:
    universe: Universe
    source: MT5Source
    repo_dir: Path
    results_dir: Path
    state_dir: Path
    notify_enabled: bool = False
    engine: LiveEngine | None = None
    sync_status: SyncStatus | None = None
    inventory: pd.DataFrame | None = None
    consistency: pd.DataFrame | None = None
    last_sessions: set = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ arranque / apertura

    def refresh(self) -> None:
        """Sincroniza resultados, inventario, desfase y consistencia; reconstruye el motor."""
        try:
            self.sync_status = sync(self.repo_dir, self.results_dir)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"sync: {e}")
            log.error("Sincronización fallida: %s", e)
        self.inventory = self.source.inventory(self.universe)
        self.inventory.to_csv(self.state_dir / "mt5_inventory.csv", index=False)
        self.source.detect_offset()
        available = set(self.inventory.loc[self.inventory["available"], "instrument"])
        events = load_events()
        today = pd.Timestamp.now(tz="UTC").date()
        eia = eia_events((pd.Timestamp(today) - pd.Timedelta(days=7)).date(), (pd.Timestamp(today) + pd.Timedelta(days=7)).date())
        events = pd.concat([events, eia], ignore_index=True) if not events.empty else eia
        self.engine = LiveEngine.from_results(self.universe, self.results_dir, events, available)
        self.run_consistency(available)

    def run_consistency(self, available: set[str]) -> None:
        path = self.results_dir / "recent_bars.parquet"
        if not path.exists():
            return
        recent = pd.read_parquet(path)
        bars = self.source.closed_bars(self.universe, sorted(available & set(recent["instrument"])), sessions_back=10)
        self.consistency = run_consistency(recent, bars)
        self.consistency.to_csv(self.state_dir / "consistency.csv", index=False)

    # ------------------------------------------------------------------ paso

    def needed(self) -> list[str]:
        need = set()
        for tid in self.universe.targets:
            sel = self.engine.target_selection(tid) if self.engine else None
            need |= self.engine.needed_instruments(tid, sel[0] if sel else [])
        return sorted(need)

    def step(self, now: pd.Timestamp | None = None) -> pd.DataFrame:
        now = now or pd.Timestamp.now(tz="UTC")
        sessions = {tid for tid in self.universe.targets if self.engine.in_session(tid, now) is not None}
        if sessions - self.last_sessions:
            self.refresh()  # apertura de una sesión: datos nocturnos, inventario y desfase al día
        self.last_sessions = sessions
        if not sessions:
            self.write_status(now, [])
            return pd.DataFrame()
        bars = self.source.closed_bars(self.universe, self.needed())
        evals, alerts = self.engine.step(bars, now)
        for tid, ev in evals.items():
            ev.to_parquet(self.state_dir / f"live_{tid}.parquet")
        new = append_alerts(self.state_dir / "alerts_live.parquet", alerts)
        for _, a in new.iterrows():
            notify(a, self.notify_enabled)
        self.write_status(now, sorted(sessions))
        return new

    def write_status(self, now: pd.Timestamp, open_targets: list[str]) -> None:
        off = self.source.offset
        status = {
            "updated_utc": now.isoformat(),
            "open_targets": open_targets,
            "server_offset_hours": off.hours if off else None,
            "offset_source": off.source if off else None,
            "results": vars(self.sync_status) if self.sync_status else None,
            "excluded": self.engine.excluded if self.engine else {},
            "models": {
                t: {"r2": s.model.r2 if s.model else None, "r2_ok": s.r2_ok, "z_star": s.z_star, "driver_ok": s.driver_ok,
                    "factors": s.model.factors if s.model else [], "note": s.note}
                for t, s in (self.engine.states.items() if self.engine else [])
            },  # fmt: skip
            "errors": self.errors[-20:],
        }
        tmp = self.state_dir / "status.tmp"
        tmp.write_text(json.dumps(status, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_dir / "status.json")

    # ------------------------------------------------------------------ bucle

    def loop(self) -> None:
        self.refresh()
        while True:
            try:
                self.step()
            except MT5Error as e:
                self.errors.append(f"{datetime.now(timezone.utc):%H:%M} MT5: {e}")
                log.error("MT5: %s; reintento en el próximo paso", e)
                try:
                    self.source.connect()
                except MT5Error:
                    pass
            except Exception as e:  # noqa: BLE001 - el runner no se detiene por un paso fallido
                self.errors.append(f"{datetime.now(timezone.utc):%H:%M} {type(e).__name__}: {e}")
                log.exception("Paso fallido")
            _time.sleep(seconds_to_next_step(_time.time()))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="runner")
    p.add_argument("--repo", default=".", help="Clon del repositorio (para sincronizar la rama results)")
    p.add_argument("--results", default="results_local")
    p.add_argument("--state", default="state")
    p.add_argument("--universe", default=None)
    p.add_argument("--notify", action="store_true", help="Notificaciones de escritorio (requiere plyer)")
    p.add_argument("--reference-symbol", default="EURUSD", help="Símbolo con ticks frecuentes para detectar la hora del servidor")
    args = p.parse_args(argv)
    universe = load_universe(args.universe) if args.universe else load_universe()
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    source = MT5Source(state_path=state / "local_state.json", reference_symbol=args.reference_symbol)
    source.connect()
    Runner(universe, source, Path(args.repo), Path(args.results), state, args.notify).loop()


if __name__ == "__main__":
    main()
