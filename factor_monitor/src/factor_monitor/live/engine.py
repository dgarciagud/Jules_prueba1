"""Motor en vivo: la misma capa intradía y las mismas reglas que el historial nocturno.

Es una clase pura: recibe velas y la hora, devuelve evaluaciones y alertas. No sabe
nada de MT5 ni del reloj, lo que permite el test de paridad con `track_record`.
Por objetivo y sesión:
  - al primer paso de la sesión se estima el modelo con las 10 sesiones anteriores
    y los factores de `selection.json`, y se fija para toda la sesión;
  - en cada paso se evalúan las velas cerradas de la sesión y se aplican las alertas
    con el z* y la mediana de R² de `thresholds.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..common.alerts import ALERT_COLUMNS, find_alerts
from ..common.calendars import is_trading_day
from ..common.events import block_mask
from ..common.intraday import IntradayModel, build_frame, last_sessions
from ..common.sessions import get_session, session_bounds_utc
from ..common.universe import Universe

log = logging.getLogger(__name__)

FIT_SESSIONS = 10
Z_DEFAULT = 2.5


@dataclass
class TargetState:
    target: str
    session: pd.Timestamp
    model: IntradayModel | None
    driver_ok: bool
    z_star: float
    r2_ok: bool
    note: str = ""


@dataclass
class LiveEngine:
    universe: Universe
    selection: dict
    thresholds: dict
    events: pd.DataFrame
    available: set[str] | None = None          # instrumentos con símbolo en MT5 (None = todos)
    states: dict[str, TargetState] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_results(cls, universe: Universe, results_dir: Path, events: pd.DataFrame, available: set[str] | None = None) -> "LiveEngine":
        sel_path, thr_path = results_dir / "selection.json", results_dir / "thresholds.json"
        selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {"targets": {}}
        thresholds = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.exists() else {"targets": {}}
        return cls(universe, selection, thresholds, events, available)

    # ------------------------------------------------------------------ configuración por objetivo

    def target_selection(self, target_id: str) -> tuple[list[str], bool] | None:
        sel = self.selection.get("targets", {}).get(target_id)
        if not sel:
            return None
        return list(sel.get("factors", [])), sel.get("status") == "ok"

    def needed_instruments(self, target_id: str, factors: list[str]) -> set[str]:
        t = self.universe.targets[target_id]
        need = set(t.basket) if t.basket else {target_id}
        need |= set(t.market.members)
        for f in factors:
            need |= set(self.universe.factors[f].instruments)
        return need

    def in_session(self, target_id: str, now: pd.Timestamp) -> pd.Timestamp | None:
        """Fecha local de la sesión en curso del objetivo, o None si está cerrada."""
        t = self.universe.targets[target_id]
        for delta in (0, -1):
            day = (now + pd.Timedelta(days=delta)).tz_convert(get_session(t.session).tz).date()
            start, end = session_bounds_utc(day, t.session)
            if start <= now < end + pd.Timedelta(minutes=10) and is_trading_day(day, t.exchange):
                return pd.Timestamp(day)
        return None

    # ------------------------------------------------------------------ paso

    def _prepare(self, target_id: str, session: pd.Timestamp, bars: dict[str, pd.DataFrame]) -> TargetState:
        sel = self.target_selection(target_id)
        thr = self.thresholds.get("targets", {}).get(target_id, {}) or {}
        z_star = float(thr.get("z_star") or Z_DEFAULT)
        if sel is None:
            return TargetState(target_id, session, None, False, z_star, False, "sin selección nocturna")
        factors, driver_ok = sel
        need = self.needed_instruments(target_id, factors)
        missing = sorted(need - set(bars)) if self.available is None else sorted(need - (self.available & set(bars)))
        if missing:
            # Sin un factor en MT5 se estima sin él (p. ej., sin tipos); sin el objetivo o el mercado, no se modeliza.
            t = self.universe.targets[target_id]
            core = (set(t.basket) if t.basket else {target_id}) | set(t.market.members)
            if core & set(missing):
                return TargetState(target_id, session, None, driver_ok, z_star, False, f"faltan en MT5: {', '.join(missing)}")
            factors = [f for f in factors if not set(self.universe.factors[f].instruments) & set(missing)]
        target = self.universe.targets[target_id]
        frame = build_frame(bars, target, self.universe, factors)
        train = last_sessions(frame, session, FIT_SESSIONS) if not frame.empty else frame
        note = f"sin {', '.join(missing)}" if missing else ""
        try:
            model = IntradayModel.fit(train, target_id, factors, target.market.type != "none")
        except (ValueError, KeyError) as e:
            return TargetState(target_id, session, None, driver_ok, z_star, False, f"no se pudo estimar: {e}")
        r2_median = thr.get("r2_median")
        r2_ok = r2_median is not None and model.r2 > float(r2_median)
        return TargetState(target_id, session, model, driver_ok, z_star, r2_ok, note)

    def step(self, bars: dict[str, pd.DataFrame], now: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Evaluación de la sesión en curso y alertas de cada objetivo abierto."""
        evals, alerts = {}, []
        for tid in self.universe.targets:
            session = self.in_session(tid, now)
            if session is None:
                continue
            st = self.states.get(tid)
            if st is None or st.session != session:
                st = self._prepare(tid, session, bars)
                self.states[tid] = st
                if st.model is None:
                    self.excluded[tid] = st.note
                    log.warning("%s: no se modeliza hoy (%s)", tid, st.note)
                else:
                    self.excluded.pop(tid, None)
            if st.model is None:
                continue
            target = self.universe.targets[tid]
            frame = build_frame(bars, target, self.universe, st.model.factors)
            cur = frame[(frame["session"] == session) & (frame.index < now.floor("5min"))] if not frame.empty else frame
            ev = st.model.evaluate(cur)
            evals[tid] = ev
            if ev.empty:
                continue
            blocked = block_mask(ev.index, self.events, tid, self.universe)
            a = find_alerts(ev, tid, st.z_star, st.model.regressors, r2_ok=st.r2_ok, driver_ok=st.driver_ok, blocked=blocked)
            alerts.append(a)
        nonempty = [a for a in alerts if not a.empty]
        return evals, pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=ALERT_COLUMNS)
