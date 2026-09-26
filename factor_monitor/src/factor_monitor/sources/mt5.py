"""Fuente MT5 (Darwinex), solo en el PC local (SPEC v3 §3.2).

El paquete `MetaTrader5` solo existe en Windows; aquí se recibe como módulo
inyectable para poder probarlo con un mock en Linux/GitHub Actions.

Hora del servidor: MT5 devuelve las marcas en la hora del broker, no en UTC.
El desfase se detecta con un símbolo de referencia con tick reciente:
    desfase = round((hora_tick − utc_ahora) / 1 h), válido si el tick tiene < 60 s.
Sin tick reciente (fin de semana) se deduce de la hora de cierre de la última vela de
EURUSD, que corresponde al cierre del forex del viernes a las 17:00 de Nueva York; si
tampoco es posible, se usa el último desfase guardado.
Convención de velas: MT5 marca la apertura, igual que el resto del paquete.
"""

from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..common.bars import BAR_COLUMNS
from ..common.universe import Universe

log = logging.getLogger(__name__)

MAX_TICK_AGE_S = 60
MIN_OFFSET_H, MAX_OFFSET_H = -12, 14
DEFAULT_REFERENCE = "EURUSD"


class MT5Error(RuntimeError):
    pass


def import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - solo en Windows
        raise MT5Error("El paquete MetaTrader5 no está instalado (pip install -e .[live], solo Windows)") from e
    return mt5


@dataclass
class OffsetResult:
    hours: int
    source: str          # "detected" | "fx_close" | "stored" | "default"
    tick_age_s: float | None = None


class MT5Source:
    def __init__(
        self,
        mt5: Any | None = None,
        state_path: Path | str = "state/local_state.json",
        reference_symbol: str = DEFAULT_REFERENCE,
        now: Callable[[], float] = _time.time,
    ):
        self.mt5 = mt5 if mt5 is not None else import_mt5()
        self.state_path = Path(state_path)
        self.reference_symbol = reference_symbol
        self.now = now
        self.offset: OffsetResult | None = None

    # ------------------------------------------------------------------ conexión

    def connect(self) -> None:
        if not self.mt5.initialize():
            raise MT5Error(f"No se pudo conectar con el terminal MT5: {self.mt5.last_error()}")

    def shutdown(self) -> None:
        self.mt5.shutdown()

    # ------------------------------------------------------------------ inventario

    def inventory(self, universe: Universe) -> pd.DataFrame:
        """Cruza `symbols_get()` con universe.yaml: qué instrumentos tienen símbolo en Darwinex."""
        available = {s.name for s in (self.mt5.symbols_get() or [])}
        rows = []
        for iid, inst in universe.instruments.items():
            sym = inst.mt5_symbol
            rows.append(
                {
                    "instrument": iid,
                    "kind": inst.kind,
                    "mt5_symbol": sym,
                    "available": bool(sym) and sym in available,
                    "reason": "" if sym and sym in available else ("sin mt5_symbol en universe.yaml" if not sym else "símbolo no encontrado en el terminal"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ hora del servidor

    def _stored_offset(self) -> int | None:
        if not self.state_path.exists():
            return None
        try:
            return int(json.loads(self.state_path.read_text(encoding="utf-8"))["server_offset_hours"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

    def _store_offset(self, hours: int) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        data.update({"server_offset_hours": hours, "detected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def detect_offset(self) -> OffsetResult:
        tick = self.mt5.symbol_info_tick(self.reference_symbol)
        now = self.now()
        stored = self._stored_offset()
        if tick is not None and getattr(tick, "time", 0):
            hours = round((tick.time - now) / 3600)
            age = now - (tick.time - hours * 3600)
            # Un tick antiguo da un "desfase" absurdo: solo valen husos reales y ticks recientes.
            if MIN_OFFSET_H <= hours <= MAX_OFFSET_H and 0 <= age < MAX_TICK_AGE_S:
                if stored is not None and hours != stored:
                    log.warning("La hora del servidor MT5 cambia de UTC%+d a UTC%+d (¿cambio de horario?)", stored, hours)
                self._store_offset(hours)
                self.offset = OffsetResult(hours, "detected", age)
                return self.offset
        inferred = self._offset_from_fx_close(now)
        if inferred is not None:
            self._store_offset(inferred)
            self.offset = OffsetResult(inferred, "fx_close")
            return self.offset
        if stored is not None:
            self.offset = OffsetResult(stored, "stored")
        else:
            log.warning("Sin tick reciente ni desfase guardado: se asume UTC+0 hasta poder detectarlo")
            self.offset = OffsetResult(0, "default")
        return self.offset

    def _offset_from_fx_close(self, now: float) -> int | None:
        """Con el mercado cerrado (fin de semana): el forex cierra el viernes a las 17:00 de
        Nueva York, así que la hora de cierre de la última vela de EURUSD da el desfase."""
        get = getattr(self.mt5, "copy_rates_from_pos", None)
        if get is None:
            return None
        rates = get(self.reference_symbol, self.mt5.TIMEFRAME_M5, 0, 1)
        if rates is None or len(rates) == 0:
            return None
        last_close_srv = int(rates[-1]["time"]) + 300
        now_ny = pd.Timestamp(now, unit="s", tz="UTC").tz_convert("America/New_York")
        friday = (now_ny - pd.Timedelta(days=(now_ny.weekday() - 4) % 7)).normalize() + pd.Timedelta(hours=17)
        if friday > now_ny:
            friday -= pd.Timedelta(days=7)
        if now_ny - friday > pd.Timedelta(days=3):
            return None  # no es fin de semana: el último cierre no es el del viernes
        hours = round((last_close_srv - friday.tz_convert("UTC").timestamp()) / 3600)
        return hours if MIN_OFFSET_H <= hours <= MAX_OFFSET_H else None

    def to_utc(self, server_seconds) -> pd.DatetimeIndex:
        if self.offset is None:
            self.detect_offset()
        return pd.to_datetime(pd.Series(server_seconds).astype("int64") - self.offset.hours * 3600, unit="s", utc=True).pipe(pd.DatetimeIndex)

    # ------------------------------------------------------------------ velas

    def bars(self, symbol: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
        """Velas M5 cerradas entre dos marcas UTC, con la convención del paquete (apertura, UTC)."""
        if self.offset is None:
            self.detect_offset()
        shift = pd.Timedelta(hours=self.offset.hours)
        # copy_rates_range espera fechas en la hora del servidor (como si fueran UTC).
        start_srv = (start_utc + shift).to_pydatetime()
        end_srv = (end_utc + shift).to_pydatetime()
        rates = self.mt5.copy_rates_range(symbol, self.mt5.TIMEFRAME_M5, start_srv, end_srv)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
        df = pd.DataFrame(rates)
        idx = self.to_utc(df["time"])
        point = getattr(self.mt5.symbol_info(symbol), "point", 0.0) or 0.0
        out = pd.DataFrame(
            {
                "open": df["open"].to_numpy(float),
                "high": df["high"].to_numpy(float),
                "low": df["low"].to_numpy(float),
                "close": df["close"].to_numpy(float),
                "spread": df["spread"].to_numpy(float) * point if "spread" in df else float("nan"),
                "n": df.get("tick_volume", pd.Series(0, index=df.index)).to_numpy(float),
            },
            index=pd.DatetimeIndex(idx, name="ts"),
        )
        return out[out.index < end_utc]

    def closed_bars(self, universe: Universe, instruments: list[str], sessions_back: int = 15) -> dict[str, pd.DataFrame]:
        """Velas cerradas de los instrumentos lógicos (por mt5_symbol) de las últimas sesiones."""
        now = pd.Timestamp(self.now(), unit="s", tz="UTC")
        last_closed = now.floor("5min")  # la vela en curso nunca entra
        start = now.normalize() - pd.offsets.BDay(sessions_back + 3)  # margen para festivos
        out = {}
        for iid in instruments:
            sym = universe.instruments[iid].mt5_symbol
            if not sym:
                continue
            df = self.bars(sym, start, last_closed)
            if len(df) and df.index.normalize().nunique() < min(sessions_back, 10):
                log.warning("%s: MT5 devuelve solo %d sesiones; amplía 'máx. barras en el gráfico'", sym, df.index.normalize().nunique())
            out[iid] = df
        return out
