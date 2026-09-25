"""Sesiones de negociación en hora local, convertidas a UTC con horario de verano.

Convención de velas en todo el paquete: la marca de una vela es su hora de
apertura en UTC y la vela cubre [marca, marca + BAR_MINUTES).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BAR_MINUTES = 5


@dataclass(frozen=True)
class Session:
    name: str
    tz: str
    open: time
    close: time

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    @property
    def length_minutes(self) -> int:
        return (self.close.hour * 60 + self.close.minute) - (self.open.hour * 60 + self.open.minute)


SESSIONS: dict[str, Session] = {
    "eu": Session("eu", "Europe/Paris", time(9, 0), time(17, 30)),
    "us": Session("us", "America/New_York", time(9, 30), time(16, 0)),
}


def get_session(session: str | Session) -> Session:
    if isinstance(session, Session):
        return session
    try:
        return SESSIONS[session]
    except KeyError:
        raise ValueError(f"Sesión desconocida: {session!r}") from None


def session_bounds_utc(day: date, session: str | Session) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Apertura y cierre de la sesión de `day` (fecha local) en UTC."""
    s = get_session(session)
    start = pd.Timestamp(datetime.combine(day, s.open, tzinfo=s.zone)).tz_convert("UTC")
    end = pd.Timestamp(datetime.combine(day, s.close, tzinfo=s.zone)).tz_convert("UTC")
    return start, end


def session_bars(day: date, session: str | Session, bar_minutes: int = BAR_MINUTES) -> pd.DatetimeIndex:
    """Marcas (apertura, UTC) de las velas de la sesión de `day`."""
    start, end = session_bounds_utc(day, session)
    return pd.date_range(start, end, freq=f"{bar_minutes}min", inclusive="left")


def _local(index: pd.DatetimeIndex, s: Session) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("Se esperan marcas con zona horaria (UTC)")
    return index.tz_convert(s.tz)


def minutes_since_open(index: pd.DatetimeIndex, session: str | Session) -> np.ndarray:
    """Minutos desde la apertura local de la sesión (negativo antes de abrir)."""
    s = get_session(session)
    local = _local(index, s)
    minutes = local.hour * 60 + local.minute
    return np.asarray(minutes - (s.open.hour * 60 + s.open.minute))


def in_session(index: pd.DatetimeIndex, session: str | Session, bar_minutes: int = BAR_MINUTES) -> np.ndarray:
    """True para las velas que empiezan y terminan dentro de la sesión, de lunes a viernes."""
    s = get_session(session)
    local = _local(index, s)
    m = minutes_since_open(index, s)
    return np.asarray((m >= 0) & (m + bar_minutes <= s.length_minutes) & (local.weekday < 5))


def session_date(index: pd.DatetimeIndex, session: str | Session) -> pd.DatetimeIndex:
    """Fecha local de la sesión (sin zona) de cada marca."""
    s = get_session(session)
    return _local(index, s).tz_localize(None).normalize()


def bucket_index(index: pd.DatetimeIndex, session: str | Session, bucket_minutes: int = 30) -> np.ndarray:
    """Franja de `bucket_minutes` dentro de la sesión (0 = primera franja)."""
    return minutes_since_open(index, session) // bucket_minutes
