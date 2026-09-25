"""Festivos por bolsa (exchange_calendars)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .sessions import session_date


@lru_cache(maxsize=None)
def _calendar(code: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(code, start="2010-01-01")


def is_trading_day(day: date | pd.Timestamp, exchange: str | None) -> bool:
    """Día hábil en `exchange`. Sin bolsa asociada, solo se excluyen los fines de semana."""
    d = pd.Timestamp(day)
    if d.tz is not None:
        d = d.tz_localize(None)
    d = d.normalize()
    if d.weekday() >= 5:
        return False
    if exchange is None:
        return True
    cal = _calendar(exchange)
    if d < cal.first_session or d > cal.last_session:
        raise ValueError(f"{d.date()} fuera del rango del calendario {exchange}")
    return bool(cal.is_session(d))


def trading_days(start: date, end: date, exchange: str | None) -> pd.DatetimeIndex:
    """Días hábiles entre `start` y `end`, ambos incluidos."""
    days = pd.bdate_range(start, end)
    if exchange is None:
        return days
    return pd.DatetimeIndex(_calendar(exchange).sessions_in_range(days[0], days[-1])) if len(days) else days


def holiday_mask(index: pd.DatetimeIndex, exchange: str | None, session: str) -> np.ndarray:
    """True en las marcas cuya fecha local de sesión no es hábil en `exchange`."""
    dates = session_date(index, session)
    unique = dates.unique()
    closed = {d: not is_trading_day(d, exchange) for d in unique}
    return np.array([closed[d] for d in dates], dtype=bool)
