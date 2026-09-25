from datetime import date

import numpy as np
import pandas as pd
import pytest

from factor_monitor.common.calendars import holiday_mask, is_trading_day
from factor_monitor.common.sessions import bucket_index, in_session, session_bars, session_bounds_utc


def utc(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize(
    "day, session, start, end",
    [
        # Invierno en ambos lados.
        (date(2024, 1, 15), "eu", "2024-01-15 08:00", "2024-01-15 16:30"),
        (date(2024, 1, 15), "us", "2024-01-15 14:30", "2024-01-15 21:00"),
        # EE.UU. ya en verano (10 mar 2024) y la UE todavía no (31 mar 2024).
        (date(2024, 3, 11), "eu", "2024-03-11 08:00", "2024-03-11 16:30"),
        (date(2024, 3, 11), "us", "2024-03-11 13:30", "2024-03-11 20:00"),
        # Ambos en verano.
        (date(2024, 4, 2), "eu", "2024-04-02 07:00", "2024-04-02 15:30"),
        (date(2024, 4, 2), "us", "2024-04-02 13:30", "2024-04-02 20:00"),
        # La UE vuelve a invierno (27 oct 2024) y EE.UU. todavía no (3 nov 2024).
        (date(2024, 10, 28), "eu", "2024-10-28 08:00", "2024-10-28 16:30"),
        (date(2024, 10, 28), "us", "2024-10-28 13:30", "2024-10-28 20:00"),
        # Días de cambio.
        (date(2024, 3, 31), "eu", "2024-03-31 07:00", "2024-03-31 15:30"),
        (date(2024, 11, 4), "us", "2024-11-04 14:30", "2024-11-04 21:00"),
    ],
)
def test_session_bounds_dst(day, session, start, end):
    assert session_bounds_utc(day, session) == (utc(start), utc(end))


def test_session_bars_count_and_convention():
    eu = session_bars(date(2024, 1, 15), "eu")
    us = session_bars(date(2024, 1, 15), "us")
    assert len(eu) == 102 and len(us) == 78
    assert eu[0] == utc("2024-01-15 08:00")
    assert eu[-1] == utc("2024-01-15 16:25")  # la última vela empieza 5 min antes del cierre


def test_in_session_and_buckets():
    idx = pd.date_range(utc("2024-01-15 07:50"), utc("2024-01-15 16:40"), freq="5min")
    mask = in_session(idx, "eu")
    assert not mask[idx.get_loc(utc("2024-01-15 07:55"))]
    assert mask[idx.get_loc(utc("2024-01-15 08:00"))]
    assert mask[idx.get_loc(utc("2024-01-15 16:25"))]
    assert not mask[idx.get_loc(utc("2024-01-15 16:30"))]
    b = bucket_index(idx[mask], "eu")
    assert b[0] == 0 and b[-1] == 16
    assert np.bincount(b).tolist() == [6] * 17
    sat = pd.date_range(utc("2024-01-13 09:00"), periods=3, freq="5min")
    assert not in_session(sat, "eu").any()


def test_holidays():
    assert not is_trading_day(date(2024, 3, 29), "XETR")  # Viernes Santo
    assert not is_trading_day(date(2024, 5, 1), "XPAR")
    assert is_trading_day(date(2024, 1, 16), "XNYS")
    assert not is_trading_day(date(2024, 1, 13), None)  # sábado
    assert is_trading_day(date(2024, 3, 29), None)


def test_us_holiday_and_mask():
    assert not is_trading_day(date(2024, 1, 15), "XNYS")  # Martin Luther King
    idx = pd.DatetimeIndex([utc("2024-01-15 15:00"), utc("2024-01-16 15:00")])
    assert holiday_mask(idx, "XNYS", "us").tolist() == [True, False]
