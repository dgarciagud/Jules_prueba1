"""Calendario de eventos macro (SPEC v3 §3.7).

`config/events.csv` es la fuente de verdad, curada y versionada. Nunca se inventan
fechas: si falta cobertura, `validate` lo avisa y ese tramo queda sin control.
La EIA semanal se genera por regla (miércoles 10:30 hora de Nueva York).
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .universe import Universe

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "events.csv"
COLUMNS = ["datetime_utc", "event_type", "region", "block_before_min", "block_after_min", "source"]

# Bloqueo por defecto (antes, después) en minutos; None = todo el día.
DEFAULT_BLOCKS: dict[str, tuple[int, int] | None] = {
    "FOMC": (15, 75),
    "ECB": (15, 75),
    "US_CPI": (15, 30),
    "US_NFP": (15, 30),
    "EZ_CPI_FLASH": (10, 20),
    "PMI": (10, 20),
    "IFO": (10, 20),
    "ZEW": (10, 20),
    "EIA": (10, 20),
    "OPEC": None,
}
# Eventos por año esperados (mín., máx.) para la validación de cobertura.
EXPECTED_PER_YEAR = {"FOMC": (8, 9), "ECB": (8, 9), "US_CPI": (12, 12), "US_NFP": (12, 12)}
# Eventos que solo afectan a objetivos con exposición al petróleo.
OIL_EVENTS = {"OPEC", "EIA"}


def load_events(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, comment="#")
    missing = set(COLUMNS[:3]) - set(df.columns)
    if missing:
        raise ValueError(f"events.csv sin columnas {sorted(missing)}")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    for c in ("block_before_min", "block_after_min", "source"):
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS]


def eia_events(start: date, end: date, exceptions: dict[date, datetime | None] | None = None) -> pd.DataFrame:
    """Inventario semanal de crudo de la EIA: miércoles 10:30 NY (excepciones a mano)."""
    ny = ZoneInfo("America/New_York")
    rows = []
    for d in pd.date_range(start, end, freq="W-WED"):
        day = d.date()
        when = datetime.combine(day, time(10, 30), tzinfo=ny)
        if exceptions and day in exceptions:
            if exceptions[day] is None:
                continue
            when = exceptions[day]
        rows.append(
            {"datetime_utc": pd.Timestamp(when).tz_convert("UTC"), "event_type": "EIA", "region": "US",
             "block_before_min": np.nan, "block_after_min": np.nan, "source": "regla: miércoles 10:30 NY"}
        )  # fmt: skip
    return pd.DataFrame(rows, columns=COLUMNS)


def validate(events: pd.DataFrame, start: date, end: date) -> list[str]:
    """Avisos de calidad y cobertura del calendario."""
    warnings = []
    if events.empty:
        return [f"events.csv vacío: no hay control de eventos entre {start} y {end}"]
    dup = events.duplicated(["datetime_utc", "event_type"])
    if dup.any():
        warnings.append(f"{int(dup.sum())} eventos duplicados")
    no_src = events["source"].isna() | (events["source"].astype(str).str.strip() == "")
    if no_src.any():
        warnings.append(f"{int(no_src.sum())} eventos sin fuente")
    unknown = set(events["event_type"]) - set(DEFAULT_BLOCKS)
    if unknown:
        warnings.append(f"tipos desconocidos: {sorted(unknown)}")
    for etype, (lo, hi) in EXPECTED_PER_YEAR.items():
        sub = events[events["event_type"] == etype]
        for year in range(start.year, end.year + 1):
            n = int((sub["datetime_utc"].dt.year == year).sum())
            full_year = date(year, 1, 1) >= start and date(year, 12, 31) <= end
            if full_year and not lo <= n <= hi:
                warnings.append(f"{etype} {year}: {n} eventos (esperados {lo}-{hi})")
            if n == 0 and not full_year:
                warnings.append(f"{etype} {year}: sin eventos en el calendario")
    return warnings


def affects(event_type: str, target_id: str, universe: Universe) -> bool:
    if event_type not in OIL_EVENTS:
        return True
    target = universe.targets[target_id]
    return "BRENT" in universe.effective_candidates(target_id)[0] or target_id == "ENERGY" or target_id == "BRENT"


def block_mask(index: pd.DatetimeIndex, events: pd.DataFrame, target_id: str, universe: Universe) -> pd.Series:
    """True en las velas bloqueadas por algún evento que afecte al objetivo.

    Una vela [t, t+5 min) está bloqueada si se solapa con [evento − antes, evento + después].
    Los eventos de día completo bloquean su fecha UTC.
    """
    mask = np.zeros(len(index), dtype=bool)
    if events.empty or len(index) == 0:
        return pd.Series(mask, index=index)
    bar_end = index + pd.Timedelta(minutes=5)
    for ev in events.itertuples(index=False):
        if not affects(ev.event_type, target_id, universe):
            continue
        default = DEFAULT_BLOCKS.get(ev.event_type, (15, 30))
        if default is None:
            mask |= np.asarray(index.normalize() == ev.datetime_utc.normalize())
            continue
        before = ev.block_before_min if pd.notna(ev.block_before_min) else default[0]
        after = ev.block_after_min if pd.notna(ev.block_after_min) else default[1]
        lo = ev.datetime_utc - pd.Timedelta(minutes=float(before))
        hi = ev.datetime_utc + pd.Timedelta(minutes=float(after))
        mask |= np.asarray((bar_end > lo) & (index < hi))
    return pd.Series(mask, index=index)
