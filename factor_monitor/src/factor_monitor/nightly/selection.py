"""Selección semanal de regresores (SPEC v3 §6).

Con datos hasta el último día hábil de la semana (normalmente el viernes):
  - un factor se selecciona si su cuota Shapley en la última estimación es >= 15 %
    y su beta tiene el mismo signo en las 4 últimas estimaciones semanales;
  - como máximo 3 factores, los de mayor cuota, más el mercado si el objetivo lo tiene;
  - si ninguno pasa, el objetivo queda como "sin_driver_macro".
La selección es válida de lunes a viernes de la semana siguiente y se guarda por ID lógico.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..common.calendars import trading_days
from ..common.universe import Universe
from .daily_layer import MKT

MIN_SHARE = 0.15
SIGN_WEEKS = 4
MAX_FACTORS = 3
STATUS_OK = "ok"
STATUS_NONE = "sin_driver_macro"
STATUS_NO_DATA = "sin_datos"


def is_week_end(day: date, exchange: str | None = "XETR") -> bool:
    """True si `day` es el último día hábil de su semana en `exchange`."""
    d = pd.Timestamp(day)
    friday = d + pd.Timedelta(days=4 - d.weekday()) if d.weekday() <= 4 else d
    later = trading_days((d + pd.Timedelta(days=1)).date(), friday.date(), exchange) if friday > d else []
    return len(later) == 0


def next_week_bounds(day: date) -> tuple[date, date]:
    d = pd.Timestamp(day)
    monday = d + pd.Timedelta(days=7 - d.weekday())
    return monday.date(), (monday + pd.Timedelta(days=4)).date()


def weekly_snapshots(factors: pd.DataFrame, target: str, as_of: date) -> pd.DataFrame:
    """Última estimación de cada semana ISO hasta `as_of` (incluida)."""
    df = factors[(factors["target"] == target) & (factors["date"] <= pd.Timestamp(as_of))]
    if df.empty:
        return df
    iso = df["date"].dt.isocalendar()
    week = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    last_dates = df.groupby(week)["date"].transform("max")
    return df[df["date"] == last_dates]


def select_target(factors: pd.DataFrame, target: str, as_of: date, has_market: bool) -> dict:
    snaps = weekly_snapshots(factors, target, as_of)
    if snaps.empty:
        return {"status": STATUS_NO_DATA, "factors": [], "market": has_market, "shares": {}, "signs": {}}
    weeks = sorted(snaps["date"].unique())[-SIGN_WEEKS:]
    recent = snaps[snaps["date"].isin(weeks)]
    latest = recent[recent["date"] == weeks[-1]].set_index("factor")
    candidates = latest.drop(MKT, errors="ignore")
    shares, signs, eligible = {}, {}, []
    for f, row in candidates.iterrows():
        hist = recent[recent["factor"] == f].sort_values("date")["std_beta"]
        sign_vals = np.sign(hist.to_numpy())
        stable = len(weeks) == SIGN_WEEKS and len(sign_vals) == SIGN_WEEKS and abs(sign_vals.sum()) == SIGN_WEEKS
        shares[f] = float(row["share"])
        signs[f] = int(sign_vals[-1]) if stable else 0
        if stable and row["share"] >= MIN_SHARE:
            eligible.append(f)
    chosen = sorted(eligible, key=lambda f: shares[f], reverse=True)[:MAX_FACTORS]
    return {
        "status": STATUS_OK if chosen else STATUS_NONE,
        "factors": chosen,
        "market": has_market,
        "shares": shares,
        "signs": signs,
        "estimate_date": pd.Timestamp(weeks[-1]).date().isoformat(),
    }


def run_selection(factors: pd.DataFrame, universe: Universe, as_of: date) -> tuple[dict, pd.DataFrame]:
    """Selección para todos los objetivos. Devuelve el JSON y las filas del historial."""
    valid_from, valid_to = next_week_bounds(as_of)
    out = {"as_of": as_of.isoformat(), "valid_from": valid_from.isoformat(), "valid_to": valid_to.isoformat(), "targets": {}}
    rows = []
    for tid, target in universe.targets.items():
        sel = select_target(factors, tid, as_of, target.market.type != "none")
        out["targets"][tid] = sel
        for f, share in sel["shares"].items():
            rows.append(
                {
                    "as_of": pd.Timestamp(as_of),
                    "valid_from": pd.Timestamp(valid_from),
                    "target": tid,
                    "factor": f,
                    "share": share,
                    "sign": sel["signs"][f],
                    "selected": f in sel["factors"],
                    "status": sel["status"],
                }
            )
    return out, pd.DataFrame(rows)


def backfill_history(factors: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    """Selecciones semanales pasadas, cada una con los datos disponibles en su fecha.

    Hace falta en la primera ejecución completa: el historial de alertas usa la
    selección vigente en cada sesión, no la de hoy.
    """
    dates = sorted(pd.to_datetime(factors["date"]).dt.date.unique())
    rows = [run_selection(factors, universe, d)[1] for d in dates if is_week_end(d)]
    rows = [r for r in rows if not r.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
