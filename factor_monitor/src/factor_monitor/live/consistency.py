"""Consistencia entre fuentes (SPEC v3 §3.5).

Correlación de retornos de 5 minutos entre Dukascopy (`recent_bars.parquet` de la
rama `results`) y MT5 en las sesiones solapadas, por instrumento:
  > 0,95 correcto · 0,90–0,95 aviso · < 0,90 no consistente (historial "no comparable").
Además se calcula la correlación con desfases de −2 a +2 velas: si el máximo no
está en 0, probablemente la hora del servidor MT5 está mal aplicada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OK, WARN = 0.95, 0.90
LAGS = range(-2, 3)
MIN_OBS = 200


def status_for(corr: float) -> str:
    if not np.isfinite(corr):
        return "sin_datos"
    if corr > OK:
        return "ok"
    if corr >= WARN:
        return "aviso"
    return "no_consistente"


def compare(dukascopy: pd.DataFrame, mt5: pd.DataFrame, instrument: str) -> dict:
    """`dukascopy` y `mt5`: velas de 5 min (índice UTC, columna close) del mismo instrumento."""
    a = np.log(dukascopy["close"]).diff()
    b = np.log(mt5["close"]).diff()
    common = a.index.intersection(b.index)
    out = {"instrument": instrument, "n": int(len(common))}
    if len(common) < MIN_OBS:
        return {**out, "corr": np.nan, "best_lag": None, "status": "sin_datos"}
    corrs = {}
    for lag in LAGS:
        shifted = b.shift(lag)
        df = pd.concat([a, shifted], axis=1, keys=["d", "m"]).loc[common].dropna()
        corrs[lag] = float(df["d"].corr(df["m"])) if len(df) >= MIN_OBS else np.nan
    best = max(corrs, key=lambda k: corrs[k] if np.isfinite(corrs[k]) else -np.inf)
    corr0 = corrs[0]
    status = status_for(corr0)
    return {
        **out,
        "corr": corr0,
        "best_lag": best,
        "status": status,
        "clock_warning": best != 0 and np.isfinite(corrs[best]) and corrs[best] > corr0 + 0.05,
        **{f"corr_lag_{k}": v for k, v in corrs.items()},
    }


def run_consistency(recent: pd.DataFrame, mt5_bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """`recent`: formato largo (instrument, ts, close…) de recent_bars.parquet."""
    rows = []
    for iid, grp in recent.groupby("instrument"):
        if iid not in mt5_bars or mt5_bars[iid].empty:
            rows.append({"instrument": iid, "n": 0, "corr": np.nan, "best_lag": None, "status": "sin_mt5"})
            continue
        duk = grp.set_index(pd.DatetimeIndex(grp["ts"], name="ts")).sort_index()
        rows.append(compare(duk, mt5_bars[iid], iid))
    return pd.DataFrame(rows)
