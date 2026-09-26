"""Agregación de minutos bid/ask a velas de 5 minutos y utilidades de fusión."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sessions import BAR_MINUTES

BAR_COLUMNS = ["open", "high", "low", "close", "spread", "n"]
M1_COLUMNS = [f"{side}_{f}" for side in ("bid", "ask") for f in ("open", "high", "low", "close")]


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=idx)


def m1_to_bars(m1: pd.DataFrame, bar_minutes: int = BAR_MINUTES, bid_only: bool = False) -> pd.DataFrame:
    """Minutos bid/ask (índice UTC = apertura del minuto) → velas de mid.

    Se descartan los minutos con ask <= bid. `spread` es el spread medio de cierre
    de los minutos de la vela y `n` el número de minutos con dato.
    Con `bid_only` (solo se descargó el bid), el precio es el bid y el spread NaN.
    Marca de la vela = apertura, en UTC.
    """
    if m1.empty:
        return empty_bars()
    missing = set(M1_COLUMNS) - set(m1.columns)
    if missing:
        raise ValueError(f"Faltan columnas M1: {sorted(missing)}")
    if bid_only:
        mid = pd.DataFrame({f: m1[f"bid_{f}"] for f in ("open", "high", "low", "close")}, index=m1.index)
        mid["spread"] = np.nan
    else:
        m1 = m1[m1["ask_close"] > m1["bid_close"]]
        mid = pd.DataFrame(
            {f: (m1[f"bid_{f}"] + m1[f"ask_{f}"]) / 2 for f in ("open", "high", "low", "close")},
            index=m1.index,
        )
        mid["spread"] = m1["ask_close"] - m1["bid_close"]
    rule = f"{bar_minutes}min"
    agg = mid.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}
    )
    agg["n"] = mid["close"].resample(rule, label="left", closed="left").count().astype(float)
    agg = agg[agg["n"] > 0]
    agg.index.name = "ts"
    return agg[BAR_COLUMNS]


def merge_bars(old: pd.DataFrame | None, new: pd.DataFrame, replace_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """Une velas. Las antiguas con marca >= `replace_from` se sustituyen por las nuevas."""
    if old is None or old.empty:
        out = new
    else:
        if replace_from is not None:
            old = old[old.index < replace_from]
        out = pd.concat([old, new])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "ts"
    return out


def split_by_year(bars: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if bars.empty:
        return {}
    return {int(y): g for y, g in bars.groupby(bars.index.year)}


def jump_count(close: pd.Series, n_mad: float = 10.0) -> int:
    """Número de retornos con desviación mayor que `n_mad` MAD respecto a la mediana."""
    r = np.log(close).diff().dropna()
    if r.empty:
        return 0
    med = r.median()
    mad = (r - med).abs().median()
    if mad == 0:
        return 0
    return int(((r - med).abs() > n_mad * mad).sum())


def quality_report(bars: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Calidad por mes: velas, días, velas por día, spread en pb y saltos."""
    if bars.empty:
        return pd.DataFrame()
    rows = []
    for period, g in bars.groupby(bars.index.strftime("%Y-%m")):
        spread_bps = (g["spread"] / g["close"]) * 1e4
        days = g.index.normalize().nunique()
        rows.append(
            {
                "instrument": instrument,
                "month": period,
                "bars": len(g),
                "days": days,
                "bars_per_day_median": float(g.groupby(g.index.normalize()).size().median()),
                "minute_coverage": float(g["n"].sum() / (len(g) * BAR_MINUTES)),
                "spread_bps_median": float(spread_bps.median()),
                "spread_bps_p99": float(spread_bps.quantile(0.99)),
                "jumps_10mad": jump_count(g["close"]),
            }
        )
    return pd.DataFrame(rows)
