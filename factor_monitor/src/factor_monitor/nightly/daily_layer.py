"""Capa diaria (SPEC v3 §6): retornos de sesión, MCO móvil, Shapley y betas.

Para cada objetivo y fecha de sesión:
  - el precio de cierre de todos los instrumentos se toma en la misma marca, la
    última vela de 5 minutos de la sesión del objetivo (tolerancia de 30 min);
  - el retorno diario es el log-retorno entre dos cierres consecutivos de sesiones
    hábiles en la bolsa del objetivo;
  - en una ventana de 60 sesiones se estima MCO con mercado + candidatos
    (ortogonalizados respecto al mercado), Shapley (LMG), betas estandarizadas y t HAC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common.calendars import is_trading_day, trading_days
from ..common.factors import Orthogonalizer, basket_return, build_factors
from ..common.models import ols, shapley_lmg, shapley_shares
from ..common.runlog import RunLog
from ..common.sessions import BAR_MINUTES, session_bounds_utc
from ..common.universe import Target, Universe

log = logging.getLogger(__name__)

STEP = "daily"
WINDOW = 60
MIN_OBS = 50
CLOSE_TOLERANCE_BARS = 6   # 30 minutos
MKT = "MKT"


@dataclass
class DailyResult:
    factors: pd.DataFrame   # date, target, factor, beta, std_beta, t_hac, shapley, share
    r2: pd.DataFrame        # date, target, r2, nobs


def close_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cierres de 5 minutos en una rejilla regular común (columnas = instrumentos)."""
    series = {k: v["close"] for k, v in bars.items() if v is not None and not v.empty}
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).sort_index()
    grid = pd.date_range(panel.index.min(), panel.index.max(), freq=f"{BAR_MINUTES}min")
    return panel.reindex(grid)


def session_closes(panel: pd.DataFrame, target: Target) -> pd.DataFrame:
    """Precio de cada instrumento en la última vela de la sesión del objetivo, por fecha hábil.

    Si un instrumento no tiene vela en esa marca, se usa la última de los 30 minutos
    anteriores; si tampoco, NaN.
    """
    if panel.empty:
        return pd.DataFrame()
    filled = panel.ffill(limit=CLOSE_TOLERANCE_BARS)
    first = panel.index.min().tz_convert("UTC").date()
    last = panel.index.max().tz_convert("UTC").date()
    rows, dates = [], []
    for d in trading_days(first, last, target.exchange):
        _, end = session_bounds_utc(d.date(), target.session)
        ts = end - pd.Timedelta(minutes=BAR_MINUTES)
        if ts in filled.index:
            rows.append(filled.loc[ts])
            dates.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def closed_mask(dates: pd.DatetimeIndex, columns: list[str], universe: Universe) -> pd.DataFrame:
    """True si la bolsa del instrumento está cerrada en la fecha."""
    out = {}
    for c in columns:
        inst = universe.instruments.get(c)
        exch = inst.exchange if inst else None
        out[c] = [not is_trading_day(d, exch) for d in dates] if exch else [False] * len(dates)
    return pd.DataFrame(out, index=dates)


def target_daily_frame(
    panel: pd.DataFrame,
    target: Target,
    universe: Universe,
    candidates: tuple[str, ...],
    basket_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Retornos diarios del objetivo (`y`), del mercado (`MKT`) y de los candidatos."""
    closes = session_closes(panel, target)
    if closes.empty:
        return pd.DataFrame()
    rets = np.log(closes).diff()
    if target.basket:
        comps = [c for c in target.basket if c in rets]
        w = basket_weights if basket_weights is not None else pd.Series(1.0, index=comps)
        y = basket_return(rets[comps], w.reindex(comps).fillna(0.0)) if comps else pd.Series(np.nan, index=rets.index)
    else:
        y = rets[target.id] if target.id in rets else pd.Series(np.nan, index=rets.index)
    out = pd.DataFrame({"y": y})
    if target.market.type == "mean":
        members = [m for m in target.market.members if m in rets]
        block = rets[members].mask(closed_mask(rets.index, members, universe))
        out[MKT] = block.mean(axis=1, skipna=True)
    out = out.join(build_factors(rets, universe, list(candidates)))
    return out


def estimate_window(window: pd.DataFrame, candidates: list[str], has_market: bool) -> dict | None:
    """MCO + Shapley en una ventana. `window` contiene y, MKT (opcional) y candidatos."""
    cols = ["y"] + ([MKT] if has_market else []) + candidates
    w = window[cols].dropna()
    if len(w) < MIN_OBS or not candidates and not has_market:
        return None
    X = w[candidates].copy()
    if has_market:
        X = Orthogonalizer.fit(X, w[MKT]).transform(X, w[MKT]) if candidates else X
        X.insert(0, MKT, w[MKT])
    res = ols(w["y"], X)
    shap = shapley_lmg(w["y"], X)
    shares = shapley_shares(shap, MKT if has_market else None)
    return {"nobs": len(w), "r2": res.r2, "params": res.params, "std": res.std_betas, "t": res.tvalues, "shapley": shap, "shares": shares}


def run_target(
    frame: pd.DataFrame,
    target: Target,
    candidates: tuple[str, ...],
    dates: pd.DatetimeIndex | None = None,
    window: int = WINDOW,
) -> DailyResult:
    """Estimación móvil para las fechas indicadas (por defecto, todas)."""
    has_market = target.market.type != "none" and MKT in frame
    cands = [c for c in candidates if c in frame]
    needed = ["y"] + ([MKT] if has_market else []) + cands
    valid = frame[needed].dropna()
    if dates is None:
        dates = valid.index
    f_rows, r_rows = [], []
    for d in dates:
        hist = valid.loc[:d].tail(window)
        if len(hist) == 0 or hist.index[-1] != d:
            continue
        est = estimate_window(hist, cands, has_market)
        if est is None:
            continue
        r_rows.append({"date": d, "target": target.id, "r2": est["r2"], "nobs": est["nobs"]})
        for f in est["shapley"].index:
            f_rows.append(
                {
                    "date": d,
                    "target": target.id,
                    "factor": f,
                    "beta": float(est["params"][f]),
                    "std_beta": float(est["std"][f]),
                    "t_hac": float(est["t"][f]),
                    "shapley": float(est["shapley"][f]),
                    "share": float(est["shares"][f]) if f in est["shares"] else np.nan,
                }
            )
    return DailyResult(pd.DataFrame(f_rows), pd.DataFrame(r_rows))


def run_daily_layer(
    bars: dict[str, pd.DataFrame],
    universe: Universe,
    runlog: RunLog,
    since: pd.Timestamp | None = None,
    basket_weights: dict[str, pd.Series] | None = None,
) -> DailyResult:
    """Capa diaria para todos los objetivos. Con `since`, solo fechas >= since."""
    panel = close_panel(bars)
    factors, r2 = [], []
    for tid, target in universe.targets.items():
        cands, excluded = universe.effective_candidates(tid)
        for e in excluded:
            runlog.warn(STEP, f"{tid}: se excluye {e.factor} ({e.reason})", target=tid, factor=e.factor)
        missing = sorted(i for i in universe.required_instruments(tid) if i not in panel)
        usable = tuple(
            c for c in cands if all(i in panel for i in universe.factors[c].instruments)
        )
        if missing:
            runlog.warn(STEP, f"{tid}: sin datos de {', '.join(missing)}", target=tid, missing=missing)
        frame = target_daily_frame(panel, target, universe, usable, (basket_weights or {}).get(tid))
        if frame.empty or frame["y"].dropna().empty:
            runlog.warn(STEP, f"{tid}: sin retornos del objetivo; se omite", target=tid)
            continue
        dates = frame.index if since is None else frame.index[frame.index >= since]
        res = run_target(frame, target, usable, dates)
        factors.append(res.factors)
        r2.append(res.r2)
    cat = lambda xs: pd.concat([x for x in xs if not x.empty], ignore_index=True) if any(not x.empty for x in xs) else pd.DataFrame()
    return DailyResult(cat(factors), cat(r2))


def merge_history(old: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Une el histórico con las filas nuevas; las nuevas sustituyen a las repetidas."""
    if old is None or old.empty:
        return new.sort_values(keys).reset_index(drop=True)
    if new.empty:
        return old
    out = pd.concat([old, new], ignore_index=True)
    return out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
