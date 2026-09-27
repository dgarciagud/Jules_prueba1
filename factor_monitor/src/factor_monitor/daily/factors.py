"""Paso 3: qué factor macro mueve cada sector (Shapley del R²).

Para cada sector de cada índice, en una ventana móvil de WINDOW sesiones:

    r_sector = a + b·r_índice + Σ_f g_f·f⊥ + e,

donde f⊥ es cada factor ortogonalizado respecto al índice en la misma ventana. Así el mercado
se queda con todo lo que comparte con el índice y los factores explican lo que el sector hace
aparte. El R² se reparte con Shapley (LMG) y se informa la cuota macro de cada factor,
Shapley_f / (R² − Shapley_índice), con el signo de su beta estandarizada.

Para el índice mismo se regresa sobre los factores sin ortogonalizar (qué macro mueve el índice).

Sincronía: el CAC 40 cierra a las 17:30 de París, antes que los factores de EE.UU. En la región
"eu" se usan rentabilidades de 2 días solapadas, que reducen a la mitad ese desfase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.models import ols, shapley_lmg, shapley_shares
from .attribution import sector_levels
from .data import RefWeights
from .universe import DailyUniverse

WINDOW = 126                 # sesiones (≈ 6 meses)
SNAPSHOT_YEARS = 5           # historial de estimaciones a fin de mes
RETURN_DAYS = {"us": 1, "eu": 2}
MKT = "MKT"
INDEX_TARGET = "_index"


def factor_changes(close: pd.DataFrame, universe: DailyUniverse, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Cambio diario de cada factor en el calendario `dates`: log-rentabilidad (precio) o diferencia (nivel)."""
    out = {}
    for fid, f in universe.factors.items():
        if fid not in close:
            continue
        lv = close[fid].ffill().reindex(dates)
        out[fid] = np.log(lv).diff() if f.kind == "price" else lv.diff()
    return pd.DataFrame(out, index=dates)


def orthogonalize(F: pd.DataFrame, m: pd.Series) -> pd.DataFrame:
    """Residuo de cada factor sobre el mercado (con constante) en la muestra dada."""
    mc = m - m.mean()
    denom = (mc**2).sum()
    if denom <= 0:
        return F - F.mean()
    Fc = F - F.mean()
    beta = Fc.mul(mc, axis=0).sum() / denom
    return Fc - np.outer(mc, beta)


def decompose(y: pd.Series, F: pd.DataFrame, mkt: pd.Series | None) -> dict | None:
    """Shapley, cuota macro y beta estandarizada en una ventana (datos ya recortados)."""
    df = pd.concat([y.rename("y"), F, mkt.rename(MKT) if mkt is not None else None], axis=1).dropna()
    factors = [c for c in F.columns if df[c].std() > 0]
    if len(df) < 0.8 * WINDOW or not factors:
        return None
    if mkt is not None:
        X = pd.concat([df[[MKT]], orthogonalize(df[factors], df[MKT])], axis=1)
    else:
        X = df[factors]
    sh = shapley_lmg(df["y"], X)
    res = ols(df["y"], X)
    share = shapley_shares(sh, MKT if mkt is not None else None)
    return {"shapley": sh, "share": share, "std_beta": res.std_betas, "r2": res.r2, "n": res.nobs}


def snapshot_dates(dates: pd.DatetimeIndex, years: int = SNAPSHOT_YEARS) -> list[pd.Timestamp]:
    """Última sesión de cada mes de los últimos `years` años, más la última fecha."""
    recent = dates[dates >= dates[-1] - pd.DateOffset(years=years)]
    s = pd.Series(recent, index=recent)
    ends = s.groupby([recent.year, recent.month]).max().tolist()
    if ends[-1] != dates[-1]:
        ends.append(dates[-1])
    return ends


def sector_factor_shapley(ix_id: str, region: str, cal: pd.DataFrame, close: pd.DataFrame, ref: RefWeights,
                          universe: DailyUniverse, snapshots: bool = True) -> pd.DataFrame:  # fmt: skip
    """Filas (index, target, date, factor, shapley, share, std_beta, r2, n) por sector y por el índice."""
    h = RETURN_DAYS.get(region, 1)
    dates = cal.index
    levels = sector_levels(cal, ref)
    Y = np.log(levels).diff()
    Y[INDEX_TARGET] = np.log(cal[ix_id]).diff()
    F = factor_changes(close, universe, dates)
    if h > 1:
        Y, F = Y.rolling(h).sum(), F.rolling(h).sum()
    mkt = Y[INDEX_TARGET]
    names = {**universe.sector_names, INDEX_TARGET: universe.indices[ix_id].name}
    rows = []
    for d in (snapshot_dates(dates) if snapshots else [dates[-1]]):
        end = dates.get_loc(d) + 1
        sl = slice(max(0, end - WINDOW), end)
        for target in Y.columns:
            y = Y[target].iloc[sl]
            if y.notna().sum() < 0.8 * WINDOW:
                continue
            is_index = target == INDEX_TARGET
            dec = decompose(y, F.iloc[sl], None if is_index else mkt.iloc[sl])
            if dec is None:
                continue
            for factor, val in dec["shapley"].items():
                rows.append({"index": ix_id, "target": target, "target_name": names.get(target, target), "date": d,
                             "factor": factor, "shapley": float(val),
                             "share": float(dec["share"].get(factor, np.nan)),
                             "std_beta": float(dec["std_beta"].get(factor, np.nan)),
                             "r2": dec["r2"], "n": dec["n"], "return_days": h})  # fmt: skip
    return pd.DataFrame(rows)
