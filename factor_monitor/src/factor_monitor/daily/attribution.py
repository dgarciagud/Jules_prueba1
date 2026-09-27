"""Paso 2: atribución sectorial y fuerza relativa.

Atribución. La aportación diaria de un miembro (ETF sectorial o acción) es
c_i,t = w_i,t-1 · r_i,t, con los pesos al cierre anterior. En un periodo de varias sesiones
las aportaciones se encadenan sobre el valor del índice al inicio,
    C_i = Σ_t c_i,t · G_t-1,   G_t-1 = Π_{u<t} (1 + r_índice,u),
de modo que Σ_i C_i + residuo = rentabilidad del índice en el periodo, exactamente.
El residuo es lo que los sectores no reproducen (casi cero en el CAC 40; en el S&P 500,
los ETF limitan el peso de sus mayores valores).

Fuerza relativa. rs_h = (1 + R_sector,h) / (1 + R_índice,h) − 1 en cada horizonte, y su
cambio respecto a hace un mes (si el sector gana o pierde impulso frente al índice).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RefWeights, drifted_weights

# Horizontes en sesiones (YTD se calcula aparte)
PERIODS = {"1D": 1, "1S": 5, "1M": 21, "3M": 63, "YTD": None, "1A": 252}
RS_HORIZONS = {"1S": 5, "1M": 21, "3M": 63, "6M": 126, "1A": 252}
MOMENTUM_LAG = 21


def member_contributions(cal: pd.DataFrame, index_id: str, ref: RefWeights) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(aportaciones diarias por miembro, pesos al cierre por miembro, rentabilidad diaria del índice)."""
    ids = [m for m in ref.members if m in cal.columns]
    px = cal[ids]
    w = drifted_weights(px, {m: ref.members[m]["weight"] for m in ids})
    r = px.pct_change(fill_method=None)
    c = (w.shift(1) * r).where(w.shift(1).notna())
    r_ix = cal[index_id].pct_change(fill_method=None)
    return c.iloc[1:], w, r_ix.iloc[1:]


def period_start(dates: pd.DatetimeIndex, period: str) -> int | None:
    """Posición de la primera sesión del periodo (que termina en la última fecha)."""
    n = PERIODS[period]
    if n is None:
        first = dates[dates.year == dates[-1].year]
        return dates.get_loc(first[0]) if len(first) else None
    return len(dates) - n if len(dates) >= n else None


def linked(c: pd.DataFrame, r_ix: pd.Series, start: int) -> tuple[pd.Series, float]:
    """Aportaciones encadenadas desde la sesión `start` hasta el final y rentabilidad del índice."""
    rr = r_ix.iloc[start:].fillna(0.0)
    growth = (1 + rr).cumprod().shift(1, fill_value=1.0)
    contrib = c.iloc[start:].fillna(0.0).mul(growth, axis=0).sum()
    return contrib, float((1 + rr).prod() - 1)


def attribution_table(ix_id: str, cal: pd.DataFrame, ref: RefWeights, sector_names: dict[str, str], names: dict[str, str]) -> pd.DataFrame:
    """Una fila por (periodo, nivel, id): aportación, rentabilidad y peso al inicio y al final.

    Niveles: "index" (el total), "residual", "sector" y, si el índice tiene componentes, "member".
    """
    c, w, r_ix = member_contributions(cal, ix_id, ref)
    member_sector = {m: v["sector"] for m, v in ref.members.items() if m in c.columns}
    rows = []
    dates = c.index
    for period in PERIODS:
        start = period_start(dates, period)
        if start is None:
            continue
        contrib, total = linked(c, r_ix, start)
        px = cal.loc[dates[start:], list(member_sector)]
        base = cal[list(member_sector)].shift(1).loc[dates[start]]
        ret_m = px.iloc[-1] / base - 1
        w0 = w.shift(1).loc[dates[start], list(member_sector)]
        w1 = w.loc[dates[-1], list(member_sector)]
        common = {"index": ix_id, "period": period, "from": dates[start].date().isoformat(), "to": dates[-1].date().isoformat()}
        rows.append({**common, "level": "index", "id": ix_id, "name": ix_id, "sector": None, "contribution": total,
                     "ret": total, "weight_start": 1.0, "weight_end": 1.0})  # fmt: skip
        rows.append({**common, "level": "residual", "id": "residual", "name": "Residuo", "sector": None,
                     "contribution": total - float(contrib.sum()), "ret": np.nan, "weight_start": np.nan, "weight_end": np.nan})  # fmt: skip
        by_sector = pd.Series(member_sector)
        for sec, mem in by_sector.groupby(by_sector).groups.items():
            mem = list(mem)
            ws, we = float(w0[mem].sum()), float(w1[mem].sum())
            # rentabilidad del sector: aportación sobre el peso inicial, en términos de su propia cesta
            sec_ret = _basket_return(cal.loc[:, mem], w.loc[:, mem], dates, start)
            rows.append({**common, "level": "sector", "id": sec, "name": sector_names.get(sec, sec), "sector": sec,
                         "contribution": float(contrib[mem].sum()), "ret": sec_ret, "weight_start": ws, "weight_end": we})  # fmt: skip
        if ref.mode == "components":
            for m, sec in member_sector.items():
                rows.append({**common, "level": "member", "id": m, "name": names.get(m, m), "sector": sec,
                             "contribution": float(contrib[m]), "ret": float(ret_m[m]),
                             "weight_start": float(w0[m]), "weight_end": float(w1[m])})  # fmt: skip
    return pd.DataFrame(rows)


def _basket_return(px: pd.DataFrame, w: pd.DataFrame, dates: pd.DatetimeIndex, start: int) -> float:
    """Rentabilidad de la cesta de miembros de un sector (acciones fijas) en el periodo."""
    w_prev = w.shift(1).loc[dates[start]]
    if w_prev.sum() <= 0 or not np.isfinite(w_prev.sum()):
        return np.nan
    base = px.shift(1).loc[dates[start]]
    rel = px.loc[dates[-1]] / base
    ok = rel.notna() & w_prev.notna()
    return float((w_prev[ok] * rel[ok]).sum() / w_prev[ok].sum() - 1)


def sector_levels(cal: pd.DataFrame, ref: RefWeights) -> pd.DataFrame:
    """Nivel diario de cada sector (cesta encadenada de sus miembros), base 1 en la primera fecha."""
    ids = [m for m in ref.members if m in cal.columns]
    w = drifted_weights(cal[ids], {m: ref.members[m]["weight"] for m in ids})
    r = cal[ids].pct_change(fill_method=None)
    wp = w.shift(1)
    out = {}
    for sec in sorted({ref.members[m]["sector"] for m in ids}):
        mem = [m for m in ids if ref.members[m]["sector"] == sec]
        num = (wp[mem] * r[mem]).sum(axis=1, min_count=1)
        den = wp[mem].where(r[mem].notna()).sum(axis=1)
        out[sec] = (1 + (num / den).fillna(0.0)).cumprod()
    lv = pd.DataFrame(out)
    # antes de que exista ningún miembro del sector no hay nivel
    first = {sec: cal[[m for m in ids if ref.members[m]["sector"] == sec]].first_valid_index() for sec in lv}
    for sec, d in first.items():
        lv.loc[lv.index < d, sec] = np.nan
    return lv


def relative_strength(ix_id: str, cal: pd.DataFrame, ref: RefWeights, sector_names: dict[str, str]) -> pd.DataFrame:
    """Fuerza relativa de cada sector frente al índice por horizonte, y su cambio en un mes."""
    lv = sector_levels(cal, ref)
    ix = cal[ix_id]
    rows = []
    for sec in lv:
        s = lv[sec]
        for h, n in RS_HORIZONS.items():
            if s.notna().sum() <= n + MOMENTUM_LAG:
                continue
            r_s = s / s.shift(n) - 1
            r_i = ix / ix.shift(n) - 1
            rs = (1 + r_s) / (1 + r_i) - 1
            rows.append({"index": ix_id, "sector": sec, "name": sector_names.get(sec, sec), "horizon": h,
                         "sector_ret": float(r_s.iloc[-1]), "index_ret": float(r_i.iloc[-1]), "rs": float(rs.iloc[-1]),
                         "rs_change_1m": float(rs.iloc[-1] - rs.iloc[-1 - MOMENTUM_LAG])})  # fmt: skip
    return pd.DataFrame(rows)


def daily_sector_contributions(ix_id: str, cal: pd.DataFrame, ref: RefWeights, sessions: int = 520) -> pd.DataFrame:
    """Aportación diaria por sector (últimas `sessions`), para los gráficos de evolución."""
    c, _, r_ix = member_contributions(cal, ix_id, ref)
    by = pd.Series({m: v["sector"] for m, v in ref.members.items() if m in c.columns})
    daily = c.T.groupby(by).sum(min_count=1).T.tail(sessions)
    long = daily.stack(future_stack=True).rename("contribution").reset_index()
    long.columns = ["date", "sector", "contribution"]
    idx = r_ix.tail(sessions).rename("index_ret").reset_index()
    idx.columns = ["date", "index_ret"]
    return long.merge(idx, on="date").assign(index=ix_id)
