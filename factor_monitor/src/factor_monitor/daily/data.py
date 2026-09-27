"""Paso 1 del informe diario: descarga de cierres, pesos sectoriales y comprobación.

- `download_all`: todo el histórico diario de cada serie (Yahoo; si falla, Stooq; si
  también falla, la copia anterior marcada como antigua).
- `reference_weights`: pesos de hoy de cada ETF sectorial (S&P 500) o acción (CAC 40),
  estimados con el propio índice (`estimate_weights`).
- `drifted_weights`: pesos hacia atrás arrastrados por precio, w_i,t ∝ w_i,T · P_i,t / P_i,T,
  que es como evoluciona un índice ponderado por capitalización entre revisiones.
- `sector_returns` y `tracking`: rentabilidad de cada sector y cuánto reproduce la suma
  ponderada de sectores la rentabilidad del índice (la validación de los pesos).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common.runlog import RunLog
from .sources import SourceError, Stooq, Yahoo
from .universe import DailyUniverse, Index, Series

log = logging.getLogger(__name__)

STALE_DAYS = 5          # una serie cuyo último dato es más antiguo que el del índice en estos días hábiles se marca
TRACKING_WINDOW = 250   # sesiones para validar los pesos


# ----------------------------------------------------------------------------- descarga


def download_all(
    universe: DailyUniverse,
    yahoo: Yahoo,
    stooq: Stooq | None,
    previous: pd.DataFrame | None,
    runlog: RunLog,
    start: str = "1995-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (precios en formato largo, cobertura por serie)."""
    frames, rows = [], []
    for s in universe.all_series():
        df, source, errors = None, None, []
        for name, fn in (("yahoo", lambda: yahoo.history(s.yahoo, start) if s.yahoo else None),
                         ("stooq", lambda: stooq.history(s.stooq) if (stooq and s.stooq) else None)):  # fmt: skip
            try:
                got = fn()
            except SourceError as e:
                errors.append(f"{name}: {e}")
                continue
            if got is not None:
                df, source = got, name
                break
        prev_s = previous[previous["id"] == s.id] if previous is not None else None
        if df is not None and prev_s is not None and len(prev_s) and len(df) < 0.9 * len(prev_s):
            # descarga truncada (Yahoo a veces devuelve solo los últimos días): se completa con la copia anterior
            older = prev_s[prev_s["date"] < df["date"].min()][["date", "close", "adjclose"]]
            runlog.warn("data", f"{s.id}: {source} devolvió {len(df)} sesiones frente a {len(prev_s)} de la copia anterior; "
                                f"se completa con ella")  # fmt: skip
            df = pd.concat([older, df], ignore_index=True)
            source = f"{source}+anterior"
        if df is None and prev_s is not None and len(prev_s):
            df = prev_s[["date", "close", "adjclose"]].copy()
            source = "anterior"
            runlog.warn("data", f"{s.id}: sin descarga ({'; '.join(errors)}); se usa la copia anterior")
        elif df is None:
            runlog.warn("data", f"{s.id}: sin datos ({'; '.join(errors) or 'sin símbolo'})")
        if df is not None:
            frames.append(df.assign(id=s.id, source=source))
        rows.append(
            {
                "id": s.id, "name": s.name, "role": s.role, "index": s.index, "source": source,
                "symbol": s.yahoo if str(source).startswith("yahoo") else s.stooq if str(source).startswith("stooq") else None,
                "first": df["date"].min() if df is not None else pd.NaT,
                "last": df["date"].max() if df is not None else pd.NaT,
                "n": 0 if df is None else len(df),
                "errors": "; ".join(errors),
            }
        )  # fmt: skip
    prices = pd.concat(frames, ignore_index=True)[["date", "id", "close", "adjclose", "source"]] if frames else pd.DataFrame(columns=["date", "id", "close", "adjclose", "source"])
    coverage = pd.DataFrame(rows)
    return prices, coverage


def mark_stale(coverage: pd.DataFrame, universe: DailyUniverse, runlog: RunLog) -> pd.DataFrame:
    """Marca las series cuyo último dato va muy por detrás del de su índice (o del índice más reciente)."""
    last_idx = coverage[coverage["role"] == "index"].set_index("id")["last"]
    ref_all = last_idx.max()
    stale = []
    for _, r in coverage.iterrows():
        ref = last_idx.get(r["index"], ref_all) if r["index"] else ref_all
        if pd.isna(r["last"]) or pd.isna(ref):
            stale.append(True)
            continue
        lag = len(pd.bdate_range(r["last"], ref)) - 1
        stale.append(lag > STALE_DAYS)
        if lag > STALE_DAYS:
            runlog.warn("data", f"{r['id']}: último dato {r['last']:%Y-%m-%d}, {lag} días hábiles por detrás")
    return coverage.assign(stale=stale)


def wide(prices: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    return prices.pivot_table(index="date", columns="id", values=field, aggfunc="last").sort_index()


# ----------------------------------------------------------------------------- pesos


def nnls(A: np.ndarray, b: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Mínimos cuadrados no negativos (Lawson–Hanson)."""
    m, n = A.shape
    x = np.zeros(n)
    passive = np.zeros(n, bool)
    w = A.T @ (b - A @ x)
    tol = 1e-12 * max(1.0, np.abs(A).max()) * max(m, n)
    for _ in range(max_iter):
        if passive.all() or w[~passive].max(initial=-np.inf) <= tol:
            break
        j = np.argmax(np.where(passive, -np.inf, w))
        passive[j] = True
        while True:
            z = np.zeros(n)
            z[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if (z[passive] > tol).all():
                x = z
                break
            neg = passive & (z <= tol)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            passive &= x > tol
        w = A.T @ (b - A @ x)
    return x


def estimate_weights(level: pd.Series, members: pd.DataFrame, min_obs: int = 60,
                     prior: dict[str, float] | None = None, shrink: float = 0.0) -> tuple[dict[str, float], dict]:
    """Pesos de referencia (en la última fecha) que mejor reproducen el índice.

    Un índice ponderado por capitalización es una cesta de acciones fijas entre revisiones:
    I_t / I_T = Σ w_i · P_i,t / P_i,T. En diferencias, ΔI_t / I_T = Σ w_i · ΔP_i,t / P_i,T,
    lineal en w; se resuelve con w ≥ 0 y Σ w = 1 (fila de penalización). Con `prior`, se añade
    shrink · ||w − prior||² (escalado por la varianza media de las columnas) para que los datos
    solo corrijan los pesos de referencia donde lo justifican.
    """
    px = members.loc[level.index].dropna(axis=1, how="any")
    if len(px) < min_obs + 1 or px.shape[1] == 0:
        raise ValueError(f"ventana insuficiente ({len(px)} sesiones, {px.shape[1]} miembros)")
    A = (px.diff() / px.iloc[-1]).iloc[1:].to_numpy()
    b = (level.diff() / level.iloc[-1]).iloc[1:].to_numpy()
    lam = 10 * np.linalg.norm(A) / np.sqrt(A.size)
    M, y = np.vstack([A, lam * np.ones(A.shape[1])]), np.append(b, lam)
    if prior and shrink > 0:
        p = np.array([prior.get(c, 0.0) for c in px.columns])
        p = p / p.sum() if p.sum() > 0 else p
        mu = np.sqrt(shrink * (A**2).sum() / A.shape[1])
        M, y = np.vstack([M, mu * np.eye(A.shape[1])]), np.concatenate([y, mu * p])
    w = nnls(M, y)
    w = w / w.sum()
    fit = A @ w
    ss = ((b - b.mean()) ** 2).sum()
    info = {"from": px.index[0].date().isoformat(), "to": px.index[-1].date().isoformat(), "n": int(len(b)),
            "r2": float(1 - ((b - fit) ** 2).sum() / ss) if ss > 0 else None,
            "excluded": sorted(set(members.columns) - set(px.columns))}  # fmt: skip
    return dict(zip(px.columns, w.tolist())), info


@dataclass
class RefWeights:
    """Pesos de referencia de un índice en `as_of`."""

    index: str
    as_of: str
    mode: str                        # etf | components
    sector_weights: dict[str, float]  # sector -> peso (suma 1)
    members: dict[str, dict]          # id -> {sector, weight}
    source: str
    fit: dict | None = None
    comparison: dict | None = None    # pesos sectoriales de un fondo (solo como referencia)

    def to_json(self) -> dict:
        return self.__dict__.copy()


def reference_weights(ix: Index, cal: pd.DataFrame, yahoo: Yahoo | None, previous: dict | None,
                      runlog: RunLog, window: int = TRACKING_WINDOW) -> RefWeights | None:  # fmt: skip
    """Pesos estimados con el propio índice en la ventana desde la última revisión (máx. `window` sesiones)."""
    members = {s.id: s.sector for s in ix.sectors.values()} if ix.sector_mode == "etf" else {c.id: c.sector for c in ix.components}
    missing = [m for m in members if m not in cal.columns]
    for m in missing:
        runlog.warn("weights", f"{ix.id}: {m} sin precios; queda fuera")
    win = cal.tail(window)
    if ix.composition_since:
        since = win[win.index >= pd.Timestamp(ix.composition_since)]
        if len(since) > 60:
            win = since
        else:
            runlog.warn("weights", f"{ix.id}: solo {len(since)} sesiones desde la revisión del {ix.composition_since}; "
                                   f"los pesos se estiman con las últimas {len(win)}")  # fmt: skip
    comparison, prior = None, None
    if ix.weights_from_fund and yahoo is not None:
        try:
            fund = yahoo.fund_sectors(ix.weights_from_fund)
            comparison = {"fund": ix.weights_from_fund, "weights": fund}
            if ix.sector_mode == "etf":
                prior = {s.id: fund.get(sec, 0.0) for sec, s in ix.sectors.items()}
        except SourceError as e:
            runlog.warn("weights", f"{ix.id}: sin pesos de {ix.weights_from_fund} ({e}); estimación sin referencia")
    if prior is None and previous and ix.sector_mode == "etf":
        prior = {m: v["weight"] for m, v in previous.get("members", {}).items()}
    try:
        w, info = estimate_weights(win[ix.id], win[[m for m in members if m in cal.columns]], prior=prior, shrink=ix.shrink)
    except ValueError as e:
        if previous:
            runlog.warn("weights", f"{ix.id}: sin estimación ({e}); se usan los pesos de {previous['as_of']}")
            return RefWeights(**previous)
        runlog.warn("weights", f"{ix.id}: sin estimación de pesos ({e})")
        return None
    for m in info["excluded"]:
        runlog.warn("weights", f"{ix.id}: {m} con huecos en la ventana; queda fuera de la estimación")
    mem = {m: {"sector": members[m], "weight": v} for m, v in w.items()}
    sw: dict[str, float] = {}
    for m in mem.values():
        sw[m["sector"]] = sw.get(m["sector"], 0.0) + m["weight"]
    return RefWeights(ix.id, cal.index[-1].date().isoformat(), ix.sector_mode, sw, mem,
                      "estimados con el índice (MCO no negativos, suma 1"
                      + (f", encogidos hacia {ix.weights_from_fund})" if prior and ix.shrink else ")"), info, comparison)  # fmt: skip


def drifted_weights(close: pd.DataFrame, ref: dict[str, float]) -> pd.DataFrame:
    """Pesos diarios arrastrados por precio desde la fecha de referencia (la última fila).

    w_i,t ∝ w_i,T · P_i,t / P_i,T; un activo sin precio en t pesa 0 y el resto se renormaliza.
    """
    cols = [c for c in ref if c in close.columns]
    px = close[cols].ffill()
    last = px.iloc[-1]
    raw = px.div(last) * pd.Series({c: ref[c] for c in cols})
    return raw.div(raw.sum(axis=1), axis=0)


def sector_returns(ix: Index, close: pd.DataFrame, ref: RefWeights) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rentabilidades diarias por sector, pesos sectoriales diarios al cierre)."""
    ids = [m for m in ref.members if m in close.columns]
    px = close[ids]
    w_m = drifted_weights(px, {m: ref.members[m]["weight"] for m in ids})
    r_m = px.pct_change(fill_method=None)
    sectors = sorted({ref.members[m]["sector"] for m in ids})
    rets, weights = {}, {}
    w_prev = w_m.shift(1)  # la rentabilidad de t se pondera con los pesos al cierre de t-1
    for sec in sectors:
        mem = [m for m in ids if ref.members[m]["sector"] == sec]
        wp = w_prev[mem]
        rets[sec] = (wp * r_m[mem]).sum(axis=1, min_count=1) / wp.where(r_m[mem].notna()).sum(axis=1)
        weights[sec] = w_m[mem].sum(axis=1)
    return pd.DataFrame(rets), pd.DataFrame(weights)


def tracking(index_close: pd.Series, rets: pd.DataFrame, weights: pd.DataFrame, window: int = TRACKING_WINDOW) -> dict:
    """Cuánto reproduce Σ w_{t-1}·r_t la rentabilidad del índice en las últimas `window` sesiones."""
    r_ix = index_close.pct_change(fill_method=None)
    implied = (weights.shift(1) * rets).sum(axis=1, min_count=1)
    df = pd.concat([r_ix.rename("ix"), implied.rename("impl")], axis=1).dropna().tail(window)
    if len(df) < 20:
        return {"n": len(df), "r2": None, "corr": None, "te_annual": None, "beta": None}
    corr = float(df["ix"].corr(df["impl"]))
    diff = df["ix"] - df["impl"]
    beta = float(np.polyfit(df["impl"], df["ix"], 1)[0])
    return {
        "n": len(df),
        "corr": corr,
        "r2": corr**2,
        "te_annual": float(diff.std() * np.sqrt(252)),
        "beta": beta,
        "from": df.index[0].date().isoformat(),
        "to": df.index[-1].date().isoformat(),
    }
