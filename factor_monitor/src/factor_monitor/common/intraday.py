"""Capa intradía (SPEC v3 §7). La usan el historial nocturno y el runner en vivo.

Flujo:
  1. `build_frame`: retornos de 5 min del objetivo, del mercado y de los factores,
     solo en velas de la sesión del objetivo, con id de sesión y franja de 30 min.
  2. `IntradayModel.fit`: con las 10 sesiones previas, ortogonaliza, estima ridge
     (λ por CV dejando fuera una sesión), σ por franja, ρ1-ρ2 de los residuos
     estandarizados, R² intradía y las desviaciones de implícito y real por L.
  3. `IntradayModel.evaluate`: con las velas cerradas de la sesión en curso calcula
     implícito, residuo, gap G_L, varianza con autocorrelación, z_L, descomposición y origen.
     Es una función pura de las velas: el runner y el historial obtienen lo mismo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendars import holiday_mask
from .factors import Orthogonalizer, basket_return, build_factors, log_returns
from .models import RidgeModel, ridge_cv
from .sessions import BAR_MINUTES, bucket_index, get_session, in_session, minutes_since_open, session_date
from .universe import Target, Universe

MKT = "MKT"
LENGTHS = (6, 12)
BUCKET_MINUTES = 30
NW_LAGS = 2
NO_ALERT_MINUTES = 30
ORIGIN_HIGH = 1.5
ORIGIN_LOW = 0.5
MIN_BUCKET_OBS = 10


# ---------------------------------------------------------------------------- datos


def build_frame(
    bars: dict[str, pd.DataFrame],
    target: Target,
    universe: Universe,
    factor_ids: list[str],
    basket_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Retornos de 5 min en la sesión del objetivo.

    Columnas: y, MKT (si el objetivo tiene mercado), factores, session (fecha local),
    bucket (franja de 30 min) y minute (minutos desde la apertura).
    """
    closes = pd.DataFrame({k: v["close"] for k, v in bars.items() if v is not None and not v.empty}).sort_index()
    if closes.empty:
        return pd.DataFrame()
    rets = log_returns(closes)
    rets = rets[in_session(rets.index, target.session)]
    if target.exchange:
        rets = rets[~holiday_mask(rets.index, target.exchange, target.session)]
    if rets.empty:
        return pd.DataFrame()
    if target.basket:
        comps = [c for c in target.basket if c in rets]
        w = basket_weights if basket_weights is not None else pd.Series(1.0, index=comps)
        y = basket_return(rets[comps], w.reindex(comps).fillna(0.0)) if comps else pd.Series(np.nan, index=rets.index)
    else:
        y = rets.get(target.id, pd.Series(np.nan, index=rets.index))
    out = pd.DataFrame({"y": y}, index=rets.index)
    if target.market.type == "mean":
        members = [m for m in target.market.members if m in rets]
        block = rets[members].copy()
        for m in members:
            inst = universe.instruments[m]
            if inst.exchange and inst.session:
                block.loc[holiday_mask(block.index, inst.exchange, inst.session), m] = np.nan
        out[MKT] = block.mean(axis=1, skipna=True)
    out = out.join(build_factors(rets, universe, factor_ids))
    out["session"] = session_date(out.index, target.session)
    out["bucket"] = bucket_index(out.index, target.session, BUCKET_MINUTES)
    out["minute"] = minutes_since_open(out.index, target.session)
    return out


def last_sessions(frame: pd.DataFrame, before: pd.Timestamp, n: int = 10) -> pd.DataFrame:
    """Las `n` sesiones completas anteriores a la fecha `before` (sin incluirla)."""
    sessions = [s for s in frame["session"].unique() if s < before]
    keep = sorted(sessions)[-n:]
    return frame[frame["session"].isin(keep)]


# ---------------------------------------------------------------------------- utilidades


def _rolling_within_session(values: pd.Series, sessions: pd.Series, length: int) -> pd.Series:
    """Suma móvil de `length` velas sin cruzar sesiones (NaN hasta tener `length` velas)."""
    return values.groupby(sessions.to_numpy()).transform(lambda s: s.rolling(length, min_periods=length).sum())


def _autocorr_within_sessions(x: pd.Series, sessions: pd.Series, lag: int) -> float:
    lagged = x.groupby(sessions.to_numpy()).shift(lag)
    df = pd.concat([x, lagged], axis=1).dropna()
    if len(df) < 10 or df.iloc[:, 0].std() == 0:
        return 0.0
    return float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1])


def design(frame: pd.DataFrame, factors: list[str], has_market: bool, ortho: Orthogonalizer | None) -> pd.DataFrame:
    """Matriz de regresores: mercado + factores ortogonalizados respecto al mercado."""
    X = frame[factors].copy()
    if has_market:
        if ortho is not None and factors:
            X = ortho.transform(X, frame[MKT])
        X.insert(0, MKT, frame[MKT])
    return X


# ---------------------------------------------------------------------------- modelo


@dataclass
class IntradayModel:
    target: str
    factors: list[str]
    has_market: bool
    ridge: RidgeModel
    ortho: Orthogonalizer | None
    sigma_bucket: pd.Series
    rho: tuple[float, ...]
    r2: float
    s_implied: dict[int, float]
    s_real: dict[int, float]
    sessions: list = field(default_factory=list)

    @property
    def regressors(self) -> list[str]:
        return ([MKT] if self.has_market else []) + self.factors

    def variance_factor(self, length: int) -> float:
        """Varianza de una suma de `length` residuos respecto a la suma de sus varianzas.

        Var(Σ ε) = Σ σ² · [1 + 2 Σ_k (1 − k/L) ρ_k], con ρ_k hasta el retardo NW_LAGS.
        Son los pesos exactos para una suma de L términos; los de Bartlett (1 − k/(q+1))
        infravaloran la varianza con autocorrelación positiva e inflan el z.
        """
        adj = 1 + 2 * sum((1 - k / length) * r for k, r in enumerate(self.rho, start=1) if k < length)
        return max(adj, 0.25)

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        return design(frame, self.factors, self.has_market, self.ortho)

    # -------------------------------------------------------------- estimación

    @classmethod
    def fit(cls, frame: pd.DataFrame, target: str, factors: list[str], has_market: bool) -> "IntradayModel":
        cols = ["y"] + ([MKT] if has_market else []) + factors
        data = frame.dropna(subset=cols)
        if len(data) < 50:
            raise ValueError(f"{target}: solo {len(data)} velas válidas para estimar")
        ortho = Orthogonalizer.fit(data[factors], data[MKT]) if has_market and factors else None
        X = design(data, list(factors), has_market, ortho)
        ridge = ridge_cv(data["y"], X, data["session"].astype("int64"))
        implied = ridge.predict(X)
        resid = data["y"] - implied
        tss = ((data["y"] - data["y"].mean()) ** 2).sum()
        r2 = float(1 - (resid**2).sum() / tss) if tss > 0 else 0.0

        global_sigma = float(resid.std())
        by_bucket = resid.groupby(data["bucket"])
        sigma = by_bucket.std().where(by_bucket.size() >= MIN_BUCKET_OBS, global_sigma).fillna(global_sigma)
        std_resid = resid / data["bucket"].map(sigma)
        rho = tuple(_autocorr_within_sessions(std_resid, data["session"], k) for k in range(1, NW_LAGS + 1))

        s_imp, s_real = {}, {}
        for L in LENGTHS:
            s_imp[L] = float(_rolling_within_session(implied, data["session"], L).std())
            s_real[L] = float(_rolling_within_session(data["y"], data["session"], L).std())
        return cls(
            target, list(factors), has_market, ridge, ortho, sigma, rho, r2, s_imp, s_real,
            sessions=sorted(data["session"].unique()),
        )  # fmt: skip

    # -------------------------------------------------------------- evaluación

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Métricas por vela cerrada de una o varias sesiones (las ventanas no cruzan sesiones)."""
        cols = ["y"] + self.regressors
        data = frame.dropna(subset=cols)
        out = pd.DataFrame(index=data.index)
        if data.empty:
            return out
        X = self._design(data)
        contrib = self.ridge.contributions(X)
        implied = contrib.sum(axis=1) + self.ridge.intercept
        resid = data["y"] - implied
        sess = data["session"]
        sigma2 = data["bucket"].map(self.sigma_bucket).fillna(float(self.sigma_bucket.mean())) ** 2
        out["session"] = sess
        out["minute"] = data["minute"]
        out["y"] = data["y"]
        out["implied"] = implied
        out["resid"] = resid
        out["eligible"] = data["minute"] >= NO_ALERT_MINUTES
        for L in LENGTHS:
            gap = _rolling_within_session(resid, sess, L)
            var = _rolling_within_session(sigma2, sess, L) * self.variance_factor(L)
            out[f"gap_{L}"] = gap
            out[f"z_{L}"] = gap / np.sqrt(var)
            I = _rolling_within_session(implied, sess, L)
            R = _rolling_within_session(data["y"], sess, L)
            out[f"implied_{L}"] = I
            out[f"real_{L}"] = R
            out[f"origin_{L}"] = classify_origin(I, R, self.s_implied[L], self.s_real[L])
            for f in contrib.columns:
                out[f"contrib_{L}_{f}"] = _rolling_within_session(contrib[f], sess, L)
        return out


def classify_origin(implied: pd.Series, real: pd.Series, s_implied: float, s_real: float) -> pd.Series:
    """`driver_moved`, `target_moved` o `mixed` (§7.2)."""
    ai, ar = implied.abs(), real.abs()
    driver = (ai > ORIGIN_HIGH * s_implied) & (ar < ORIGIN_LOW * ai)
    target = (ai < ORIGIN_LOW * s_implied) & (ar > ORIGIN_HIGH * s_real)
    out = pd.Series("mixed", index=implied.index, dtype=object)
    out[driver] = "driver_moved"
    out[target] = "target_moved"
    out[implied.isna() | real.isna()] = None
    return out


def main_factor(row: pd.Series, L: int, regressors: list[str]) -> str | None:
    """Factor con mayor aportación absoluta al implícito acumulado en L."""
    vals = {f: row.get(f"contrib_{L}_{f}") for f in regressors}
    vals = {f: abs(v) for f, v in vals.items() if v is not None and np.isfinite(v)}
    return max(vals, key=vals.get) if vals else None
