"""Construcción de retornos, factores, factor de mercado y ortogonalización (§4.3, §5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .universe import Target, Universe


def log_returns(close: pd.DataFrame, bar_minutes: int = 5) -> pd.DataFrame:
    """Log-retornos sobre una rejilla regular. Una vela ausente deja NaN, sin rellenar."""
    if close.empty:
        return close.copy()
    grid = pd.date_range(close.index.min(), close.index.max(), freq=f"{bar_minutes}min")
    full = close.reindex(grid)
    return np.log(full).diff().reindex(close.index)


def build_factors(returns: pd.DataFrame, universe: Universe, factor_ids: list[str] | None = None) -> pd.DataFrame:
    """Retornos de los factores a partir de los retornos de los instrumentos."""
    out = {}
    for fid in factor_ids or list(universe.factors):
        f = universe.factors[fid]
        if any(i not in returns for i in f.instruments):
            continue
        if f.instrument is not None:
            out[fid] = f.sign * returns[f.instrument]
        else:
            out[fid] = f.sign * (returns[f.long] - returns[f.short])
    return pd.DataFrame(out, index=returns.index)


def market_return(returns: pd.DataFrame, target: Target, closed: pd.DataFrame | None = None) -> pd.Series | None:
    """Factor de mercado del objetivo (§5.2).

    Media de los miembros disponibles. `closed` (mismo índice, columnas = miembros)
    marca con True los instrumentos cuya bolsa está en festivo: no entran en la media.
    Devuelve None si el objetivo no tiene factor de mercado.
    """
    if target.market.type == "none":
        return None
    members = [m for m in target.market.members if m in returns]
    if not members:
        return pd.Series(np.nan, index=returns.index, name="MKT")
    block = returns[members].copy()
    if closed is not None:
        mask = closed.reindex(index=block.index, columns=members, fill_value=False).astype(bool)
        block = block.mask(mask)
    return block.mean(axis=1, skipna=True).rename("MKT")


def basket_return(component_returns: pd.DataFrame, weights: pd.Series, min_weight: float = 0.8) -> pd.Series:
    """Retorno de una cesta con pesos fijos.

    Si faltan componentes en una vela, se renormaliza entre los disponibles
    siempre que cubran al menos `min_weight` del peso total; si no, NaN.
    """
    w = weights.reindex(component_returns.columns).fillna(0.0)
    w = w / w.sum()
    avail = component_returns.notna()
    covered = avail.mul(w, axis=1).sum(axis=1)
    num = component_returns.fillna(0.0).mul(w, axis=1).sum(axis=1)
    out = num / covered.where(covered > 0)
    return out.where(covered >= min_weight - 1e-12)


@dataclass
class Orthogonalizer:
    """Ortogonaliza cada factor respecto al mercado con coeficientes fijados en la ventana."""

    alpha: pd.Series
    gamma: pd.Series

    @classmethod
    def fit(cls, factors: pd.DataFrame, market: pd.Series) -> "Orthogonalizer":
        alpha, gamma = {}, {}
        for col in factors:
            df = pd.concat([factors[col], market], axis=1, keys=["f", "m"]).dropna()
            m = df["m"].to_numpy()
            f = df["f"].to_numpy()
            var = m.var()
            g = float(((m - m.mean()) * (f - f.mean())).mean() / var) if var > 0 else 0.0
            gamma[col] = g
            alpha[col] = float(f.mean() - g * m.mean())
        return cls(pd.Series(alpha), pd.Series(gamma))

    def transform(self, factors: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
        cols = [c for c in factors if c in self.gamma]
        g, a = self.gamma[cols], self.alpha[cols]
        return factors[cols] - np.outer(market.to_numpy(), g.to_numpy()) - a.to_numpy()
