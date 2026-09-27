"""Modelos: MCO con errores HAC y descomposición Shapley (LMG) del R²."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

MAX_SHAPLEY_REGRESSORS = 10


def _clean(y: pd.Series, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    return df["__y__"].to_numpy(float), df[X.columns].to_numpy(float), df.index


def newey_west_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


@dataclass
class OLSResult:
    params: pd.Series          # incluye "const"
    se_hac: pd.Series
    tvalues: pd.Series
    std_betas: pd.Series       # beta · std(x) / std(y), sin constante
    r2: float
    nobs: int


def ols(y: pd.Series, X: pd.DataFrame, hac_lags: int | None = None) -> OLSResult:
    yv, xv, _ = _clean(y, X)
    n, k = xv.shape
    Z = np.column_stack([np.ones(n), xv])
    beta, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    resid = yv - Z @ beta
    tss = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / tss if tss > 0 else 0.0

    lags = newey_west_lags(n) if hac_lags is None else hac_lags
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    u = Z * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = ZtZ_inv @ S @ ZtZ_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    names = ["const", *X.columns]
    params = pd.Series(beta, index=names)
    se_s = pd.Series(se, index=names)
    sy = yv.std()
    std_b = pd.Series(beta[1:] * xv.std(axis=0) / sy if sy > 0 else np.zeros(k), index=X.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = params / se_s
    return OLSResult(params, se_s, t, std_b, float(r2), n)


def shapley_lmg(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Descomposición Shapley (LMG) del R²: la suma de las contribuciones es el R² total.

    El R² de cada subconjunto S sale de las covarianzas, R²_S = c_Sᵀ Σ_SS⁻¹ c_S / var(y),
    sin repetir la regresión.
    """
    yv, xv, _ = _clean(y, X)
    k = xv.shape[1]
    if k > MAX_SHAPLEY_REGRESSORS:
        raise ValueError(f"Shapley con {k} regresores es demasiado costoso (máx. {MAX_SHAPLEY_REGRESSORS})")
    if len(yv) < 3 or yv.var() <= 0:
        return pd.Series(np.zeros(k), index=X.columns)
    xc, yc = xv - xv.mean(axis=0), yv - yv.mean()
    sxx, sxy, syy = xc.T @ xc, xc.T @ yc, yc @ yc
    r2 = {(): 0.0}
    for size in range(1, k + 1):
        for subset in combinations(range(k), size):
            idx = list(subset)
            beta = np.linalg.lstsq(sxx[np.ix_(idx, idx)], sxy[idx], rcond=None)[0]
            r2[subset] = float(sxy[idx] @ beta / syy)
    contrib = np.zeros(k)
    for j in range(k):
        others = [i for i in range(k) if i != j]
        for size in range(k):
            w = factorial(size) * factorial(k - size - 1) / factorial(k)
            for s in combinations(others, size):
                with_j = tuple(sorted((*s, j)))
                contrib[j] += w * (r2[with_j] - r2[s])
    return pd.Series(contrib, index=X.columns)


def shapley_shares(shapley: pd.Series, market: str | None = None) -> pd.Series:
    """Cuota de cada factor no de mercado (§6).

    Con mercado: Shapley_f / (R² − Shapley_mercado). Sin mercado: Shapley_f / R².
    """
    total = shapley.sum()
    if market is not None:
        denom = total - shapley.get(market, 0.0)
        s = shapley.drop(market, errors="ignore")
    else:
        denom, s = total, shapley
    if denom <= 0:
        return s * 0.0
    return s / denom
