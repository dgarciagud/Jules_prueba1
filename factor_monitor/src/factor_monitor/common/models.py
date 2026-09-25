"""Modelos: MCO con errores HAC, Shapley (LMG) del R² y ridge con validación cruzada por sesiones."""

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


def _r2(yv: np.ndarray, xv: np.ndarray) -> float:
    if xv.shape[1] == 0:
        return 0.0
    Z = np.column_stack([np.ones(len(yv)), xv])
    beta, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    resid = yv - Z @ beta
    tss = ((yv - yv.mean()) ** 2).sum()
    return float(1 - (resid @ resid) / tss) if tss > 0 else 0.0


def shapley_lmg(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Descomposición Shapley (LMG) del R²: la suma de las contribuciones es el R² total."""
    yv, xv, _ = _clean(y, X)
    k = xv.shape[1]
    if k > MAX_SHAPLEY_REGRESSORS:
        raise ValueError(f"Shapley con {k} regresores es demasiado costoso (máx. {MAX_SHAPLEY_REGRESSORS})")
    r2 = {(): 0.0}
    for size in range(1, k + 1):
        for subset in combinations(range(k), size):
            r2[subset] = _r2(yv, xv[:, subset])
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


@dataclass
class RidgeModel:
    coef: pd.Series        # sobre factores estandarizados
    intercept: float
    x_mean: pd.Series
    x_std: pd.Series
    lam: float

    @property
    def betas(self) -> pd.Series:
        """Betas en las unidades originales de los factores."""
        return self.coef / self.x_std

    def predict(self, X: pd.DataFrame) -> pd.Series:
        Xs = (X[self.coef.index] - self.x_mean) / self.x_std
        return Xs @ self.coef + self.intercept

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aportación de cada factor al implícito (sin la constante)."""
        Xs = (X[self.coef.index] - self.x_mean) / self.x_std
        return Xs * self.coef


def _ridge_solve(yv: np.ndarray, xs: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    n, k = xs.shape
    ym = yv.mean()
    xm = xs.mean(axis=0)
    Xc, yc = xs - xm, yv - ym
    coef = np.linalg.solve(Xc.T @ Xc + lam * n * np.eye(k), Xc.T @ yc)
    return coef, float(ym - xm @ coef)


def ridge_fit(y: pd.Series, X: pd.DataFrame, lam: float) -> RidgeModel:
    yv, xv, _ = _clean(y, X)
    mean, std = xv.mean(axis=0), xv.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    coef, b0 = _ridge_solve(yv, (xv - mean) / std, lam)
    # El intercepto se expresa sobre los factores estandarizados con la media de la ventana.
    return RidgeModel(pd.Series(coef, X.columns), b0, pd.Series(mean, X.columns), pd.Series(std, X.columns), lam)


DEFAULT_LAMBDAS = np.concatenate([[0.0], np.logspace(-5, 0, 26)])


def ridge_cv(
    y: pd.Series,
    X: pd.DataFrame,
    groups: pd.Series,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
) -> RidgeModel:
    """Ridge con λ elegido dejando fuera una sesión cada vez (`groups` = id de sesión)."""
    df = pd.concat([y.rename("__y__"), X, groups.rename("__g__")], axis=1).dropna()
    yv = df["__y__"].to_numpy(float)
    xv = df[X.columns].to_numpy(float)
    g = df["__g__"].to_numpy()
    mean, std = xv.mean(axis=0), xv.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    xs = (xv - mean) / std
    uniq = np.unique(g)
    if len(uniq) < 2:
        return ridge_fit(y, X, float(lambdas[0]))
    errors = np.zeros(len(lambdas))
    for grp in uniq:
        test = g == grp
        for i, lam in enumerate(lambdas):
            coef, b0 = _ridge_solve(yv[~test], xs[~test], lam)
            errors[i] += ((yv[test] - xs[test] @ coef - b0) ** 2).sum()
    best = float(lambdas[int(np.argmin(errors))])
    return ridge_fit(y, X, best)
