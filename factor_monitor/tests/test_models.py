import numpy as np
import pandas as pd
import pytest

from factor_monitor.common.models import ols, ridge_cv, shapley_lmg, shapley_shares


def synthetic(n=3000, betas=(0.8, -0.5, 0.0), noise=0.5, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(betas))), columns=[f"f{i}" for i in range(len(betas))])
    y = pd.Series(X.to_numpy() @ np.array(betas) + noise * rng.normal(size=n))
    return y, X


def test_ols_recovers_betas_with_hac():
    y, X = synthetic()
    res = ols(y, X)
    assert res.params[["f0", "f1", "f2"]].to_numpy() == pytest.approx([0.8, -0.5, 0.0], abs=0.03)
    assert abs(res.tvalues["f0"]) > 10 and abs(res.tvalues["f2"]) < 4
    assert 0 < res.r2 < 1


def test_shapley_sums_to_r2():
    rng = np.random.default_rng(2)
    y, X = synthetic(betas=(0.5, 0.3, 0.2, 0.1))
    X["f1"] += 0.6 * X["f0"]  # regresores correlacionados
    s = shapley_lmg(y, X)
    assert s.sum() == pytest.approx(ols(y, X).r2, abs=1e-12)
    assert (s >= -1e-12).all()


def test_shapley_shares_exclude_market():
    s = pd.Series({"MKT": 0.5, "A": 0.15, "B": 0.05})
    shares = shapley_shares(s, market="MKT")
    assert shares["A"] == pytest.approx(0.75) and "MKT" not in shares
    assert shapley_shares(s)["A"] == pytest.approx(0.15 / 0.7)


def test_ridge_recovers_betas_with_session_cv():
    y, X = synthetic(n=1020, noise=0.3)
    groups = pd.Series(np.repeat(np.arange(10), 102))
    model = ridge_cv(y, X, groups)
    assert model.betas.to_numpy() == pytest.approx([0.8, -0.5, 0.0], abs=0.05)
    pred = model.predict(X)
    contrib = model.contributions(X)
    assert np.allclose(contrib.sum(axis=1) + model.intercept, pred)


def test_ridge_shrinks_noise_only_model():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(size=300))
    model = ridge_cv(y, X, pd.Series(np.repeat(np.arange(10), 30)))
    assert model.lam > 0
    assert np.abs(model.betas).max() < 0.15
