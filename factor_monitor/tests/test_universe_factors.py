import numpy as np
import pandas as pd
import pytest

from factor_monitor.common.factors import (
    Orthogonalizer,
    basket_return,
    build_factors,
    log_returns,
    market_return,
)
from factor_monitor.common.universe import UniverseError, load_universe, parse_universe


@pytest.fixture(scope="module")
def universe():
    return load_universe()


def test_config_loads(universe):
    assert set(universe.targets) == {"SPX", "SX5E", "DAX", "CAC", "IBEX", "BANKS", "ENERGY"}


def test_spx_has_no_ia_and_no_market(universe):
    cands, _ = universe.effective_candidates("SPX")
    assert "IA" not in cands
    assert universe.targets["SPX"].market.type == "none"


def test_overlap_rule_excludes_ia_even_if_configured(universe):
    raw = {
        "instruments": {k: {"kind": v.kind} for k, v in universe.instruments.items()},
        "factors": {"IA": {"long": "NDX", "short": "SPX", "contains": {"SPX": 1.0}}},
        "targets": {"SPX": {"session": "us", "candidates": ["IA"], "market": {"type": "none"}}},
    }
    u = parse_universe(raw)
    cands, excl = u.effective_candidates("SPX")
    assert cands == () and excl[0].factor == "IA"


def test_market_cannot_contain_target():
    raw = {
        "instruments": {"A": {"kind": "target"}, "B": {"kind": "target"}},
        "factors": {},
        "targets": {"A": {"session": "eu", "market": {"type": "mean", "members": ["A", "B"]}}},
    }
    with pytest.raises(UniverseError, match="leave-one-out"):
        parse_universe(raw)


def test_build_factors_signs(universe):
    idx = pd.RangeIndex(3)
    r = pd.DataFrame(
        {"BRENT": [0.01, 0, 0], "BUND": [0.02, 0, 0], "EURUSD": [0.03, 0, 0], "NDX": [0.05, 0, 0], "SPX": [0.01, 0, 0]},
        index=idx,
    )
    f = build_factors(r, universe)
    assert f.loc[0, "BRENT"] == pytest.approx(0.01)
    assert f.loc[0, "BUND"] == pytest.approx(-0.02)
    assert f.loc[0, "USD"] == pytest.approx(-0.03)
    assert f.loc[0, "IA"] == pytest.approx(0.04)
    assert "TBOND" not in f  # sin datos, no se construye


def test_market_leave_one_out_skips_holidays(universe):
    r = pd.DataFrame({"SX5E": [0.01, 0.01], "CAC": [0.02, 0.02], "IBEX": [0.06, 0.06]})
    closed = pd.DataFrame({"IBEX": [False, True]})
    m = market_return(r, universe.targets["DAX"], closed)
    assert m.iloc[0] == pytest.approx(0.03)
    assert m.iloc[1] == pytest.approx(0.015)
    assert market_return(r, universe.targets["SPX"]) is None


def test_log_returns_gap_is_nan():
    idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-15 08:00", "2024-01-15 08:05", "2024-01-15 08:15"], utc=True))
    r = log_returns(pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=idx))
    assert np.isnan(r["A"].iloc[0]) and np.isnan(r["A"].iloc[2])
    assert r["A"].iloc[1] == pytest.approx(np.log(1.01))


def test_orthogonalization_zero_correlation():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(size=2000))
    F = pd.DataFrame({"A": 0.7 * m + rng.normal(size=2000), "B": -0.3 * m + rng.normal(size=2000)})
    ortho = Orthogonalizer.fit(F, m)
    Fo = ortho.transform(F, m)
    for c in Fo:
        assert abs(np.corrcoef(Fo[c], m)[0, 1]) < 1e-10
    assert ortho.gamma["A"] == pytest.approx(0.7, abs=0.05)


def test_basket_return_renormalizes():
    comp = pd.DataFrame({"X": [0.01, 0.01, np.nan], "Y": [0.03, np.nan, np.nan]})
    w = pd.Series({"X": 0.9, "Y": 0.1})
    b = basket_return(comp, w)
    assert b.iloc[0] == pytest.approx(0.012)
    assert b.iloc[1] == pytest.approx(0.01)  # cubre el 90 % ≥ 80 %
    assert np.isnan(b.iloc[2])
