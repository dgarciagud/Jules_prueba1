"""Historial de alertas y calibración de umbrales (SPEC v3 §8.1, §9).

Reproduce sobre el histórico de 5 minutos de Dukascopy la capa intradía y las
alertas con el mismo código de `common/`, sin mirar al futuro:
  - la selección de factores es la vigente en cada sesión (`selection_history`);
  - cada sesión se evalúa con un modelo estimado en las 10 sesiones anteriores;
  - el filtro de R² usa la mediana de las 250 sesiones anteriores;
  - el umbral z* se recalibra el primer día hábil de cada mes con las 250 sesiones
    anteriores y se mantiene durante el mes (el mismo valor lo usa el monitor en vivo);
  - los eventos bloquean según el calendario.

Resultado de cada alerta, con el gap anclado C_k = G_t0 + Σ_{j=1..k} ε_{t0+j}:
convergencia (|C_k| ≤ 0,5·|G_t0|), retorno en el sentido de la convergencia,
quién ajusta y reversión neta de costes, a k = 6 y 12 velas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common.alerts import ALERT_COLUMNS, alert_positions, find_alerts
from ..common.events import block_mask
from ..common.intraday import LENGTHS, IntradayModel, build_frame, last_sessions
from ..common.runlog import RunLog
from ..common.universe import Universe

log = logging.getLogger(__name__)

STEP = "track_record"
FIT_SESSIONS = 10
LOOKBACK_SESSIONS = 250
MIN_HISTORY_SESSIONS = 20
HORIZONS = (6, 12)
TARGET_ALERTS_PER_WEEK = 2.0
Z_FLOOR = 2.0
Z_DEFAULT = 2.5
Z_MAX = 10.0
MIN_CELL = 30
MIN_YEAR = 10


@dataclass
class TrackRecord:
    alerts: pd.DataFrame        # una fila por alerta con sus resultados
    intraday_r2: pd.DataFrame   # session, target, r2
    thresholds: dict            # z* y mediana de R² vigentes por objetivo


# ---------------------------------------------------------------------------- selección vigente


def selection_for(history: pd.DataFrame, target: str, session: pd.Timestamp) -> tuple[list[str], bool] | None:
    """Factores seleccionados vigentes en `session` y si el objetivo tiene driver macro."""
    if history is None or history.empty:
        return None
    h = history[(history["target"] == target) & (history["valid_from"] <= session)]
    if h.empty:
        return None
    last = h[h["as_of"] == h["as_of"].max()]
    if session > last["valid_from"].iloc[0] + pd.Timedelta(days=6):
        return None  # selección caducada (p. ej., el nocturno no corrió esa semana)
    factors = last.loc[last["selected"], "factor"].tolist()
    return factors, bool((last["status"] == "ok").any())


# ---------------------------------------------------------------------------- calibración


def count_alerts(z: np.ndarray, base: np.ndarray, sessions: np.ndarray, ts_ns: np.ndarray, z_star: float) -> int:
    return len(alert_positions(z, base, sessions, ts_ns, z_star))


def calibrate_z(evals: pd.DataFrame, base: np.ndarray, rate_per_session: float) -> float:
    """z* ≥ Z_FLOOR que produce `rate_per_session` alertas por sesión en `evals` (bisección)."""
    n_sessions = evals["session"].nunique()
    if n_sessions < MIN_HISTORY_SESSIONS:
        return Z_DEFAULT
    z = np.column_stack([evals[f"z_{L}"].to_numpy(float) for L in LENGTHS])
    sess = evals["session"].to_numpy()
    ts = evals.index.as_unit("ns").asi8
    target = rate_per_session * n_sessions
    if count_alerts(z, base, sess, ts, Z_FLOOR) <= target:
        return Z_FLOOR
    lo, hi = Z_FLOOR, Z_MAX
    for _ in range(20):
        mid = (lo + hi) / 2
        if count_alerts(z, base, sess, ts, mid) > target:
            lo = mid
        else:
            hi = mid
    return round(hi, 3)


# ---------------------------------------------------------------------------- resultados


def outcomes(alert: pd.Series, session_evals: pd.DataFrame, cost_bps: float) -> dict:
    """Resultados de una alerta con el gap anclado en t0."""
    out = {}
    pos = session_evals.index.get_loc(alert["ts"])
    g0 = alert["gap"]
    sign = np.sign(g0)
    for k in HORIZONS:
        after = session_evals.iloc[pos + 1 : pos + 1 + k]
        if len(after) < k:
            out.update({f"converged_{k}": np.nan, f"ret_conv_{k}": np.nan, f"target_share_{k}": np.nan, f"net_bps_{k}": np.nan})
            continue
        c_k = g0 + after["resid"].sum()
        closure_target = -sign * after["y"].sum()
        closure_implied = sign * after["implied"].sum()
        closure = closure_target + closure_implied
        out[f"converged_{k}"] = float(abs(c_k) <= 0.5 * abs(g0))
        out[f"ret_conv_{k}"] = float(closure_target)
        out[f"target_share_{k}"] = float(closure_target / closure) if closure > 0 else np.nan
        out[f"net_bps_{k}"] = float(closure_target * 1e4 - cost_bps)
    return out


def cost_for(bars_target: pd.DataFrame | None, ts: pd.Timestamp, commission_bps: float) -> float:
    """Coste de ida y vuelta en pb: spread mediano de la franja de 30 min (Dukascopy) + comisión."""
    if bars_target is None or bars_target.empty:
        return commission_bps
    b = bars_target
    same_slot = b[(b.index.hour == ts.hour) & ((b.index.minute // 30) == (ts.minute // 30))]
    same_slot = same_slot[same_slot.index < ts].tail(20 * 6)
    spread_bps = (same_slot["spread"] / same_slot["close"]).median() * 1e4 if not same_slot.empty else np.nan
    # Histórico descargado solo con bid: sin spread, el coste es solo la comisión.
    return commission_bps if not np.isfinite(spread_bps) else float(spread_bps + commission_bps)


# ---------------------------------------------------------------------------- motor


def run_target(
    bars: dict[str, pd.DataFrame],
    universe: Universe,
    target_id: str,
    selection_history: pd.DataFrame,
    events: pd.DataFrame,
    commission_bps: float = 0.0,
    since: pd.Timestamp | None = None,
    prior_r2: pd.Series | None = None,
    z_fixed: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """Alertas, R² intradía y umbral vigente (calibrado con las últimas 250 sesiones).

    Con `since`, solo se evalúan las sesiones >= since; `prior_r2` y `z_fixed`
    aportan el R² histórico y el umbral vigente (modo incremental).
    """
    target = universe.targets[target_id]
    candidates = list(universe.effective_candidates(target_id)[0])
    frame = build_frame(bars, target, universe, candidates)
    if frame.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS), pd.DataFrame(columns=["session", "target", "r2"]), None
    has_market = target.market.type != "none"
    sessions = sorted(frame["session"].unique())
    evals_parts, r2_rows, meta = [], [], {}
    for i, sess in enumerate(sessions):
        if i < FIT_SESSIONS or (since is not None and sess < since):
            continue
        sel = selection_for(selection_history, target_id, sess)
        if sel is None:
            continue
        factors, driver_ok = sel
        factors = [f for f in factors if f in frame]
        try:
            model = IntradayModel.fit(last_sessions(frame, sess, FIT_SESSIONS), target_id, factors, has_market)
        except ValueError:
            continue
        ev = model.evaluate(frame[frame["session"] == sess])
        if ev.empty:
            continue
        evals_parts.append(ev)
        r2_rows.append({"session": sess, "target": target_id, "r2": model.r2})
        meta[sess] = (model.regressors, driver_ok)
    r2 = pd.DataFrame(r2_rows, columns=["session", "target", "r2"])
    if not evals_parts:
        return pd.DataFrame(columns=ALERT_COLUMNS), r2, None
    evals = pd.concat(evals_parts).sort_index()

    blocked = block_mask(evals.index, events, target_id, universe)
    all_r2 = pd.concat([prior_r2.rename("r2") if prior_r2 is not None else pd.Series(dtype=float), r2.set_index("session")["r2"]])
    all_r2 = all_r2[~all_r2.index.duplicated(keep="last")].sort_index()
    r2_median = all_r2.shift(1).rolling(LOOKBACK_SESSIONS, min_periods=MIN_HISTORY_SESSIONS).median()
    r2_ok = (all_r2 > r2_median).reindex(r2["session"]).fillna(False)

    driver = evals["session"].map({s: m[1] for s, m in meta.items()}).astype(bool)
    base_all = (evals["eligible"].astype(bool) & ~blocked & evals["session"].map(r2_ok).fillna(False).astype(bool) & driver).to_numpy()

    # Umbral por mes, calibrado con las sesiones anteriores al mes.
    months = evals["session"].dt.to_period("M")
    rate = TARGET_ALERTS_PER_WEEK / 5.0
    z_by_month = {}
    for month in months.unique():
        if z_fixed is not None:
            z_by_month[month] = z_fixed
            continue
        start = month.to_timestamp()
        prior_sessions = [s for s in evals["session"].unique() if s < start][-LOOKBACK_SESSIONS:]
        mask = evals["session"].isin(prior_sessions).to_numpy()
        z_by_month[month] = calibrate_z(evals[mask], base_all[mask], rate) if mask.any() else Z_DEFAULT

    alerts = []
    bars_target = bars.get(target_id)
    for sess, ev in evals.groupby("session"):
        regressors, driver_ok = meta[sess]
        z_star = z_by_month[pd.Timestamp(sess).to_period("M")]
        a = find_alerts(ev, target_id, z_star, regressors, r2_ok=r2_ok, driver_ok=driver_ok, blocked=blocked.loc[ev.index])
        for _, al in a.iterrows():
            res = outcomes(al, ev, cost_for(bars_target, al["ts"], commission_bps))
            alerts.append({**al.to_dict(), **res, "year": pd.Timestamp(sess).year})
    z_current = None
    if z_fixed is None:
        recent = sorted(evals["session"].unique())[-LOOKBACK_SESSIONS:]
        mask = evals["session"].isin(recent).to_numpy()
        z_current = calibrate_z(evals[mask], base_all[mask], rate)
    return pd.DataFrame(alerts, columns=None if alerts else ALERT_COLUMNS), r2, z_current


# ---------------------------------------------------------------------------- agregación


def aggregate(alerts: pd.DataFrame) -> pd.DataFrame:
    """Métricas por celda con agregación jerárquica y estabilidad por año (§9)."""
    if alerts.empty:
        return pd.DataFrame()
    levels = [
        ("target×factor×origin", ["target", "main_factor", "origin"]),
        ("target×origin", ["target", "origin"]),
        ("origin", ["origin"]),
    ]
    metrics = {f"conv_{k}": (f"converged_{k}", "mean") for k in HORIZONS}
    metrics |= {f"ret_conv_bps_{k}": (f"ret_conv_{k}", lambda s: s.mean() * 1e4) for k in HORIZONS}
    metrics |= {f"target_share_{k}": (f"target_share_{k}", "median") for k in HORIZONS}
    metrics |= {f"net_bps_{k}": (f"net_bps_{k}", "mean") for k in HORIZONS}
    rows = []
    for level, keys in levels:
        g = alerts.groupby(keys, dropna=False)
        agg = g.agg(n=("ts", "size"), **metrics).reset_index()
        yearly = alerts.groupby(keys + ["year"], dropna=False).agg(n=("ts", "size"), net=("net_bps_12", "mean")).reset_index()
        stab = []
        for _, r in agg.iterrows():
            sel = yearly
            for k in keys:
                sel = sel[sel[k] == r[k]] if pd.notna(r[k]) else sel[sel[k].isna()]
            enough = sel[sel["n"] >= MIN_YEAR]
            pos = int((enough["net"] > 0).sum())
            stab.append({"years_ok": len(enough), "years_positive": pos, "persistent": len(enough) > 0 and pos >= 2 * len(enough) / 3})
        agg = pd.concat([agg, pd.DataFrame(stab)], axis=1)
        agg.insert(0, "level", level)
        agg["sufficient"] = agg["n"] >= MIN_CELL
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def cell_for(agg: pd.DataFrame, target: str, factor: str | None, origin: str | None) -> pd.Series | None:
    """Celda que se muestra con una alerta: la más detallada con al menos 30 alertas."""
    if agg is None or agg.empty:
        return None
    for level, cond in (
        ("target×factor×origin", (agg["target"] == target) & (agg["main_factor"] == factor) & (agg["origin"] == origin)),
        ("target×origin", (agg["target"] == target) & (agg["origin"] == origin)),
        ("origin", agg["origin"] == origin),
    ):
        hit = agg[(agg["level"] == level) & cond & agg["sufficient"]]
        if not hit.empty:
            return hit.iloc[0]
    return None


# ---------------------------------------------------------------------------- orquestación


def run_track_record(
    bars: dict[str, pd.DataFrame],
    universe: Universe,
    selection_history: pd.DataFrame,
    events: pd.DataFrame,
    runlog: RunLog,
    commission_bps: dict[str, float] | None = None,
    since: pd.Timestamp | None = None,
    prior_r2: pd.DataFrame | None = None,
    prior_thresholds: dict | None = None,
) -> TrackRecord:
    """Historial completo (since=None) o incremental (since + históricos previos)."""
    all_alerts, all_r2, thresholds = [], [], {}
    for tid in universe.targets:
        prior = None
        if prior_r2 is not None and not prior_r2.empty:
            prior = prior_r2[prior_r2["target"] == tid].set_index("session")["r2"]
        z_fixed = None
        if since is not None:
            z_fixed = ((prior_thresholds or {}).get("targets", {}).get(tid, {}) or {}).get("z_star", Z_DEFAULT)
        comm = (commission_bps or {}).get(tid, 0.0)
        alerts, r2, z_current = run_target(bars, universe, tid, selection_history, events, comm, since, prior, z_fixed)
        all_alerts.append(alerts)
        all_r2.append(r2)
        r2_hist = pd.concat([prior if prior is not None else pd.Series(dtype=float), r2.set_index("session")["r2"] if not r2.empty else pd.Series(dtype=float)])
        r2_hist = r2_hist[~r2_hist.index.duplicated(keep="last")].sort_index()
        # Igual que el filtro del historial: sin 20 sesiones previas no hay mediana (no se alerta).
        entry = {"r2_median": float(r2_hist.tail(LOOKBACK_SESSIONS).median()) if len(r2_hist) >= MIN_HISTORY_SESSIONS else None}
        entry["z_star"] = z_current if z_current is not None else (z_fixed if z_fixed is not None else Z_DEFAULT)
        entry["sessions"] = int(len(r2_hist))
        thresholds[tid] = entry
        if r2.empty:
            runlog.warn(STEP, f"{tid}: sin sesiones evaluables en el historial", target=tid)
    cat = lambda xs, cols=None: pd.concat([x for x in xs if not x.empty], ignore_index=True) if any(not x.empty for x in xs) else pd.DataFrame(columns=cols)
    return TrackRecord(
        cat(all_alerts, ALERT_COLUMNS),
        cat(all_r2, ["session", "target", "r2"]),
        {"target_alerts_per_week": TARGET_ALERTS_PER_WEEK, "z_floor": Z_FLOOR, "targets": thresholds},
    )
