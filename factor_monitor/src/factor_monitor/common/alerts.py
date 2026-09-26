"""Reglas de alerta (SPEC v3 §8). Función pura sobre la evaluación intradía.

Una vela genera alerta si se cumplen todas:
  - |z_L| >= z* con L = 6 o L = 12 (si ambas, la de mayor |z|);
  - R² intradía por encima de su mediana (`r2_ok`);
  - el objetivo tiene driver macro (`driver_ok`);
  - la vela no está en una ventana de bloqueo de eventos;
  - no es la primera media hora de sesión (`eligible`).
Enfriamiento: tras una alerta, no hay otra hasta que se cumpla lo que ocurra más
tarde de: 30 minutos transcurridos, o z (de la L que alertó) haya cruzado 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .intraday import LENGTHS, main_factor

COOLDOWN_MINUTES = 30

ALERT_COLUMNS = [
    "ts", "session", "target", "L", "z", "z_star", "gap", "main_factor", "origin", "implied", "real",
]  # fmt: skip


def alert_positions(
    z: np.ndarray,
    base: np.ndarray,
    sessions: np.ndarray,
    ts_ns: np.ndarray,
    z_star: float,
    cooldown_minutes: int = COOLDOWN_MINUTES,
) -> list[tuple[int, int]]:
    """Núcleo de las reglas sobre arrays. Devuelve (posición, índice de L) de cada alerta.

    `z`: matriz (velas × longitudes); `base`: condiciones distintas de z ya combinadas.
    """
    out = []
    cooldown = cooldown_minutes * 60 * 1_000_000_000
    n = len(base)
    last_t, last_j, prev_sign, crossed, cur_sess = None, 0, 0.0, True, None
    absz = np.abs(z)
    for i in range(n):
        if sessions[i] != cur_sess:
            cur_sess, last_t, crossed = sessions[i], None, True
        if last_t is not None and not crossed:
            zc = z[i, last_j]
            if zc == zc and np.sign(zc) != prev_sign:  # zc == zc descarta NaN
                crossed = True
        if not base[i]:
            continue
        if last_t is not None and (not crossed or ts_ns[i] - last_t < cooldown):
            continue
        row = absz[i]
        valid = row >= z_star  # NaN da False
        if not valid.any():
            continue
        j = int(np.nanargmax(np.where(valid, row, -np.inf)))
        out.append((i, j))
        last_t, last_j, crossed, prev_sign = ts_ns[i], j, False, np.sign(z[i, j])
    return out


def find_alerts(
    evals: pd.DataFrame,
    target: str,
    z_star: float,
    regressors: list[str],
    r2_ok: bool | pd.Series = True,
    driver_ok: bool = True,
    blocked: pd.Series | None = None,
    cooldown_minutes: int = COOLDOWN_MINUTES,
) -> pd.DataFrame:
    """Alertas de un objetivo sobre las velas evaluadas (una o varias sesiones).

    `r2_ok` puede ser un booleano o una Serie indexada por sesión.
    """
    if evals.empty or not driver_ok:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    evals = evals.sort_index()
    blocked = blocked.reindex(evals.index, fill_value=False) if blocked is not None else pd.Series(False, index=evals.index)
    if isinstance(r2_ok, pd.Series):
        r2_mask = evals["session"].map(r2_ok).fillna(False).astype(bool)
    else:
        r2_mask = pd.Series(bool(r2_ok), index=evals.index)
    base = (evals["eligible"].astype(bool) & ~blocked.astype(bool) & r2_mask).to_numpy()
    z = np.column_stack([evals[f"z_{L}"].to_numpy(float) if f"z_{L}" in evals else np.full(len(evals), np.nan) for L in LENGTHS])
    pos = alert_positions(z, base, evals["session"].to_numpy(), evals.index.as_unit("ns").asi8, z_star, cooldown_minutes)
    rows = []
    for i, j in pos:
        L = LENGTHS[j]
        row = evals.iloc[i]
        rows.append(
            {
                "ts": evals.index[i], "session": row["session"], "target": target, "L": L, "z": float(z[i, j]),
                "z_star": z_star, "gap": float(row[f"gap_{L}"]), "main_factor": main_factor(row, L, regressors),
                "origin": row[f"origin_{L}"], "implied": float(row[f"implied_{L}"]), "real": float(row[f"real_{L}"]),
            }
        )  # fmt: skip
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)
