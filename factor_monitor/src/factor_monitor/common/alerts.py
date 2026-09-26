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
    blocked = blocked.reindex(evals.index, fill_value=False) if blocked is not None else pd.Series(False, index=evals.index)
    if isinstance(r2_ok, pd.Series):
        r2_mask = evals["session"].map(r2_ok).fillna(False).astype(bool)
    else:
        r2_mask = pd.Series(bool(r2_ok), index=evals.index)
    base = evals["eligible"].astype(bool) & ~blocked.astype(bool) & r2_mask

    rows = []
    for _, sess in evals.groupby("session", sort=True):
        last_ts, last_L, crossed = None, None, True
        prev_sign = None
        for ts, row in sess.iterrows():
            if last_ts is not None and not crossed:
                z_cur = row.get(f"z_{last_L}")
                if z_cur is not None and np.isfinite(z_cur) and prev_sign is not None and np.sign(z_cur) != prev_sign:
                    crossed = True
            if not base.loc[ts]:
                continue
            if last_ts is not None:
                if not crossed or ts - last_ts < pd.Timedelta(minutes=cooldown_minutes):
                    continue
            cands = {L: row.get(f"z_{L}") for L in LENGTHS}
            cands = {L: z for L, z in cands.items() if z is not None and np.isfinite(z) and abs(z) >= z_star}
            if not cands:
                continue
            L = max(cands, key=lambda k: abs(cands[k]))
            z = float(cands[L])
            rows.append(
                {
                    "ts": ts, "session": row["session"], "target": target, "L": L, "z": z, "z_star": z_star,
                    "gap": float(row[f"gap_{L}"]), "main_factor": main_factor(row, L, regressors),
                    "origin": row[f"origin_{L}"], "implied": float(row[f"implied_{L}"]), "real": float(row[f"real_{L}"]),
                }
            )  # fmt: skip
            last_ts, last_L, crossed, prev_sign = ts, L, False, np.sign(z)
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)
