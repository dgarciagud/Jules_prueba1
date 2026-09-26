"""Registro y notificación de las alertas en vivo (SPEC v3 §8.3)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

KEY = ["ts", "target"]


def append_alerts(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Añade al registro las alertas que no estuvieran ya. Devuelve solo las nuevas."""
    if new.empty:
        return new
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=new.columns)
    if not old.empty:
        seen = set(zip(pd.to_datetime(old["ts"], utc=True), old["target"]))
        new = new[[(pd.Timestamp(t), g) not in seen for t, g in zip(new["ts"], new["target"])]]
    if new.empty:
        return new
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new.reset_index(drop=True)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return new


def notify(alert: pd.Series, enabled: bool = True) -> None:
    """Notificación de escritorio (plyer, opcional). Sin plyer, solo queda en el log."""
    text = (
        f"{alert['target']}: z={alert['z']:+.2f} (L={alert['L']}) · {alert['origin']} · "
        f"factor {alert.get('main_factor') or '—'} · gap {alert['gap'] * 1e4:+.1f} pb"
    )
    log.warning("ALERTA %s", text)
    if not enabled:
        return
    try:
        from plyer import notification  # type: ignore[import-not-found]

        notification.notify(title="Monitor de factores", message=text, timeout=10)
    except Exception as e:  # noqa: BLE001 - la notificación nunca debe parar el runner
        log.debug("Sin notificación de escritorio: %s", e)
