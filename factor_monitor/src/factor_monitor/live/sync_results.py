"""Sincroniza la rama `results` en una carpeta local (SPEC v3 §10.2).

`git fetch` de la rama y extracción con `git archive` (sin cambiar la rama del clon).
Funciona igual en Windows y Linux porque la descompresión la hace Python.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

BRANCH = "results"


@dataclass
class SyncStatus:
    ok: bool
    data_end: date | None
    stale: bool
    message: str


def last_business_day(today: date) -> date:
    return (pd.Timestamp(today) - pd.offsets.BDay(1)).date()


def is_stale(data_end: date | None, today: date) -> bool:
    """Datos desactualizados si el fin de datos es anterior al último día hábil."""
    return data_end is None or data_end < last_business_day(today)


def read_status(results_dir: Path, today: date) -> SyncStatus:
    meta_path = results_dir / "run_meta.json"
    if not meta_path.exists():
        return SyncStatus(False, None, True, "Sin resultados nocturnos: ejecuta el workflow nightly")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    end = date.fromisoformat(meta["data_end"]) if meta.get("data_end") else None
    stale = is_stale(end, today)
    failed = [k for k, v in meta.get("steps", {}).items() if v.get("status") == "failed"]
    msg = f"Datos hasta {end}" + (" (desactualizados)" if stale else "") + (f"; pasos fallidos: {', '.join(failed)}" if failed else "")
    return SyncStatus(True, end, stale, msg)


def sync(
    repo_dir: Path | str,
    results_dir: Path | str,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    today: date | None = None,
) -> SyncStatus:
    """Actualiza `results_dir` con la rama `results`. Si falla, conserva lo que hubiera."""
    repo_dir, results_dir = Path(repo_dir), Path(results_dir)
    today = today or date.today()
    fetch = run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", f"{BRANCH}:refs/remotes/origin/{BRANCH}"], capture_output=True)
    if fetch.returncode != 0:
        log.warning("No se pudo descargar la rama %s: %s", BRANCH, (fetch.stderr or b"").decode(errors="replace")[:300])
        status = read_status(results_dir, today)
        return SyncStatus(status.ok, status.data_end, True, "Sin conexión con GitHub; se usan los últimos resultados. " + status.message)
    arch = run(["git", "-C", str(repo_dir), "archive", "--format=zip", f"origin/{BRANCH}"], capture_output=True)
    if arch.returncode != 0:
        raise RuntimeError(f"git archive falló: {(arch.stderr or b'').decode(errors='replace')[:300]}")
    tmp = results_dir.with_name(results_dir.name + ".tmp")
    if tmp.exists():
        for p in sorted(tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(arch.stdout)) as z:
        z.extractall(tmp)
    # Sustitución por fichero: si algo falla a mitad, la carpeta anterior sigue siendo usable.
    results_dir.mkdir(parents=True, exist_ok=True)
    for f in tmp.rglob("*"):
        if f.is_file():
            dest = results_dir / f.relative_to(tmp)
            dest.parent.mkdir(parents=True, exist_ok=True)
            f.replace(dest)
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.rmdir() if p.is_dir() else p.unlink()
    tmp.rmdir()
    return read_status(results_dir, today)
