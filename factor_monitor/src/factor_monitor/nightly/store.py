"""Almacén de velas de 5 minutos en assets de una GitHub Release (SPEC v3 §3.1).

Cada clave (instrumento, año) se guarda como `{instrumento}_{año}.{versión}.parquet`.
`manifest.json` indica qué versión es la vigente de cada clave.

Orden de escritura para que un fallo a mitad nunca deje el almacén inconsistente:
  1. subir los ficheros nuevos con nombre versionado (no pisan nada);
  2. subir el manifiesto nuevo;
  3. borrar las versiones antiguas que ya no referencia el manifiesto.
Si falla 1, el manifiesto anterior sigue apuntando a ficheros que existen.
Si el manifiesto se pierde, se reconstruye con la versión más reciente de cada clave.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
ASSET_RE = re.compile(r"^(?P<inst>[A-Za-z0-9]+)_(?P<year>\d{4})\.(?P<version>\d{8}T\d{6}Z-[0-9a-f]{8})\.parquet$")


class Backend(Protocol):
    def list_assets(self) -> list[str]: ...
    def download(self, name: str, dest: Path) -> Path: ...
    def upload(self, path: Path, name: str) -> None: ...
    def delete(self, name: str) -> None: ...


class LocalBackend:
    """Directorio local (tests y ejecuciones sin GitHub)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_assets(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file())

    def download(self, name: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy2(self.root / name, dest / name))

    def upload(self, path: Path, name: str) -> None:
        shutil.copy2(path, self.root / name)

    def delete(self, name: str) -> None:
        (self.root / name).unlink(missing_ok=True)


class GhReleaseBackend:
    """Assets de una release de GitHub mediante la CLI `gh` (preinstalada en los runners)."""

    def __init__(self, tag: str = "data-store", repo: str | None = None):
        self.tag = tag
        self.repo = repo
        self._ensure_release()

    def _gh(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["gh", *args] + (["--repo", self.repo] if self.repo else [])
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    def _ensure_release(self) -> None:
        if self._gh("release", "view", self.tag, check=False).returncode == 0:
            return
        created = self._gh(
            "release", "create", self.tag,
            "--title", "data-store",
            "--notes", "Almacén de velas de 5 minutos. No borrar.",
            "--prerelease",
            check=False,
        )  # fmt: skip
        # Varios trabajos en paralelo pueden intentar crearla a la vez: basta con que exista.
        if created.returncode != 0 and self._gh("release", "view", self.tag, check=False).returncode != 0:
            raise RuntimeError(f"No se pudo crear la release {self.tag}: {created.stderr.strip()[:300]}")

    def list_assets(self) -> list[str]:
        out = self._gh("release", "view", self.tag, "--json", "assets").stdout
        return sorted(a["name"] for a in json.loads(out)["assets"])

    def download(self, name: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        self._gh("release", "download", self.tag, "--pattern", name, "--dir", str(dest), "--clobber")
        return dest / name

    def upload(self, path: Path, name: str) -> None:
        staged = path if path.name == name else Path(shutil.copy2(path, path.with_name(name)))
        self._gh("release", "upload", self.tag, str(staged), "--clobber")

    def delete(self, name: str) -> None:
        self._gh("release", "delete-asset", self.tag, name, "--yes")


@dataclass
class Entry:
    asset: str
    rows: int
    start: str | None
    end: str | None
    sha256: str


class BarStore:
    def __init__(self, backend: Backend, cache_dir: Path):
        self.backend = backend
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, Entry] = self._load_manifest()

    @staticmethod
    def key(instrument: str, year: int) -> str:
        return f"{instrument}_{year}"

    # ------------------------------------------------------------------ manifiesto

    def _load_manifest(self) -> dict[str, Entry]:
        assets = self.backend.list_assets()
        if MANIFEST in assets:
            path = self.backend.download(MANIFEST, self.cache)
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {k: Entry(**v) for k, v in raw["entries"].items()}
        if any(ASSET_RE.match(a) for a in assets):
            log.warning("manifest.json no encontrado; se reconstruye con la última versión de cada clave")
        return self._rebuild(assets)

    def _rebuild(self, assets: list[str]) -> dict[str, Entry]:
        latest: dict[str, str] = {}
        for a in assets:
            m = ASSET_RE.match(a)
            if m:
                k = self.key(m["inst"], int(m["year"]))
                if k not in latest or a > latest[k]:
                    latest[k] = a
        return {k: Entry(asset=a, rows=-1, start=None, end=None, sha256="") for k, a in latest.items()}

    def _write_manifest(self, entries: dict[str, Entry]) -> None:
        path = self.cache / MANIFEST
        payload = {
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entries": {k: vars(v) for k, v in sorted(entries.items())},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.backend.upload(path, MANIFEST)

    # ------------------------------------------------------------------ lectura

    def keys(self, instrument: str | None = None) -> list[tuple[str, int]]:
        out = []
        for k in self.manifest:
            inst, year = k.rsplit("_", 1)
            if instrument is None or inst == instrument:
                out.append((inst, int(year)))
        return sorted(out)

    def read(self, instrument: str, year: int) -> pd.DataFrame | None:
        entry = self.manifest.get(self.key(instrument, year))
        if entry is None:
            return None
        local = self.cache / entry.asset
        if not local.exists():
            self.backend.download(entry.asset, self.cache)
        return pd.read_parquet(local)

    def read_range(self, instrument: str, years: list[int]) -> pd.DataFrame | None:
        frames = [f for y in years if (f := self.read(instrument, y)) is not None]
        return pd.concat(frames).sort_index() if frames else None

    def last_timestamp(self, instrument: str) -> pd.Timestamp | None:
        keys = self.keys(instrument)
        for _, year in reversed(keys):
            df = self.read(instrument, year)
            if df is not None and not df.empty:
                return df.index.max()
        return None

    # ------------------------------------------------------------------ escritura

    def write(self, frames: dict[tuple[str, int], pd.DataFrame]) -> list[str]:
        """Escribe varias claves de forma segura. Devuelve los assets subidos."""
        if not frames:
            return []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        new_entries = dict(self.manifest)
        uploaded = []
        for (inst, year), df in sorted(frames.items()):
            tmp = self.cache / f"_tmp_{inst}_{year}.parquet"
            df.to_parquet(tmp)
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            name = f"{inst}_{year}.{stamp}-{digest[:8]}.parquet"
            final = tmp.rename(self.cache / name)
            self.backend.upload(final, name)
            uploaded.append(name)
            new_entries[self.key(inst, year)] = Entry(
                asset=name,
                rows=len(df),
                start=df.index.min().isoformat() if len(df) else None,
                end=df.index.max().isoformat() if len(df) else None,
                sha256=digest,
            )
        self._write_manifest(new_entries)
        self.manifest = new_entries
        self.collect_garbage()
        return uploaded

    def collect_garbage(self) -> list[str]:
        """Borra versiones que el manifiesto vigente ya no referencia."""
        referenced = {e.asset for e in self.manifest.values()}
        removed = []
        for a in self.backend.list_assets():
            if ASSET_RE.match(a) and a not in referenced:
                self.backend.delete(a)
                removed.append(a)
        return removed
