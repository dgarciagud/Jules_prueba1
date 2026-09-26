"""Descarga de minutos bid/ask de Dukascopy (solo en GitHub Actions).

Motor principal: `dukascopy-node` vía subprocess. Respaldo: lector propio de los
ficheros diarios de velas `.bi5` (SPEC v3 §3.1, §2.4 de la spec de spillovers).
"""

from __future__ import annotations

import logging
import lzma
import os
import subprocess
import tempfile
import time as _time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..common.bars import M1_COLUMNS

log = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
# Registro de vela M1: segundos desde el inicio del día, open, close, low, high (enteros) y volumen.
CANDLE_DTYPE = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])
SIDES = ("bid", "ask")


class DownloadError(RuntimeError):
    pass


# --------------------------------------------------------------------------- .bi5


def candle_url(symbol: str, day: date, side: str) -> str:
    """URL del fichero diario de velas M1. El mes va indexado desde 0."""
    return f"{BASE_URL}/{symbol.upper()}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side.upper()}_candles_min_1.bi5"


def parse_candles_bi5(raw: bytes, day: date, point_factor: float) -> pd.DataFrame:
    """Decodifica un fichero de velas M1. Un fichero vacío es un día sin datos.

    Se descartan las velas planas de volumen 0 (minutos sin ticks que Dukascopy rellena).
    """
    cols = ["open", "high", "low", "close", "volume"]
    if not raw:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
    data = lzma.decompress(raw)
    if len(data) % CANDLE_DTYPE.itemsize:
        raise DownloadError(f"Tamaño de fichero .bi5 no múltiplo de {CANDLE_DTYPE.itemsize} bytes")
    arr = np.frombuffer(data, dtype=CANDLE_DTYPE)
    ts = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(arr["t"].astype(np.int64), unit="s")
    df = pd.DataFrame(
        {
            "open": arr["o"] / point_factor,
            "high": arr["h"] / point_factor,
            "low": arr["l"] / point_factor,
            "close": arr["c"] / point_factor,
            "volume": arr["v"].astype(float),
        },
        index=pd.DatetimeIndex(ts, name="ts"),
    )
    return df[df["volume"] > 0]


def http_get(
    url: str,
    retries: int = 6,
    pause: float = 2.0,
    timeout: float = 30.0,
    rate_limit_pause: float = 15.0,
    max_wait: float = 300.0,
    sleep: Callable[[float], None] = _time.sleep,
) -> bytes:
    """GET con backoff exponencial. Un 404 es un día sin datos (bytes vacíos).

    Ante un 429/503 (límite de peticiones de Dukascopy) la espera parte de
    `rate_limit_pause` y respeta `Retry-After` si viene en la respuesta.
    """
    for attempt in range(retries):
        wait = min(pause * 2**attempt, max_wait)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            err: Exception = e
            if e.code in (429, 503):
                retry_after = e.headers.get("Retry-After") if e.headers else None
                base = float(retry_after) if retry_after and retry_after.isdigit() else rate_limit_pause * 2**attempt
                wait = min(base, max_wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        if attempt < retries - 1:
            log.warning("Fallo al descargar %s (%s); reintento en %.0f s", url, err, wait)
            sleep(wait)
    raise DownloadError(f"No se pudo descargar {url} tras {retries} intentos ({err})")


def fetch_m1_bi5(
    symbol: str,
    start: date,
    end: date,
    point_factor: float,
    getter: Callable[[str], bytes] = http_get,
    pause: float = 0.5,
) -> pd.DataFrame:
    """Minutos bid/ask de `start` a `end` (incluidos) con el lector .bi5.

    `pause` separa las peticiones de cada día para no activar el límite de Dukascopy.
    """
    frames = {side: [] for side in SIDES}
    day = start
    while day <= end:
        if day.weekday() != 5:  # los sábados no hay datos
            for side in SIDES:
                frames[side].append(parse_candles_bi5(getter(candle_url(symbol, day, side)), day, point_factor))
            if pause:
                _time.sleep(pause)
        day += timedelta(days=1)
    return _join_sides({s: pd.concat(f) if f else None for s, f in frames.items()})


# --------------------------------------------------------------------------- dukascopy-node


def node_command(symbol: str, start: date, end: date, side: str, directory: Path, file_name: str) -> list[str]:
    """Comando de dukascopy-node. `--date-to` se pasa como el día siguiente a `end`."""
    return [
        "npx", "--no-install", "dukascopy-node",
        "--instrument", symbol,
        "--date-from", start.isoformat(),
        "--date-to", (end + timedelta(days=1)).isoformat(),
        "--timeframe", "m1",
        "--price-type", side,
        "--format", "csv",
        "--volumes",
        "--directory", str(directory),
        "--file-name", file_name,
        "--batch-size", "5",
        "--batch-pause", "2000",
        "--retries", "8",
        "--retry-pause", "5000",
        "--silent",
    ]  # fmt: skip


def parse_node_csv(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    ts_col = df.columns[0]
    ts = df[ts_col]
    if np.issubdtype(ts.dtype, np.number):
        idx = pd.to_datetime(ts.astype("int64"), unit="ms", utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)
    df.index = pd.DatetimeIndex(idx, name="ts")
    return df[["open", "high", "low", "close"]].astype(float)


def node_env() -> dict[str, str]:
    """Entorno para dukascopy-node. El `fetch` de Node no lee HTTPS_PROXY salvo con NODE_USE_ENV_PROXY."""
    env = dict(os.environ)
    if any(env.get(k) for k in ("HTTPS_PROXY", "https_proxy")):
        env.setdefault("NODE_USE_ENV_PROXY", "1")
    return env


def fetch_m1_node(
    symbol: str,
    start: date,
    end: date,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    cwd: Path | None = None,
    rate_limit_retries: int = 3,
    rate_limit_pause: float = 60.0,
    sleep: Callable[[float], None] = _time.sleep,
) -> pd.DataFrame:
    """Minutos bid/ask con dukascopy-node.

    Si falla por límite de peticiones (429), espera `rate_limit_pause` · 2^k y reintenta.
    Cualquier otro fallo se propaga de inmediato.
    """
    frames = {}
    with tempfile.TemporaryDirectory() as tmp:
        for side in SIDES:
            name = f"{symbol}_{side}_{start:%Y%m%d}_{end:%Y%m%d}"
            cmd = node_command(symbol, start, end, side, Path(tmp), name)
            for attempt in range(rate_limit_retries + 1):
                proc = runner(cmd, cwd=cwd, capture_output=True, text=True, env=node_env())
                if proc.returncode == 0:
                    break
                # Node escribe avisos en stderr y el error de la descarga en stdout: se miran los dos.
                detail = " | ".join(x.strip() for x in (proc.stdout, proc.stderr) if x and x.strip())[-500:]
                if "429" not in detail or attempt == rate_limit_retries:
                    raise DownloadError(f"dukascopy-node falló ({proc.returncode}): {detail}")
                wait = rate_limit_pause * 2**attempt
                log.warning("dukascopy-node: límite de peticiones para %s; reintento en %.0f s", symbol, wait)
                sleep(wait)
            path = Path(tmp) / f"{name}.csv"
            if not path.exists():
                raise DownloadError(f"dukascopy-node no generó {path.name}")
            df = parse_node_csv(path)
            lo = pd.Timestamp(start, tz="UTC")
            hi = pd.Timestamp(end + timedelta(days=1), tz="UTC")
            frames[side] = df[(df.index >= lo) & (df.index < hi)] if len(df) else df
    return _join_sides(frames)


# --------------------------------------------------------------------------- común


def _join_sides(frames: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    parts = []
    for side in SIDES:
        df = frames.get(side)
        if df is None or df.empty:
            return pd.DataFrame(columns=M1_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
        df = df[["open", "high", "low", "close"]].add_prefix(f"{side}_")
        parts.append(df[~df.index.duplicated(keep="last")])
    out = parts[0].join(parts[1], how="inner").sort_index()
    out.index.name = "ts"
    return out[M1_COLUMNS]


def download_m1(
    symbol: str,
    start: date,
    end: date,
    point_factor: float | None,
    engine: str = "node",
    fallback: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Minutos bid/ask. Con `engine="node"` y `fallback`, si dukascopy-node falla se usa el lector .bi5."""
    if engine not in ("node", "bi5"):
        raise ValueError(f"Motor desconocido: {engine}")
    if engine == "node":
        try:
            node_kw = {k: v for k, v in kwargs.items() if k in ("runner", "cwd", "sleep", "rate_limit_pause")}
            return fetch_m1_node(symbol, start, end, **node_kw)
        except (DownloadError, FileNotFoundError, OSError) as e:
            if not fallback:
                raise
            log.warning("dukascopy-node falló para %s (%s); se usa el lector .bi5", symbol, e)
    if point_factor is None:
        raise DownloadError(f"{symbol}: falta point_factor para el lector .bi5")
    return fetch_m1_bi5(
        symbol, start, end, point_factor, **{k: v for k, v in kwargs.items() if k in ("getter", "pause")}
    )
