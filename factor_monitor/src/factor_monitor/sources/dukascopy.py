"""Descarga de minutos bid/ask de Dukascopy (solo en GitHub Actions).

Motores:
  - `jetta` (por defecto): cliente propio del API JSON `jetta.dukascopy.com`, con
    ritmo fijo entre peticiones. Dukascopy limita a unas 3 peticiones por minuto
    por IP (medido: 1 petición cada 20 s → 43/45 correctas; cada 10 s → 79/80; en
    ráfagas → 429 casi inmediato), así que el ritmo lo marca `Pacer` (10 s por defecto).
  - `node`: `dukascopy-node` vía subprocess (mismo API, pero en ráfagas).
  - `bi5`: lector propio de los ficheros diarios `.bi5` de `datafeed.dukascopy.com`.
"""

from __future__ import annotations

import json
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
JETTA_URL = "https://jetta.dukascopy.com/v1/candles/minute"
# Registro de vela M1: segundos desde el inicio del día, open, close, low, high (enteros) y volumen.
CANDLE_DTYPE = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])
SIDES = ("bid", "ask")


class DownloadError(RuntimeError):
    """Fallo de descarga. `partial` lleva los minutos ya descargados (si los hay)."""

    def __init__(self, message: str, partial: "pd.DataFrame | None" = None):
        super().__init__(message)
        self.partial = partial


class Pacer:
    """Ritmo mínimo entre peticiones y espera larga ante un límite (429)."""

    def __init__(
        self,
        interval: float = 10.0,
        rate_limit_pause: float = 60.0,
        max_rate_limit_retries: int = 3,
        sleep: Callable[[float], None] = _time.sleep,
        clock: Callable[[], float] = _time.monotonic,
    ):
        self.interval = interval
        self.rate_limit_pause = rate_limit_pause
        self.max_rate_limit_retries = max_rate_limit_retries
        self.sleep = sleep
        self.clock = clock
        self._last: float | None = None
        self.requests = 0
        self.rate_limited = 0

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.interval - (self.clock() - self._last)
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()
        self.requests += 1

    def backoff(self, attempt: int) -> None:
        self.rate_limited += 1
        wait = min(self.rate_limit_pause * 2**attempt, 900.0)
        log.warning("Límite de peticiones de Dukascopy; espera de %.0f s", wait)
        self.sleep(wait)


def http_get_paced(url: str, pacer: Pacer, timeout: float = 30.0, retries: int = 4) -> bytes:
    """GET respetando el ritmo. 404 = sin datos. 429/503 = espera larga y reintento."""
    rl_attempt, err_attempt = 0, 0
    while True:
        pacer.wait()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            if e.code in (429, 503):
                if rl_attempt >= pacer.max_rate_limit_retries:
                    raise DownloadError(f"Límite de peticiones persistente en {url}") from e
                pacer.backoff(rl_attempt)
                rl_attempt += 1
                continue
            err: Exception = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        err_attempt += 1
        if err_attempt >= retries:
            raise DownloadError(f"No se pudo descargar {url} ({err})")
        log.warning("Fallo al descargar %s (%s); reintento", url, err)
        pacer.sleep(5.0 * err_attempt)


# --------------------------------------------------------------------------- jetta


def jetta_code(name: str) -> str:
    """Código de jetta a partir del nombre de Dukascopy: "USA500.IDX/USD" → "USA500.IDX-USD"."""
    return name.replace("/", "-")


def jetta_url(code: str, day: date, side: str) -> str:
    """URL de las velas de minuto de un día completo. El mes va indexado desde 1."""
    return f"{JETTA_URL}/{code}/{side.upper()}/{day.year}/{day.month}/{day.day}"


def decode_jetta_candles(raw: bytes) -> pd.DataFrame:
    """Decodifica la respuesta JSON de jetta (deltas enteros sobre una vela base).

    Replica `normaliseCandles` de dukascopy-node: la vela base más los deltas
    acumulados dan cada minuto; los huecos (minutos sin ticks) se omiten.
    """
    cols = ["open", "high", "low", "close", "volume"]
    empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
    if not raw:
        return empty
    data = json.loads(raw)
    times = data.get("times") or []
    if not times:
        return empty
    mult = float(data["multiplier"])
    shift = int(data["shift"])
    base = {k: round(float(data[k]) / mult) for k in ("open", "high", "low", "close")}
    ts = int(data["timestamp"]) + np.cumsum(np.asarray(times, dtype=np.int64)) * shift
    out = {}
    for k, deltas in (("open", "opens"), ("high", "highs"), ("low", "lows"), ("close", "closes")):
        units = base[k] + np.cumsum(np.asarray(data[deltas], dtype=np.int64))
        out[k] = units * mult
    decimals = max(0, -int(np.floor(np.log10(mult)))) if mult < 1 else 0
    df = pd.DataFrame({k: np.round(v, decimals) for k, v in out.items()}, index=pd.DatetimeIndex(pd.to_datetime(ts, unit="ms", utc=True), name="ts"))
    df["volume"] = np.asarray(data.get("volumes", [np.nan] * len(times)), dtype=float)
    return df


def fetch_m1_jetta(
    code: str,
    start: date,
    end: date,
    pacer: Pacer | None = None,
    sides: tuple[str, ...] = SIDES,
    getter: Callable[[str], bytes] | None = None,
    skip_days: set[date] | None = None,
) -> pd.DataFrame:
    """Minutos de `start` a `end` (incluidos), un día por petición, sin sábados.

    Con `sides=("bid",)` la mitad de peticiones: el ask se deja igual al bid y el
    spread queda en NaN. `skip_days` evita volver a pedir días ya guardados.
    Si la descarga se corta, el DownloadError lleva en `partial` lo descargado.
    """
    pacer = pacer or Pacer()
    get = getter or (lambda url: http_get_paced(url, pacer))
    frames = {side: [] for side in sides}

    def assemble() -> pd.DataFrame:
        n = min(len(f) for f in frames.values()) if frames else 0
        joined = {s: pd.concat(f[:n]) if n else None for s, f in frames.items()}
        if "ask" not in sides:
            joined["ask"] = joined.get("bid")
        return _join_sides(joined, allow_same=("ask" not in sides))

    day = start
    while day <= end:
        if day.weekday() != 5 and not (skip_days and day in skip_days):
            try:
                got = {side: decode_jetta_candles(get(jetta_url(code, day, side))) for side in sides}
            except DownloadError as e:
                raise DownloadError(str(e), partial=assemble()) from e
            for side in sides:
                frames[side].append(got[side])
        day += timedelta(days=1)
    return assemble()


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


def _join_sides(frames: dict[str, pd.DataFrame | None], allow_same: bool = False) -> pd.DataFrame:
    """Une bid y ask. Con `allow_same` (solo bid descargado), ask = bid."""
    parts = []
    for side in SIDES:
        df = frames.get(side)
        if df is None or df.empty:
            return pd.DataFrame(columns=M1_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
        df = df[["open", "high", "low", "close"]].add_prefix(f"{side}_")
        parts.append(df[~df.index.duplicated(keep="last")])
    out = parts[0].join(parts[1], how="inner").sort_index()
    out.index.name = "ts"
    out = out[M1_COLUMNS]
    if allow_same:
        out.attrs["bid_only"] = True
    return out


def download_m1(
    symbol: str,
    start: date,
    end: date,
    point_factor: float | None,
    engine: str = "jetta",
    fallback: bool = True,
    code: str | None = None,
    pacer: Pacer | None = None,
    sides: tuple[str, ...] = SIDES,
    **kwargs,
) -> pd.DataFrame:
    """Minutos bid/ask con el motor indicado.

    Con `fallback`, si `node` falla se usa el lector .bi5. `jetta` no tiene respaldo
    automático: el .bi5 sufre el mismo límite y solo alargaría el fallo.
    """
    if engine not in ("jetta", "node", "bi5"):
        raise ValueError(f"Motor desconocido: {engine}")
    if engine == "jetta":
        if code is None:
            raise DownloadError(f"{symbol}: falta el código de jetta (dukascopy_code)")
        return fetch_m1_jetta(code, start, end, pacer=pacer, sides=sides, getter=kwargs.get("getter"), skip_days=kwargs.get("skip_days"))
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
