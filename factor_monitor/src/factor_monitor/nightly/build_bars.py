"""Descarga de Dukascopy y mantenimiento del almacén de velas de 5 minutos.

Subcomandos:
  matrix       imprime la matriz JSON (instrumento × año) para `backfill.yml`
  backfill     descarga un instrumento y un año y deja el parquet en --out
  publish      sube al almacén los parquets de --src (un único escritor)
  incremental  días nuevos + los 3 últimos días hábiles, para todos los instrumentos
  quality      informe de calidad por instrumento y mes
  calibrate    compara el .bi5 con dukascopy-node para fijar `point_factor`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.bars import m1_to_bars, merge_bars, quality_report, split_by_year
from ..common.runlog import RunLog
from ..common.universe import Instrument, Universe, load_universe
from ..sources.dukascopy import DownloadError, Pacer, download_m1, fetch_m1_bi5, fetch_m1_node
from .store import BarStore, Backend, GhReleaseBackend, LocalBackend

log = logging.getLogger(__name__)

MAX_MATRIX = 250          # GitHub admite 256 trabajos por matriz
LOOKBACK_BUSINESS_DAYS = 3
STEP = "bars"


def yesterday_utc() -> date:
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def select_instruments(universe: Universe, spec: str, runlog: RunLog | None = None) -> dict[str, Instrument]:
    available = universe.downloadable()
    if runlog is not None:
        for k, inst in universe.instruments.items():
            if not inst.dukascopy_id:
                runlog.warn(STEP, f"{k}: sin dukascopy_id verificado; se excluye de la descarga", instrument=k)
    if spec in ("", "all"):
        return available
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [w for w in wanted if w not in available]
    if unknown:
        raise SystemExit(f"Instrumentos sin dukascopy_id o desconocidos: {unknown}")
    return {w: available[w] for w in wanted}


def build_matrix(instruments: list[str], start_year: int, end_year: int) -> list[dict]:
    """Instrumento × año; si supera el máximo, agrupa años consecutivos."""
    years = list(range(start_year, end_year + 1))
    chunk = 1
    while len(instruments) * -(-len(years) // chunk) > MAX_MATRIX:
        chunk += 1
    out = []
    for inst in instruments:
        for i in range(0, len(years), chunk):
            ys = years[i : i + chunk]
            out.append({"instrument": inst, "years": f"{ys[0]}-{ys[-1]}"})
    return out


@dataclass
class DownloadOptions:
    engine: str = "jetta"
    pace: float = 10.0
    sides: tuple[str, ...] = ("bid", "ask")
    _pacer: Pacer | None = None

    @property
    def pacer(self) -> Pacer:
        # Un único Pacer por proceso: el límite de Dukascopy es por IP, no por instrumento.
        if self._pacer is None:
            self._pacer = Pacer(interval=self.pace)
        return self._pacer


def download_bars(inst: Instrument, start: date, end: date, opts: DownloadOptions | str) -> pd.DataFrame:
    if isinstance(opts, str):
        opts = DownloadOptions(engine=opts)
    m1 = download_m1(
        inst.dukascopy_id, start, end, inst.point_factor, engine=opts.engine,
        code=inst.dukascopy_code, pacer=opts.pacer, sides=opts.sides,
    )  # fmt: skip
    return m1_to_bars(m1, bid_only="ask" not in opts.sides)


# ---------------------------------------------------------------------------- comandos


def cmd_matrix(args, universe: Universe) -> None:
    end_year = args.end_year or yesterday_utc().year
    insts = sorted(select_instruments(universe, args.instruments))
    print(json.dumps({"include": build_matrix(insts, args.start_year, end_year)}))


def cmd_backfill(args, universe: Universe) -> None:
    inst = select_instruments(universe, args.instrument)[args.instrument]
    y0, _, y1 = args.years.partition("-")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runlog = RunLog()
    failed = []
    opts = options_from(args)
    for year in range(int(y0), int(y1 or y0) + 1):
        start = date(year, 1, 1)
        end = min(date(year, 12, 31), yesterday_utc())
        if start > end:
            continue
        try:
            bars = download_bars(inst, start, end, opts)
        except DownloadError as e:
            runlog.warn(STEP, f"{inst.id} {year}: {e}", instrument=inst.id, year=year)
            failed.append(year)
            continue
        if bars.empty:
            runlog.warn(STEP, f"{inst.id} {year}: sin datos", instrument=inst.id, year=year)
            continue
        bars.to_parquet(out / f"{inst.id}_{year}.parquet")
        log.info("%s %s: %d velas", inst.id, year, len(bars))
    if opts.engine == "jetta":
        log.info("Peticiones: %d; límites (429): %d", opts.pacer.requests, opts.pacer.rate_limited)
        runlog.step(STEP, "ok" if not failed else "failed", requests=opts.pacer.requests, rate_limited=opts.pacer.rate_limited)
    if args.meta:
        runlog.merge_into(Path(args.meta))
    if failed:
        # Un año sin descargar deja el trabajo en rojo: nunca un artefacto vacío con éxito.
        log.error("%s: fallaron los años %s", inst.id, failed)
        sys.exit(1)


def cmd_publish(args, universe: Universe, backend: Backend) -> None:
    store = BarStore(backend, Path(args.cache))
    frames = {}
    for path in sorted(Path(args.src).rglob("*.parquet")):
        inst, year = path.stem.rsplit("_", 1)
        new = pd.read_parquet(path)
        old = store.read(inst, int(year))
        frames[(inst, int(year))] = merge_bars(old, new)
    uploaded = store.write(frames)
    log.info("Publicadas %d claves", len(uploaded))


def run_incremental(
    universe: Universe,
    store: BarStore,
    until: date,
    engine: "DownloadOptions | str",
    runlog: RunLog,
    instruments: str = "all",
) -> list[str]:
    """Descarga incremental de todos los instrumentos. Devuelve los que fallaron."""
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    failed = []
    for iid, inst in sorted(select_instruments(universe, instruments, runlog).items()):
        last = store.last_timestamp(iid)
        if last is None:
            runlog.warn(STEP, f"{iid}: sin histórico en el almacén; requiere backfill", instrument=iid)
            continue
        start = (pd.Timestamp(last.date()) - pd.offsets.BDay(LOOKBACK_BUSINESS_DAYS)).date()
        if start > until:
            continue
        try:
            new = download_bars(inst, start, until, engine)
        except DownloadError as e:
            runlog.warn(STEP, f"{iid}: {e}", instrument=iid)
            failed.append(iid)
            continue
        if new.empty:
            runlog.warn(STEP, f"{iid}: sin datos nuevos desde {start}", instrument=iid)
            continue
        # Solo se sustituye desde la primera vela recibida: una descarga incompleta
        # nunca borra datos antiguos que no ha podido reemplazar.
        replace_from = new.index.min()
        for year, part in split_by_year(new).items():
            old = store.read(iid, year)
            frames[(iid, year)] = merge_bars(old, part, replace_from=replace_from)
    store.write(frames)
    runlog.step(STEP, "ok" if not failed else "partial", until=until.isoformat(), updated=len(frames), failed=failed)
    return failed


def cmd_incremental(args, universe: Universe, backend: Backend) -> None:
    runlog = RunLog()
    store = BarStore(backend, Path(args.cache))
    until = date.fromisoformat(args.until) if args.until else yesterday_utc()
    failed = run_incremental(universe, store, until, options_from(args), runlog, args.instruments)
    if args.meta:
        runlog.merge_into(Path(args.meta))
    if failed and args.strict:
        sys.exit(1)


def calibrate_factor(node_m1: pd.DataFrame, raw_m1: pd.DataFrame) -> tuple[float, float]:
    """Factor de escala del .bi5: mediana de entero / precio, redondeada a potencia de 10."""
    joined = node_m1[["bid_close"]].join(raw_m1[["bid_close"]], rsuffix="_raw", how="inner")
    if joined.empty:
        raise DownloadError("Sin minutos comunes entre dukascopy-node y el .bi5")
    ratio = float((joined["bid_close_raw"] / joined["bid_close"]).median())
    return ratio, float(10 ** round(np.log10(ratio)))


def cmd_calibrate(args, universe: Universe) -> None:
    inst = select_instruments(universe, args.instrument)[args.instrument]
    day = date.fromisoformat(args.day)
    node_m1 = fetch_m1_node(inst.dukascopy_id, day, day)
    raw_m1 = fetch_m1_bi5(inst.dukascopy_id, day, day, point_factor=1.0)
    ratio, suggested = calibrate_factor(node_m1, raw_m1)
    ok = inst.point_factor is not None and np.isclose(inst.point_factor, suggested)
    print(json.dumps({"instrument": inst.id, "ratio": ratio, "suggested": suggested, "configured": inst.point_factor, "ok": bool(ok)}))
    if not ok:
        sys.exit(1)


def cmd_quality(args, universe: Universe, backend: Backend) -> None:
    store = BarStore(backend, Path(args.cache))
    reports = []
    for iid in sorted(universe.downloadable()):
        years = [y for _, y in store.keys(iid)]
        if args.years:
            years = years[-args.years :]
        bars = store.read_range(iid, years)
        if bars is not None:
            reports.append(quality_report(bars, iid))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (pd.concat(reports) if reports else pd.DataFrame()).to_csv(out, index=False)


# ---------------------------------------------------------------------------- CLI


def options_from(args) -> DownloadOptions:
    sides = tuple(x.strip() for x in getattr(args, "sides", "bid,ask").split(",") if x.strip())
    return DownloadOptions(engine=args.engine, pace=getattr(args, "pace", 10.0), sides=sides)


def add_download_opts(sp) -> None:
    sp.add_argument("--engine", choices=["jetta", "node", "bi5"], default="jetta")
    sp.add_argument("--pace", type=float, default=10.0, help="Segundos mínimos entre peticiones (motor jetta)")
    sp.add_argument("--sides", default="bid,ask", help="'bid,ask' o 'bid' (la mitad de peticiones, sin spread)")


def make_backend(args) -> Backend:
    if args.store_dir:
        return LocalBackend(Path(args.store_dir))
    return GhReleaseBackend(tag=args.release_tag)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="build_bars")
    p.add_argument("--universe", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    def store_opts(sp):
        sp.add_argument("--store-dir", default=None, help="Almacén local en lugar de la release")
        sp.add_argument("--release-tag", default="data-store")
        sp.add_argument("--cache", default=".cache/store")

    sp = sub.add_parser("matrix")
    sp.add_argument("--instruments", default="all")
    sp.add_argument("--start-year", type=int, default=2017)
    sp.add_argument("--end-year", type=int, default=None)

    sp = sub.add_parser("backfill")
    sp.add_argument("--instrument", required=True)
    sp.add_argument("--years", required=True, help="AAAA o AAAA-AAAA")
    sp.add_argument("--out", required=True)
    add_download_opts(sp)
    sp.add_argument("--meta", default=None)

    sp = sub.add_parser("publish")
    sp.add_argument("--src", required=True)
    store_opts(sp)

    sp = sub.add_parser("incremental")
    sp.add_argument("--instruments", default="all")
    sp.add_argument("--until", default=None)
    add_download_opts(sp)
    sp.add_argument("--meta", default=None)
    sp.add_argument("--strict", action="store_true", help="Código de salida 1 si falla algún instrumento")
    store_opts(sp)

    sp = sub.add_parser("quality")
    sp.add_argument("--out", default="reports/data_quality.csv")
    sp.add_argument("--years", type=int, default=None, help="Solo los últimos N años")
    store_opts(sp)

    sp = sub.add_parser("calibrate")
    sp.add_argument("--instrument", required=True)
    sp.add_argument("--day", required=True, help="Día hábil reciente, AAAA-MM-DD")

    args = p.parse_args(argv)
    universe = load_universe(args.universe) if args.universe else load_universe()
    if args.cmd == "matrix":
        cmd_matrix(args, universe)
    elif args.cmd == "backfill":
        cmd_backfill(args, universe)
    elif args.cmd == "calibrate":
        cmd_calibrate(args, universe)
    else:
        backend = make_backend(args)
        {"publish": cmd_publish, "incremental": cmd_incremental, "quality": cmd_quality}[args.cmd](
            args, universe, backend
        )


if __name__ == "__main__":
    main()
