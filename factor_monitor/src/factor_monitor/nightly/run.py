"""Pipeline nocturno (SPEC v3 §10.1).

Pasos: bars → daily → selection → recent_bars → run_meta. Cada paso escribe sus
ficheros solo si termina bien; si falla, sus resultados anteriores quedan intactos,
el error se anota en `run_meta.json` y el proceso termina con código 1 (después de
escribir lo demás), para que GitHub avise del fallo.

Ficheros en --results-dir (rama `results`):
  shapley.parquet            date, target, factor, beta, std_beta, t_hac, shapley, share
  daily_r2.parquet           date, target, r2, nobs
  selection.json             selección vigente
  selection_history.parquet  todas las selecciones semanales (para el historial sin look-ahead)
  recent_bars.parquet        últimas sesiones de velas de 5 min (test de consistencia local)
  run_meta.json              estado de cada paso, fecha de fin de datos, versión y avisos
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from ..common.runlog import RunLog
from ..common.universe import Universe, load_universe
from .build_bars import add_download_opts, options_from, run_incremental, yesterday_utc
from .daily_layer import merge_history, run_daily_layer
from .selection import is_week_end, run_selection
from .store import BarStore, GhReleaseBackend, LocalBackend

log = logging.getLogger(__name__)

RECOMPUTE_BUSINESS_DAYS = 3
RECENT_BUSINESS_DAYS = 15
STEPS = ("bars", "daily", "selection", "recent_bars")


# ---------------------------------------------------------------------------- E/S


def read_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def load_bars(store: BarStore, instruments: list[str], years: list[int]) -> dict[str, pd.DataFrame]:
    out = {}
    for iid in instruments:
        df = store.read_range(iid, years)
        if df is not None and not df.empty:
            out[iid] = df
    return out


# ---------------------------------------------------------------------------- pasos


def step_daily(results: Path, store: BarStore, universe: Universe, runlog: RunLog, years: int, full: bool) -> None:
    shapley_path, r2_path = results / "shapley.parquet", results / "daily_r2.parquet"
    old_f, old_r2 = read_parquet(shapley_path), read_parquet(r2_path)
    since = None
    if not full and old_r2 is not None and not old_r2.empty:
        since = pd.Timestamp(old_r2["date"].max()) - pd.offsets.BDay(RECOMPUTE_BUSINESS_DAYS)
    this_year = yesterday_utc().year
    bars = load_bars(store, sorted(universe.downloadable()), list(range(this_year - years + 1, this_year + 1)))
    if not bars:
        raise RuntimeError("El almacén no tiene velas; ejecutar antes el backfill")
    res = run_daily_layer(bars, universe, runlog, since=since)
    if res.r2.empty:
        raise RuntimeError("La capa diaria no produjo ninguna estimación")
    write_parquet(merge_history(old_f, res.factors, ["date", "target", "factor"]), shapley_path)
    write_parquet(merge_history(old_r2, res.r2, ["date", "target"]), r2_path)
    runlog.step("daily", "ok", since=None if since is None else since.date().isoformat(), rows=len(res.r2))


def step_selection(results: Path, universe: Universe, runlog: RunLog, force: bool) -> None:
    factors, r2 = read_parquet(results / "shapley.parquet"), read_parquet(results / "daily_r2.parquet")
    if factors is None or r2 is None or r2.empty:
        raise RuntimeError("Sin resultados de la capa diaria")
    data_end = pd.Timestamp(r2["date"].max()).date()
    sel_path = results / "selection.json"
    current = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else None
    if current is not None and current.get("as_of") == data_end.isoformat() and not force:
        runlog.step("selection", "skipped", reason="ya calculada con estos datos", as_of=data_end.isoformat())
        return
    if current is not None and not force and not is_week_end(data_end):
        runlog.step("selection", "skipped", reason="no es fin de semana hábil", as_of=current.get("as_of"))
        return
    sel, rows = run_selection(factors, universe, data_end)
    write_json(sel, sel_path)
    hist_path = results / "selection_history.parquet"
    write_parquet(merge_history(read_parquet(hist_path), rows, ["as_of", "target", "factor"]), hist_path)
    none = [t for t, v in sel["targets"].items() if v["status"] != "ok"]
    runlog.step("selection", "ok", as_of=sel["as_of"], valid_from=sel["valid_from"], without_driver=none)


def step_recent_bars(results: Path, store: BarStore, universe: Universe, runlog: RunLog) -> None:
    end = yesterday_utc()
    start = (pd.Timestamp(end) - pd.offsets.BDay(RECENT_BUSINESS_DAYS)).tz_localize("UTC")
    bars = load_bars(store, sorted(universe.downloadable()), sorted({start.year, end.year}))
    frames = []
    for iid, df in bars.items():
        part = df[df.index >= start].reset_index()
        part.insert(0, "instrument", iid)
        frames.append(part)
    if not frames:
        raise RuntimeError("Sin velas recientes")
    write_parquet(pd.concat(frames, ignore_index=True), results / "recent_bars.parquet")
    runlog.step("recent_bars", "ok", since=start.date().isoformat())


def run_step(name: str, fn: Callable[[], None], runlog: RunLog) -> bool:
    try:
        fn()
        if name not in runlog.steps:
            runlog.step(name, "ok")
        return runlog.steps[name]["status"] != "failed"
    except Exception as e:  # noqa: BLE001 - cada paso se aísla
        log.error("Paso %s fallido:\n%s", name, traceback.format_exc())
        runlog.step(name, "failed", error=f"{type(e).__name__}: {e}")
        return False


def write_run_meta(results: Path, runlog: RunLog) -> dict:
    path = results / "run_meta.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    r2 = read_parquet(results / "daily_r2.parquet")
    last_ok = dict(previous.get("last_success", {}))
    for name, info in runlog.steps.items():
        if info["status"] in ("ok", "partial", "skipped"):
            last_ok[name] = info["at"]
    meta = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end": None if r2 is None or r2.empty else pd.Timestamp(r2["date"].max()).date().isoformat(),
        "code_version": os.environ.get("GITHUB_SHA", "local"),
        "steps": runlog.steps,
        "last_success": last_ok,
        "warnings": runlog.warnings,
    }
    write_json(meta, path)
    return meta


def run(args, universe: Universe, store: BarStore | None) -> int:
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    runlog = RunLog()
    ok = True
    if args.skip_download:
        runlog.step("bars", "skipped", reason="--skip-download")
    else:
        until = date.fromisoformat(args.until) if args.until else yesterday_utc()
        ok &= run_step("bars", lambda: run_incremental(universe, store, until, options_from(args), runlog), runlog)
        if runlog.steps.get("bars", {}).get("failed"):
            ok = False  # fallo parcial: se sigue con lo disponible, pero el job debe avisar
    ok &= run_step("daily", lambda: step_daily(results, store, universe, runlog, args.years, args.full), runlog)
    ok &= run_step("selection", lambda: step_selection(results, universe, runlog, args.force_selection), runlog)
    ok &= run_step("recent_bars", lambda: step_recent_bars(results, store, universe, runlog), runlog)
    meta = write_run_meta(results, runlog)
    log.info("Fin de datos: %s; pasos: %s", meta["data_end"], {k: v["status"] for k, v in meta["steps"].items()})
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="nightly")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--universe", default=None)
    p.add_argument("--store-dir", default=None, help="Almacén local en lugar de la release")
    p.add_argument("--release-tag", default="data-store")
    p.add_argument("--cache", default=".cache/store")
    add_download_opts(p)
    p.add_argument("--until", default=None)
    p.add_argument("--years", type=int, default=2, help="Años de velas que lee la capa diaria")
    p.add_argument("--full", action="store_true", help="Recalcula la capa diaria entera con --years años")
    p.add_argument("--force-selection", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args(argv)
    universe = load_universe(args.universe) if args.universe else load_universe()
    backend = LocalBackend(Path(args.store_dir)) if args.store_dir else GhReleaseBackend(tag=args.release_tag)
    store = BarStore(backend, Path(args.cache))
    sys.exit(run(args, universe, store))


if __name__ == "__main__":
    main()
