"""Pipeline del informe diario. Paso 1: datos.

    python -m factor_monitor.daily.run --out daily_out [--previous daily_prev]

Escribe en --out:
  prices.parquet          cierres (date, id, close, adjclose, source)
  coverage.csv            una fila por serie: fuente, primer y último dato, avisos
  weights_ref.json        pesos sectoriales de referencia por índice
  sectors.parquet         (date, index, sector, ret, weight) en el calendario de cada índice
  data_meta.json          estado, seguimiento de los pesos (R² frente al índice) y avisos
Paso 2 (`run_analysis`, también con --offline sobre una salida existente):
  attribution.parquet     aportación por periodo (1D…1A) de cada sector y, en el CAC 40, cada acción
  relative_strength.parquet  fuerza relativa de cada sector frente a su índice por horizonte
  contributions_daily.parquet  aportación diaria por sector (últimos dos años)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..common.runlog import RunLog
from .attribution import attribution_table, daily_sector_contributions, relative_strength
from .data import RefWeights, download_all, mark_stale, reference_weights, sector_returns, tracking, wide
from .sources import Http, Stooq, Yahoo
from .universe import DailyUniverse, load_daily_universe

log = logging.getLogger("factor_monitor.daily")


def _read_prev(prev: Path | None, name: str):
    if prev is None or not (prev / name).exists():
        return None
    p = prev / name
    return pd.read_parquet(p) if p.suffix == ".parquet" else json.loads(p.read_text(encoding="utf-8"))


def _write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def index_calendar(close: pd.DataFrame, index_id: str) -> pd.DataFrame:
    """Cierres en las sesiones del índice; huecos cortos (festivos locales de un componente) se arrastran."""
    return close.loc[close[index_id].notna()].ffill(limit=5)


def run_data(universe: DailyUniverse, out: Path, yahoo: Yahoo, stooq: Stooq | None, prev: Path | None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    runlog = RunLog()
    prices, coverage = download_all(universe, yahoo, stooq, _read_prev(prev, "prices.parquet"), runlog)
    coverage = mark_stale(coverage, universe, runlog)
    prices.to_parquet(out / "prices.parquet", index=False)
    coverage.to_csv(out / "coverage.csv", index=False, date_format="%Y-%m-%d")

    close = wide(prices, "close")
    prev_w = _read_prev(prev, "weights_ref.json") or {}
    weights, sectors, track = {}, [], {}
    for iid, ix in universe.indices.items():
        if iid not in close:
            runlog.warn("weights", f"{iid}: sin datos del índice")
            continue
        cal = index_calendar(close, iid)
        ref = reference_weights(ix, cal, yahoo, prev_w.get(iid), runlog)
        if ref is None:
            continue
        weights[iid] = ref.to_json()
        rets, w = sector_returns(ix, cal, ref)
        track[iid] = tracking(cal[iid], rets, w)
        long = rets.stack(future_stack=True).rename("ret").to_frame().join(w.stack(future_stack=True).rename("weight"))
        long = long.reset_index().rename(columns={"level_0": "date", "level_1": "sector"})
        long.columns = ["date", "sector", "ret", "weight"]
        sectors.append(long.dropna(subset=["weight"]).assign(index=iid))
        t = track[iid]
        if t["r2"] is not None and t["r2"] < 0.95:
            runlog.warn("weights", f"{iid}: los sectores reproducen mal el índice (R² {t['r2']:.3f} en {t['n']} sesiones)")
    _write_json(weights, out / "weights_ref.json")
    if sectors:
        pd.concat(sectors, ignore_index=True)[["date", "index", "sector", "ret", "weight"]].to_parquet(out / "sectors.parquet", index=False)

    n_fail = int(coverage["source"].isna().sum())
    runlog.step("data", "ok" if n_fail == 0 else "partial", series=len(coverage), failed=n_fail,
                from_previous=int((coverage["source"] == "anterior").sum()))  # fmt: skip
    ok_analysis = run_analysis(universe, out, runlog, prices=prices, weights=weights)
    idx_last = coverage.loc[coverage["role"] == "index", "last"]
    meta = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end": None if idx_last.dropna().empty else pd.Timestamp(idx_last.max()).date().isoformat(),
        "code_version": os.environ.get("GITHUB_SHA", "local"),
        "steps": runlog.steps,
        "tracking": track,
        "sources": coverage["source"].value_counts(dropna=False).rename(lambda x: x or "fallo").to_dict(),
        "warnings": runlog.warnings,
    }
    _write_json(meta, out / "data_meta.json")
    log.info("Fin de datos: %s; series: %s; seguimiento: %s", meta["data_end"], meta["sources"],
             {k: round(v["r2"], 4) if v.get("r2") is not None else None for k, v in track.items()})  # fmt: skip
    return 0 if n_fail == 0 and ok_analysis else 1


def run_analysis(universe: DailyUniverse, out: Path, runlog: RunLog, prices: pd.DataFrame | None = None,
                 weights: dict | None = None) -> bool:  # fmt: skip
    """Paso 2: atribución sectorial y fuerza relativa a partir de precios y pesos de referencia."""
    prices = prices if prices is not None else pd.read_parquet(out / "prices.parquet")
    weights = weights if weights is not None else json.loads((out / "weights_ref.json").read_text(encoding="utf-8"))
    close = wide(prices, "close")
    names = {s.id: s.name for s in universe.all_series()}
    attr, rs, daily = [], [], []
    try:
        for iid in universe.indices:
            if iid not in weights or iid not in close:
                runlog.warn("attribution", f"{iid}: sin pesos o sin precios; se omite")
                continue
            cal = index_calendar(close, iid)
            ref = RefWeights(**weights[iid])
            attr.append(attribution_table(iid, cal, ref, universe.sector_names, names))
            rs.append(relative_strength(iid, cal, ref, universe.sector_names))
            daily.append(daily_sector_contributions(iid, cal, ref))
        if attr:
            pd.concat(attr, ignore_index=True).to_parquet(out / "attribution.parquet", index=False)
            pd.concat(rs, ignore_index=True).to_parquet(out / "relative_strength.parquet", index=False)
            pd.concat(daily, ignore_index=True).to_parquet(out / "contributions_daily.parquet", index=False)
        runlog.step("attribution", "ok" if attr else "failed", indices=len(attr))
        return bool(attr)
    except Exception as e:  # noqa: BLE001 - el paso 2 no debe tirar la descarga
        log.exception("Atribución fallida")
        runlog.step("attribution", "failed", error=f"{type(e).__name__}: {e}")
        return False


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="factor_monitor.daily.run")
    p.add_argument("--out", default="daily_out")
    p.add_argument("--previous", default=None, help="Salida anterior (respaldo si una serie falla)")
    p.add_argument("--universe", default=None)
    p.add_argument("--no-stooq", action="store_true")
    p.add_argument("--offline", action="store_true", help="No descarga: recalcula el análisis sobre los datos de --out")
    a = p.parse_args(argv)
    universe = load_daily_universe(a.universe)
    if a.offline:
        runlog = RunLog()
        ok = run_analysis(universe, Path(a.out), runlog)
        runlog.merge_into(Path(a.out) / "data_meta.json")
        sys.exit(0 if ok else 1)
    sys.exit(run_data(universe, Path(a.out), Yahoo(Http()), None if a.no_stooq else Stooq(), Path(a.previous) if a.previous else None))


if __name__ == "__main__":
    main()
