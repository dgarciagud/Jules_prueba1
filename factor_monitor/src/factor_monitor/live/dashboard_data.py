"""Lectura y preparación de datos para el dashboard (sin Streamlit, con tests).

El dashboard solo LEE: lo que escribe el runner (`state/`) y lo que publica el
nocturno (`results_local/`). Todas las funciones toleran ficheros ausentes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.intraday import LENGTHS
from ..nightly.track_record import cell_for

# Color fijo por factor (sigue a la entidad, nunca al orden): paleta de referencia
# validada en modo claro y oscuro, en el orden de sus slots.
FACTOR_ORDER = ["MKT", "BRENT", "BUND", "TBOND", "USD", "IA"]
FACTOR_COLORS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
FACTOR_COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
FACTOR_LABELS = {"MKT": "Mercado", "BRENT": "Petróleo", "BUND": "Bund", "TBOND": "T-Bond", "USD": "Dólar", "IA": "IA"}
ACTIVE_MINUTES = 30


@dataclass
class Sources:
    results: Path
    state: Path

    def read_json(self, base: Path, name: str) -> dict | None:
        p = base / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def read_parquet(self, base: Path, name: str) -> pd.DataFrame | None:
        p = base / name
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception:  # noqa: BLE001 - un fichero a medio escribir no debe tumbar la app
            return None

    def read_csv(self, base: Path, name: str) -> pd.DataFrame | None:
        p = base / name
        return pd.read_csv(p) if p.exists() else None

    # --- nocturno
    def run_meta(self): return self.read_json(self.results, "run_meta.json")
    def selection(self): return self.read_json(self.results, "selection.json")
    def thresholds(self): return self.read_json(self.results, "thresholds.json")
    def shapley(self): return self.read_parquet(self.results, "shapley.parquet")
    def daily_r2(self): return self.read_parquet(self.results, "daily_r2.parquet")
    def track_record(self): return self.read_parquet(self.results, "track_record.parquet")

    # --- runner
    def status(self): return self.read_json(self.state, "status.json")
    def alerts(self): return self.read_parquet(self.state, "alerts_live.parquet")
    def consistency(self): return self.read_csv(self.state, "consistency.csv")
    def inventory(self): return self.read_csv(self.state, "mt5_inventory.csv")

    def live_eval(self, target: str) -> pd.DataFrame | None:
        return self.read_parquet(self.state, f"live_{target}.parquet")


# ---------------------------------------------------------------------------- régimen


def latest_shares(shapley: pd.DataFrame | None) -> pd.DataFrame:
    """Última cuota Shapley por objetivo y factor (sin el mercado) para el heatmap."""
    if shapley is None or shapley.empty:
        return pd.DataFrame(columns=["target", "factor", "share", "std_beta", "date"])
    last = shapley[shapley["date"] == shapley.groupby("target")["date"].transform("max")]
    return last[last["factor"] != "MKT"][["target", "factor", "share", "std_beta", "date"]].reset_index(drop=True)


def regime_table(shapley: pd.DataFrame | None, r2: pd.DataFrame | None, selection: dict | None) -> pd.DataFrame:
    """Una fila por objetivo: R², factores seleccionados, estado y cuotas."""
    shares = latest_shares(shapley)
    rows = []
    targets = sorted(set(shares["target"]) | set((selection or {}).get("targets", {})))
    for t in targets:
        sel = (selection or {}).get("targets", {}).get(t, {})
        r2_last = None
        if r2 is not None and not r2.empty and (r2["target"] == t).any():
            r2_last = float(r2[r2["target"] == t].sort_values("date")["r2"].iloc[-1])
        sh = shares[shares["target"] == t].set_index("factor")["share"]
        rows.append(
            {
                "objetivo": t,
                "R²": r2_last,
                "seleccionados": ", ".join(FACTOR_LABELS.get(f, f) for f in sel.get("factors", [])) or "—",
                "estado": {"ok": "con driver", "sin_driver_macro": "sin driver macro"}.get(sel.get("status"), sel.get("status", "—")),
                **{FACTOR_LABELS.get(f, f): float(sh[f]) if f in sh else np.nan for f in FACTOR_ORDER if f != "MKT"},
            }
        )
    return pd.DataFrame(rows)


def shares_history(shapley: pd.DataFrame | None, target: str) -> pd.DataFrame:
    """Cuotas Shapley en el tiempo de un objetivo (incluido el mercado, como parte del R²)."""
    if shapley is None or shapley.empty:
        return pd.DataFrame(columns=["date", "factor", "shapley"])
    df = shapley[shapley["target"] == target][["date", "factor", "shapley"]].copy()
    df["factor"] = pd.Categorical(df["factor"], categories=FACTOR_ORDER, ordered=True)
    return df.sort_values(["date", "factor"])


# ---------------------------------------------------------------------------- monitor


def monitor_table(
    targets: list[str],
    evals: dict[str, pd.DataFrame | None],
    status: dict | None,
    alerts: pd.DataFrame | None,
    track: pd.DataFrame | None,
    inventory: pd.DataFrame | None,
    now: pd.Timestamp,
) -> pd.DataFrame:
    """Tabla del monitor, ordenada por |z| / z* (§11.2)."""
    models = (status or {}).get("models", {})
    excluded = (status or {}).get("excluded", {})
    avail = set(inventory.loc[inventory["available"].astype(bool), "instrument"]) if inventory is not None else None
    rows = []
    for t in targets:
        m = models.get(t, {})
        z_star = m.get("z_star")
        ev = evals.get(t)
        row = {"objetivo": t, "en MT5": "sí" if avail is None or t in avail else "no", "z*": z_star}
        if ev is None or ev.empty:
            row.update({"z": np.nan, "|z|/z*": np.nan, "L": None, "gap (pb)": np.nan, "factor principal": None, "origen": None,
                        "estado": excluded.get(t) or ("sesión cerrada" if t not in (status or {}).get("open_targets", []) else "sin datos")})  # fmt: skip
            rows.append(row)
            continue
        last = ev.iloc[-1]
        zs = {L: last.get(f"z_{L}") for L in LENGTHS}
        zs = {L: z for L, z in zs.items() if z is not None and np.isfinite(z)}
        L = max(zs, key=lambda k: abs(zs[k])) if zs else None
        z = zs.get(L, np.nan) if L else np.nan
        contrib = {c.split("_", 2)[2]: last[c] for c in ev.columns if L and c.startswith(f"contrib_{L}_")}
        main = max(contrib, key=lambda f: abs(contrib[f]) if np.isfinite(contrib[f]) else -1) if contrib else None
        recent = None
        if alerts is not None and not alerts.empty:
            a = alerts[(alerts["target"] == t)]
            a = a[pd.to_datetime(a["ts"], utc=True) >= now - pd.Timedelta(minutes=ACTIVE_MINUTES)]
            recent = a.iloc[-1] if not a.empty else None
        origin = last.get(f"origin_{L}") if L else None
        cell = cell_for(track, t, main, origin)
        row.update(
            {
                "z": float(z) if np.isfinite(z) else np.nan,
                "|z|/z*": abs(z) / z_star if z_star and np.isfinite(z) else np.nan,
                "L": L,
                "gap (pb)": float(last.get(f"gap_{L}", np.nan)) * 1e4 if L else np.nan,
                "factor principal": FACTOR_LABELS.get(main, main),
                "origen": origin,
                "estado": "ALERTA" if recent is not None else ("vigilando" if m.get("r2_ok") and m.get("driver_ok") else "filtrado"),
                "historial": f"{cell['level']}: n={int(cell['n'])}, conv. 60 min {cell['conv_12']:.0%}" if cell is not None else "insuficiente",
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values("|z|/z*", ascending=False, na_position="last").reset_index(drop=True)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].where(out[c].notna(), "—")  # sin "None" en pantalla
    return out


# ---------------------------------------------------------------------------- detalle


def cumulative_paths(ev: pd.DataFrame) -> pd.DataFrame:
    """Acumulado real e implícito desde la apertura, en pb (formato largo)."""
    if ev is None or ev.empty:
        return pd.DataFrame(columns=["ts", "serie", "pb"])
    df = pd.DataFrame(
        {"Real": ev["y"].cumsum() * 1e4, "Implícito": ev["implied"].cumsum() * 1e4}, index=ev.index
    ).reset_index(names="ts")
    return df.melt(id_vars="ts", var_name="serie", value_name="pb")


def decomposition(ev: pd.DataFrame, L: int) -> pd.DataFrame:
    """Aportación de cada factor al implícito acumulado en L (última vela), en pb."""
    if ev is None or ev.empty:
        return pd.DataFrame(columns=["factor", "pb"])
    last = ev.iloc[-1]
    rows = [
        {"factor": c.split("_", 2)[2], "pb": float(last[c]) * 1e4}
        for c in ev.columns
        if c.startswith(f"contrib_{L}_") and np.isfinite(last[c])
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["etiqueta"] = df["factor"].map(lambda f: FACTOR_LABELS.get(f, f))
    return df


def data_status(run_meta: dict | None, status: dict | None, now: pd.Timestamp) -> list[tuple[str, str]]:
    """Avisos del sistema como (nivel, texto) con nivel en {"ok", "warning", "error"}."""
    out = []
    if run_meta is None:
        out.append(("error", "Sin resultados nocturnos: ejecuta el workflow nightly y sincroniza."))
    else:
        failed = [k for k, v in run_meta.get("steps", {}).items() if v.get("status") == "failed"]
        out.append(("error" if failed else "ok", f"Nocturno: datos hasta {run_meta.get('data_end')}" + (f" · pasos fallidos: {', '.join(failed)}" if failed else "")))
    res = (status or {}).get("results") or {}
    if res.get("stale"):
        out.append(("warning", "Datos nocturnos desactualizados: se usa la última selección válida."))
    if status is None:
        out.append(("error", "El runner no está en marcha (sin status.json)."))
    else:
        upd = pd.Timestamp(status["updated_utc"])
        if upd.tzinfo is None:
            upd = upd.tz_localize("UTC")
        if now - upd > pd.Timedelta(minutes=12):
            out.append(("warning", f"El runner no actualiza desde {upd:%H:%M} UTC."))
        src = status.get("offset_source")
        if src == "default":
            out.append(("warning", "Hora del servidor MT5 sin detectar (se asume UTC+0)."))
    return out
