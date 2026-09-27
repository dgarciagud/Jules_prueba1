"""Informe HTML nocturno para GitHub Pages (una sola página autocontenida).

Lee los ficheros de la rama `results` y escribe `index.html`: estado del nocturno,
selección de la semana, régimen de factores (heatmap y evolución), historial de
alertas y umbrales. Sin dependencias externas: SVG en línea, CSS con modo claro y
oscuro, tooltips nativos (<title>) y una tabla junto a cada gráfico.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_ORDER = ["MKT", "BRENT", "BUND", "TBOND", "USD", "IA"]
FACTOR_LABELS = {"MKT": "Mercado", "BRENT": "Petróleo", "BUND": "Bund", "TBOND": "T-Bond", "USD": "Dólar", "IA": "IA"}
HISTORY_DAYS = 250

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--surface-2:#f3f2ef;--text:#0b0b0b;--text-2:#52514e;--muted:#8a8985;--rule:#e2e1dc;
--seq:#1c5cab;--good:#0ca30c;--warn:#b27700;--crit:#d03b3b;
--f-MKT:#2a78d6;--f-BRENT:#eb6834;--f-BUND:#1baf7a;--f-TBOND:#eda100;--f-USD:#e87ba4;--f-IA:#008300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--surface-2:#242422;--text:#fff;--text-2:#c3c2b7;
--muted:#8f8e86;--rule:#383835;--seq:#6da7ec;--good:#0ca30c;--warn:#fab219;--crit:#e66767;
--f-MKT:#3987e5;--f-BRENT:#d95926;--f-BUND:#199e70;--f-TBOND:#c98500;--f-USD:#d55181;--f-IA:#008300}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--surface-2:#242422;--text:#fff;--text-2:#c3c2b7;--muted:#8f8e86;--rule:#383835;
--seq:#6da7ec;--good:#0ca30c;--warn:#fab219;--crit:#e66767;
--f-MKT:#3987e5;--f-BRENT:#d95926;--f-BUND:#199e70;--f-TBOND:#c98500;--f-USD:#d55181;--f-IA:#008300}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.15rem;margin:36px 0 10px}
.sub{color:var(--text-2);margin:0 0 16px}.muted{color:var(--muted);font-size:.85rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.card{background:var(--surface-2);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:1.25rem}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.8rem;border:1px solid currentColor}
.ok{color:var(--good)}.warn{color:var(--warn)}.crit{color:var(--crit)}
.tbl{width:100%;border-collapse:collapse;font-size:.9rem}.tbl th,.tbl td{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:left}
.tbl td.n,.tbl th.n{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 10px;font-size:.85rem;color:var(--text-2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
svg{display:block;width:100%;height:auto}svg text{fill:var(--text-2);font-size:11px}svg .ax{stroke:var(--rule)}
details{margin-top:8px}summary{cursor:pointer;color:var(--text-2)}
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def pct(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.0%}"


def num(x, fmt="{:+.2f}") -> str:
    return "—" if x is None or not np.isfinite(x) else fmt.format(x)


def read(results: Path):
    def pq(name):
        p = results / name
        return pd.read_parquet(p) if p.exists() else None

    def js(name):
        p = results / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    return {
        "meta": js("run_meta.json"), "selection": js("selection.json"), "thresholds": js("thresholds.json"),
        "shapley": pq("shapley.parquet"), "r2": pq("daily_r2.parquet"), "track": pq("track_record.parquet"),
        "alerts": pq("track_record_alerts.parquet"),
    }  # fmt: skip


# ---------------------------------------------------------------------------- piezas SVG


def heatmap_svg(shares: pd.DataFrame) -> str:
    """Filas = objetivos, columnas = factores (sin el mercado); opacidad = cuota."""
    factors = [f for f in FACTOR_ORDER if f != "MKT" and f in set(shares["factor"])]
    targets = sorted(shares["target"].unique())
    cw, ch, lw, top = 92, 34, 70, 26
    w, h = lw + cw * len(factors), top + ch * len(targets)
    out = [f'<svg viewBox="0 0 {w} {h}" style="max-width:{w * 1.5:.0f}px" role="img" aria-label="Cuota Shapley por objetivo y factor">']
    for j, f in enumerate(factors):
        out.append(f'<text x="{lw + j * cw + cw / 2}" y="16" text-anchor="middle">{esc(FACTOR_LABELS.get(f, f))}</text>')
    for i, t in enumerate(targets):
        y = top + i * ch
        out.append(f'<text x="{lw - 8}" y="{y + ch / 2 + 4}" text-anchor="end">{esc(t)}</text>')
        for j, f in enumerate(factors):
            row = shares[(shares["target"] == t) & (shares["factor"] == f)]
            x = lw + j * cw
            if row.empty:
                out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="var(--surface-2)"><title>{esc(t)} · {esc(FACTOR_LABELS.get(f, f))}: no candidato</title></rect>')
                continue
            s = float(row["share"].iloc[0]) if np.isfinite(row["share"].iloc[0]) else 0.0
            ink = "#fff" if s > 0.55 else "var(--text)"
            out.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="var(--seq)" fill-opacity="{0.08 + 0.92 * max(0.0, min(1.0, s)):.3f}">'
                f"<title>{esc(t)} · {esc(FACTOR_LABELS.get(f, f))}: cuota {s:.0%}</title></rect>"
                f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" style="fill:{ink}">{s:.0%}</text>'
            )  # fmt: skip
    out.append("</svg>")
    return "".join(out)


def stacked_svg(hist: pd.DataFrame, title: str) -> str:
    """Área apilada de la contribución de cada factor al R² (últimos días)."""
    piv = hist.pivot_table(index="date", columns="factor", values="shapley", aggfunc="last").sort_index().fillna(0.0)
    piv = piv[[f for f in FACTOR_ORDER if f in piv.columns]]
    if piv.empty:
        return ""
    w, h, l, r, t, b = 340, 170, 34, 6, 18, 20
    n = len(piv)
    xs = np.linspace(l, w - r, n) if n > 1 else np.array([l])
    ymax = max(0.05, float(piv.sum(axis=1).max()))
    ys = lambda v: t + (h - t - b) * (1 - v / ymax)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}"><text x="{l}" y="12" style="fill:var(--text);font-weight:600">{esc(title)}</text>']
    for frac in (0, 0.5, 1):
        v = ymax * frac
        out.append(f'<line class="ax" x1="{l}" x2="{w - r}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/><text x="{l - 4}" y="{ys(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    base = np.zeros(n)
    for f in piv.columns:
        top_ = base + piv[f].to_numpy()
        pts = [f"{x:.1f},{ys(y):.1f}" for x, y in zip(xs, top_)] + [f"{x:.1f},{ys(y):.1f}" for x, y in zip(xs[::-1], base[::-1])]
        last = piv[f].iloc[-1]
        out.append(f'<polygon points="{" ".join(pts)}" fill="var(--f-{f})" stroke="var(--surface)" stroke-width="1"><title>{esc(FACTOR_LABELS.get(f, f))}: {last:.1%} del R² (última fecha)</title></polygon>')
        base = top_
    d0, d1 = piv.index[0], piv.index[-1]
    out.append(f'<text x="{l}" y="{h - 4}">{pd.Timestamp(d0):%d-%m-%y}</text><text x="{w - r}" y="{h - 4}" text-anchor="end">{pd.Timestamp(d1):%d-%m-%y}</text></svg>')
    return "".join(out)


def legend(factors: list[str]) -> str:
    items = "".join(f'<span><i style="background:var(--f-{f})"></i>{esc(FACTOR_LABELS.get(f, f))}</span>' for f in factors)
    return f'<div class="legend">{items}</div>'


def table(df: pd.DataFrame, numeric: set[str] = frozenset()) -> str:
    head = "".join(f'<th class="{"n" if c in numeric else ""}">{esc(c)}</th>' for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(f'<td class="{"n" if c in numeric else ""}">{esc(v)}</td>' for c, v in zip(df.columns, r)) + "</tr>"
        for r in df.itertuples(index=False)
    )
    return f'<div class="scroll"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


# ---------------------------------------------------------------------------- secciones


def section_status(meta: dict | None, now: datetime) -> str:
    if not meta:
        return '<p class="crit">Todavía no hay resultados del nocturno.</p>'
    steps = meta.get("steps", {})
    failed = [k for k, v in steps.items() if v.get("status") == "failed"]
    end = meta.get("data_end")
    stale = end is None or (pd.Timestamp(now.date()) - pd.offsets.BDay(1)).date() > pd.Timestamp(end).date()
    cards = [
        ("Datos hasta", esc(end or "—"), "crit" if end is None else ("warn" if stale else "ok")),
        ("Última ejecución", esc(str(meta.get("run_utc", ""))[:16].replace("T", " ")) + " UTC", ""),
        ("Pasos fallidos", esc(", ".join(failed) or "ninguno"), "crit" if failed else "ok"),
        ("Avisos", str(len(meta.get("warnings", []))), "warn" if meta.get("warnings") else "ok"),
    ]
    html_cards = "".join(f'<div class="card"><span class="muted">{k}</span><b class="{cls}">{v}</b></div>' for k, v, cls in cards)
    rows = pd.DataFrame(
        [{"paso": k, "estado": v.get("status"), "detalle": v.get("error") or v.get("reason") or v.get("mode") or ""} for k, v in steps.items()]
    )
    warns = meta.get("warnings", [])
    warn_html = ""
    if warns:
        items = "".join(f"<li>{esc(w.get('message'))}</li>" for w in warns[:60])
        warn_html = f"<details><summary>Ver {len(warns)} avisos</summary><ul>{items}</ul></details>"
    return f'<div class="cards">{html_cards}</div><details><summary>Pasos del nocturno</summary>{table(rows)}</details>{warn_html}'


def section_selection(sel: dict | None, r2: pd.DataFrame | None) -> str:
    if not sel:
        return '<p class="muted">Sin selección todavía.</p>'
    rows = []
    for t, v in sorted(sel.get("targets", {}).items()):
        r2v = None
        if r2 is not None and (r2["target"] == t).any():
            r2v = float(r2[r2["target"] == t].sort_values("date")["r2"].iloc[-1])
        status = {"ok": "con driver", "sin_driver_macro": "sin driver macro", "sin_datos": "sin datos"}.get(v.get("status"), v.get("status"))
        facs = ", ".join(f"{FACTOR_LABELS.get(f, f)} ({'+' if v.get('signs', {}).get(f, 0) > 0 else '−'})" for f in v.get("factors", []))
        rows.append({"objetivo": t, "R² diario": num(r2v, "{:.2f}"), "factores seleccionados (signo)": facs or "—", "estado": status,
                     "mercado": "sí" if v.get("market") else "no"})  # fmt: skip
    return (
        f'<p class="sub">Válida del {esc(sel.get("valid_from"))} al {esc(sel.get("valid_to"))} · calculada con datos hasta {esc(sel.get("as_of"))}.</p>'
        + table(pd.DataFrame(rows), {"R² diario"})
    )


def section_regime(shapley: pd.DataFrame | None) -> str:
    if shapley is None or shapley.empty:
        return '<p class="muted">Sin resultados de la capa diaria.</p>'
    last = shapley[shapley["date"] == shapley.groupby("target")["date"].transform("max")]
    shares = last[last["factor"] != "MKT"]
    heat = heatmap_svg(shares)
    tbl = shares.assign(factor=shares["factor"].map(lambda f: FACTOR_LABELS.get(f, f)), cuota=shares["share"].map(pct),
                        beta=shares["std_beta"].map(num))[["target", "factor", "cuota", "beta"]].rename(columns={"target": "objetivo", "beta": "beta est."})  # fmt: skip
    cutoff = shapley["date"].max() - pd.Timedelta(days=int(HISTORY_DAYS * 1.45))
    recent = shapley[shapley["date"] >= cutoff]
    present = [f for f in FACTOR_ORDER if f in set(recent["factor"])]
    multiples = "".join(stacked_svg(g, t) for t, g in sorted(recent.groupby("target")))
    return (
        '<p class="sub">Cuota Shapley de cada factor sobre el R² sin el mercado (última estimación, ventana de 60 sesiones).</p>'
        f"{heat}<details><summary>Ver tabla</summary>{table(tbl)}</details>"
        "<h2>Evolución: contribución de cada factor al R²</h2>"
        f'<p class="sub">Últimos {HISTORY_DAYS} días hábiles; la altura total es el R² diario.</p>{legend(present)}<div class="grid">{multiples}</div>'
    )


def section_track(track: pd.DataFrame | None, alerts: pd.DataFrame | None, thresholds: dict | None) -> str:
    parts = []
    if thresholds:
        rows = [{"objetivo": t, "z*": num(v.get("z_star"), "{:.2f}"), "mediana R² intradía": num(v.get("r2_median"), "{:.2f}"),
                 "sesiones": v.get("sessions")} for t, v in sorted(thresholds.get("targets", {}).items())]  # fmt: skip
        parts.append(f'<p class="sub">Umbral calibrado para ~{thresholds.get("target_alerts_per_week", 2):g} alertas por objetivo y semana (mínimo z* = {thresholds.get("z_floor", 2):g}).</p>')
        parts.append(table(pd.DataFrame(rows), {"z*", "mediana R² intradía", "sesiones"}))
    if track is None or track.empty:
        parts.append('<p class="muted">Sin historial de alertas todavía.</p>')
        return "".join(parts)
    top = track[track["level"] == "target×factor×origin"].sort_values("n", ascending=False)
    view = pd.DataFrame(
        {
            "objetivo": top["target"], "factor": top["main_factor"].map(lambda f: FACTOR_LABELS.get(f, f)), "origen": top["origin"],
            "alertas": top["n"], "conv. 30 min": top["conv_6"].map(pct), "conv. 60 min": top["conv_12"].map(pct),
            "neto 60 min (pb)": top["net_bps_12"].map(lambda v: num(v, "{:+.1f}")),
            "años +/total": top["years_positive"].astype(str) + "/" + top["years_ok"].astype(str),
            "suficiente": top["sufficient"].map(lambda b: "sí" if b else "no (<30)"),
            "persistente": top["persistent"].map(lambda b: "sí" if b else "no"),
        }
    )  # fmt: skip
    parts.append("<h2>Historial por celda</h2>")
    parts.append('<p class="sub">Convergencia medida sobre el gap fijado en el momento de la alerta. Neto = reversión menos coste (spread + comisión). Una alerta no es una señal de trading.</p>')
    parts.append(table(view, {"alertas", "conv. 30 min", "conv. 60 min", "neto 60 min (pb)", "años +/total"}))
    if alerts is not None and not alerts.empty:
        last = alerts.sort_values("ts", ascending=False).head(25)
        view2 = pd.DataFrame(
            {"fecha (UTC)": pd.to_datetime(last["ts"]).dt.strftime("%Y-%m-%d %H:%M"), "objetivo": last["target"], "z": last["z"].map(num),
             "gap (pb)": (last["gap"] * 1e4).map(lambda v: num(v, "{:+.1f}")), "factor": last["main_factor"].map(lambda f: FACTOR_LABELS.get(f, f)),
             "origen": last["origin"], "convergió 60 min": last["converged_12"].map(lambda v: "—" if pd.isna(v) else ("sí" if v else "no"))}
        )  # fmt: skip
        parts.append(f"<details><summary>Últimas {len(view2)} alertas históricas</summary>{table(view2, {'z', 'gap (pb)'})}</details>")
    return "".join(parts)


def build_report(results: Path, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    d = read(results)
    body = f"""<main>
<h1>Monitor de factores</h1>
<p class="sub">Informe nocturno · generado {now:%Y-%m-%d %H:%M} UTC · datos de Dukascopy (CFD) · mide co-movimiento, no causalidad.</p>
<h2>Estado</h2>{section_status(d["meta"], now)}
<h2>Selección de la semana</h2>{section_selection(d["selection"], d["r2"])}
<h2>Régimen de factores</h2>{section_regime(d["shapley"])}
<h2>Umbrales e historial de alertas</h2>{section_track(d["track"], d["alerts"], d["thresholds"])}
<p class="muted" style="margin-top:40px">Las alertas en vivo solo se ven en el PC con MT5. Este informe se regenera cada noche (L–V, 22:30 UTC).</p>
</main>"""
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitor de factores</title><style>{CSS}</style></head><body>{body}</body></html>"""


def write_report(results: Path) -> Path:
    path = results / "index.html"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(build_report(results), encoding="utf-8")
    tmp.replace(path)
    (results / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages sirve los ficheros tal cual
    return path
