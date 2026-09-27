"""Paso 4: informe HTML estático (GitHub Pages).

Una sola página autocontenida (sin librerías externas), con modo claro y oscuro. Por índice:
resumen, atribución sectorial por periodo (y por acción en el CAC 40), fuerza relativa y
factores macro por sector. El color solo codifica signo e intensidad (azul positivo, rojo
negativo); los sectores se identifican siempre por su nombre. Cada marca tiene tooltip y
cada gráfico su tabla.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import PERIODS, RS_HORIZONS

PERIOD_LABELS = {"1D": "1 día", "1S": "1 semana", "1M": "1 mes", "3M": "3 meses", "YTD": "Año", "1A": "12 meses"}
FACTOR_LABELS = {"US10Y": "Tipo 10a EE.UU.", "BUND": "Bund largo", "OIL": "Petróleo", "USD": "Dólar", "EURUSD": "EUR/USD",
                 "AI": "Semis / IA", "GOLD": "Oro", "VIX": "VIX"}  # fmt: skip
DEFAULT_INDEX, DEFAULT_PERIOD = "SPX", "1S"

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--surface-2:#f0efec;--text:#0b0b0b;--text-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--pos:#2a78d6;--neg:#e34948;--neutral:#b5b3ab;--accent:#2a78d6;--warn:#b57800}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;
--text:#fff;--text-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--pos:#3987e5;--neg:#e66767;--neutral:#5e5d58;--accent:#3987e5;--warn:#fab219}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--surface-2:#242422;--text:#fff;--text-2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--pos:#3987e5;--neg:#e66767;--neutral:#5e5d58;--accent:#3987e5;--warn:#fab219}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 6px}h3{font-size:15px;margin:22px 0 6px}
.sub,.muted{color:var(--text-2)}.small{font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:14px 0}
.seg{display:inline-flex;max-width:100%;overflow-x:auto;border:1px solid var(--ring);border-radius:8px;background:var(--surface)}
.seg button{font:inherit;font-size:14px;border:0;background:none;color:var(--text-2);padding:6px 12px;cursor:pointer;white-space:nowrap}
.seg button[aria-pressed="true"]{background:var(--surface-2);color:var(--text);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 14px}
.tile span{display:block;color:var(--text-2);font-size:13px}.tile b{font-size:22px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px;margin:10px 0}
.lead{font-size:16px;margin:8px 0}
svg{display:block;width:100%;height:auto;max-width:760px}
.bars{display:grid;gap:4px;margin:8px 0}
.br{display:grid;grid-template-columns:minmax(90px,34%) 1fr 76px;align-items:center;gap:8px;font-size:13px}
.bl{text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bt{position:relative;height:16px}.bt b{position:absolute;top:-4px;bottom:-4px;width:1px;background:var(--axis)}
.bt i{position:absolute;top:0;height:16px;border-radius:4px}
.bt i.pos{background:var(--pos)}.bt i.neg{background:var(--neg)}.bt i.res{background:var(--neutral)}
.bv{color:var(--text-2);font-variant-numeric:tabular-nums;white-space:nowrap}svg text{fill:var(--text-2);font-size:12px}svg .ink{fill:var(--text)}
svg .grid{stroke:var(--grid)}svg .axis{stroke:var(--axis)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--text-2);font-weight:600}td.n,th.n{text-align:right}
td.cell{text-align:right;min-width:62px}
details{margin:8px 0}summary{cursor:pointer;color:var(--text-2)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--text-2);margin:6px 0}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.warn{color:var(--warn)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media (max-width:760px){.two{grid-template-columns:1fr}}
[hidden]{display:none!important}
"""

JS = """
const state={ix:document.body.dataset.ix,per:document.body.dataset.per};
function apply(){document.querySelectorAll('[data-ix]').forEach(e=>{if(e!==document.body)e.hidden=e.dataset.ix!==state.ix});
document.querySelectorAll('[data-per]').forEach(e=>{if(e!==document.body&&e.dataset.ix===state.ix)e.hidden=e.dataset.per!==state.per});
document.querySelectorAll('button[data-set]').forEach(b=>b.setAttribute('aria-pressed',state[b.dataset.set]===b.dataset.val));
try{localStorage.setItem('fm',JSON.stringify(state))}catch(e){}}
document.querySelectorAll('button[data-set]').forEach(b=>b.addEventListener('click',()=>{state[b.dataset.set]=b.dataset.val;apply()}));
try{const s=JSON.parse(localStorage.getItem('fm')||'{}');if(s.ix&&document.querySelector('[data-ix="'+s.ix+'"]'))state.ix=s.ix;if(s.per)state.per=s.per}catch(e){}
apply();
"""


# ----------------------------------------------------------------------------- formato


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def num(x, digits=2, signed=True, suffix="") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace("-", "−").replace(".", ",") + suffix


def pct(x, digits=2, signed=True) -> str:
    return num(None if x is None else x * 100, digits, signed, " %")


def pp(x, digits=2) -> str:
    return num(None if x is None else x * 100, digits, True, " pp")


def fill(v: float, vmax: float) -> str:
    """Color de celda divergente: azul positivo, rojo negativo, intensidad ∝ |v|/vmax."""
    if v is None or not np.isfinite(v) or vmax <= 0:
        return ""
    a = min(1.0, abs(v) / vmax)
    pole = "var(--pos)" if v > 0 else "var(--neg)"
    ink = "#fff" if a > 0.6 else "var(--text)"
    return f' style="background:color-mix(in srgb,{pole} {a * 88:.0f}%,var(--surface));color:{ink}"'


def seg(name: str, options: list[tuple[str, str]], current: str) -> str:
    btn = "".join(f'<button type="button" data-set="{name}" data-val="{esc(v)}" aria-pressed="{str(v == current).lower()}">{esc(lbl)}</button>'
                  for v, lbl in options)  # fmt: skip
    return f'<div class="seg" role="group">{btn}</div>'


# ----------------------------------------------------------------------------- gráficos


def bars_svg(rows: list[tuple[str, float, str]], title: str, fmt=pp) -> str:
    """Barras horizontales divergentes en HTML (se adaptan al ancho). rows = (etiqueta, valor, tooltip)."""
    if not rows:
        return '<p class="muted small">Sin datos.</p>'
    vals = [v for _, v, _ in rows if v is not None and np.isfinite(v)]
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    span = (hi - lo) or 1.0
    zero = (0 - lo) / span * 100
    out = [f'<div class="bars" role="img" aria-label="{esc(title)}">']
    for label, v, tip in rows:
        bar = ""
        if v is not None and np.isfinite(v):
            w = max(abs(v) / span * 100, 0.4)
            x = zero if v >= 0 else zero - w
            cls = "res" if label == "Residuo" else ("pos" if v >= 0 else "neg")
            bar = f'<i class="{cls}" style="left:{x:.2f}%;width:{w:.2f}%"></i>'
        out.append(f'<div class="br" title="{esc(tip)}"><span class="bl">{esc(label)}</span>'
                   f'<span class="bt"><b style="left:{zero:.2f}%"></b>{bar}</span><span class="bv">{esc(fmt(v))}</span></div>')  # fmt: skip
    out.append("</div>")
    return "".join(out)


def rotation_svg(df: pd.DataFrame) -> str:
    """Rotación: x = fuerza relativa a 3 meses, y = su cambio en el último mes. Un punto por sector, con nombre."""
    if df.empty:
        return ""
    W, H, m = 640, 420, 44
    xs, ys = df["rs"].to_numpy(), df["rs_change_1m"].to_numpy()
    xr = max(abs(xs).max(), 0.01) * 1.25
    yr = max(abs(ys).max(), 0.01) * 1.25
    X = lambda v: m + (v + xr) / (2 * xr) * (W - 2 * m)  # noqa: E731
    Y = lambda v: H - m - (v + yr) / (2 * yr) * (H - 2 * m)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rotación sectorial">']
    out.append(f'<line class="axis" x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{m - 10}" y2="{H - m + 10}"/>')
    out.append(f'<line class="axis" x1="{m - 10}" x2="{W - m + 10}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    for txt, x, y, a in (("Liderando", W - m, m - 16, "end"), ("Mejorando", m, m - 16, "start"),
                         ("Rezagados", m, H - m + 26, "start"), ("Perdiendo impulso", W - m, H - m + 26, "end")):  # fmt: skip
        out.append(f'<text x="{x}" y="{y}" text-anchor="{a}" class="small">{txt}</text>')
    out.append(f'<text x="{W / 2}" y="{H - 4}" text-anchor="middle">Fuerza relativa a 3 meses →</text>')
    out.append(f'<text x="12" y="{H / 2}" text-anchor="middle" transform="rotate(-90 12 {H / 2})">Cambio en 1 mes →</text>')
    boxes: list[tuple[float, float, float, float]] = []  # etiquetas y puntos ya colocados (x0, y0, x1, y1)
    pts = [(X(r["rs"]), Y(r["rs_change_1m"])) for _, r in df.iterrows()]
    boxes += [(px - 6, py - 6, px + 6, py + 6) for px, py in pts]

    def free(b):
        return all(b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3] for o in boxes)

    for _, r in df.sort_values("rs").iterrows():
        cx, cy = X(r["rs"]), Y(r["rs_change_1m"])
        tw = 6.6 * len(r["name"])
        best = None
        for dy in (0, -14, 14, -28, 28, -42, 42):
            for side in (1, -1):
                x0 = cx + 9 if side > 0 else cx - 9 - tw
                if x0 < 2 or x0 + tw > W - 2:
                    continue
                bx = (x0, cy + dy - 9, x0 + tw, cy + dy + 4)
                if free(bx):
                    best = (bx, side, dy)
                    break
            if best:
                break
        if best is None:
            x0 = min(max(cx + 9, 2), W - 2 - tw)
            best = ((x0, cy - 9, x0 + tw, cy + 4), 1, 0)
        bx, side, dy = best
        boxes.append(bx)
        tip = f"{r['name']}: fuerza relativa 3M {pct(r['rs'], 1)}, cambio en 1M {pct(r['rs_change_1m'], 1)}"
        if dy:
            out.append(f'<line class="grid" x1="{cx:.1f}" y1="{cy:.1f}" x2="{(bx[0] if side > 0 else bx[2]):.1f}" y2="{cy + dy - 3:.1f}"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="transparent"><title>{esc(tip)}</title></circle>')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="var(--accent)" stroke="var(--surface)" stroke-width="2"><title>{esc(tip)}</title></circle>')
        out.append(f'<text x="{bx[0]:.1f}" y="{cy + dy + 1:.1f}" class="ink">{esc(r["name"])}</text>')
    out.append("</svg>")
    return "".join(out)


def legend_div(pos: str, neg: str) -> str:
    return f'<div class="legend"><span><i style="background:var(--pos)"></i>{esc(pos)}</span><span><i style="background:var(--neg)"></i>{esc(neg)}</span></div>'


# ----------------------------------------------------------------------------- secciones


def section_summary(ix: str, name: str, attr: pd.DataFrame) -> str:
    a = attr[(attr["index"] == ix)]
    tot = a[a.level == "index"].set_index("period")["contribution"]
    tiles = "".join(f'<div class="tile"><span>{esc(PERIOD_LABELS[p])}</span><b>{pct(tot.get(p), 2)}</b></div>' for p in PERIODS if p in tot)
    wk = a[(a.period == "1S") & (a.level == "sector")].sort_values("contribution")
    lead = ""
    if len(wk) >= 2:
        top, bot = wk.iloc[-1], wk.iloc[0]
        lead = (f'<p class="lead">En la última semana el {esc(name)} {"sube" if tot.get("1S", 0) >= 0 else "baja"} {pct(abs(tot.get("1S", 0)), 2, False)}. '
                f'Lo que más suma: <b>{esc(top["name"])}</b> ({pp(top["contribution"])}); lo que más resta: <b>{esc(bot["name"])}</b> ({pp(bot["contribution"])}).</p>')  # fmt: skip
    return f'<div class="tiles">{tiles}</div>{lead}'


def section_attribution(ix: str, attr: pd.DataFrame) -> str:
    out = []
    a = attr[attr["index"] == ix]
    for p in PERIODS:
        g = a[a.period == p]
        if g.empty:
            continue
        tot = g[g.level == "index"].iloc[0]
        sec = g[g.level == "sector"].sort_values("contribution", ascending=False)
        res = g[g.level == "residual"].iloc[0]
        rows = [(r["name"], r["contribution"], f"{r['name']}: aporta {pp(r['contribution'])}; el sector {pct(r['ret'])}; peso {pct(r['weight_start'], 1, False)}")
                for _, r in sec.iterrows()]  # fmt: skip
        rows.append(("Residuo", res["contribution"], f"Residuo (lo que los sectores no reproducen): {pp(res['contribution'])}"))
        table = "".join(f'<tr><td>{esc(r["name"])}</td><td class="n">{pp(r["contribution"])}</td><td class="n">{pct(r["ret"])}</td>'
                        f'<td class="n">{pct(r["weight_start"], 1, False)}</td></tr>' for _, r in sec.iterrows())  # fmt: skip
        block = [f'<div data-ix="{ix}" data-per="{p}"{"" if p == DEFAULT_PERIOD else " hidden"}>',
                 f'<p class="muted small">{esc(tot["from"])} a {esc(tot["to"])} · índice {pct(tot["contribution"])} · '
                 f'aportación = peso × rentabilidad del sector, en puntos porcentuales</p>',
                 legend_div("Suma", "Resta"), bars_svg(rows, f"Atribución {ix} {p}"),
                 f'<details><summary>Ver tabla</summary><div class="scroll"><table><tr><th>Sector</th><th class="n">Aportación</th>'
                 f'<th class="n">Rentabilidad</th><th class="n">Peso inicial</th></tr>{table}<tr><td>Residuo</td><td class="n">{pp(res["contribution"])}</td>'
                 f'<td></td><td></td></tr></table></div></details>']  # fmt: skip
        mem = g[g.level == "member"].sort_values("contribution")
        if not mem.empty:
            top = mem.tail(6).iloc[::-1]
            bot = mem.head(6)
            mk = lambda d: [(r["name"], r["contribution"], f"{r['name']}: aporta {pp(r['contribution'])}; la acción {pct(r['ret'])}; peso {pct(r['weight_start'], 1, False)}")  # noqa: E731
                            for _, r in d.iterrows()]  # fmt: skip
            block.append(f'<div class="two"><div><h3>Acciones que más suman</h3>{bars_svg(mk(top), "Más suman")}</div>'
                         f'<div><h3>Acciones que más restan</h3>{bars_svg(mk(bot), "Más restan")}</div></div>')  # fmt: skip
        block.append("</div>")
        out.append("".join(block))
    return "".join(out)


def section_rs(ix: str, rs: pd.DataFrame) -> str:
    g = rs[rs["index"] == ix] if not rs.empty else rs
    if g.empty:
        return '<p class="muted">Sin datos suficientes todavía.</p>'
    wide_rs = g.pivot_table(index="name", columns="horizon", values="rs")
    mom = g[g.horizon == "3M"].set_index("name")["rs_change_1m"]
    hs = [h for h in RS_HORIZONS if h in wide_rs.columns]
    order = wide_rs["3M"].sort_values(ascending=False).index if "3M" in wide_rs else wide_rs.index
    vmax = float(np.nanmax(np.abs(wide_rs.to_numpy()))) if wide_rs.size else 1.0
    head = "".join(f'<th class="n">{esc(h)}</th>' for h in hs)
    body = ""
    for name in order:
        cells = "".join(f'<td class="cell"{fill(wide_rs.loc[name, h], vmax)} title="{esc(name)} frente al índice, {h}: {pct(wide_rs.loc[name, h], 1)}">'
                        f'{pct(wide_rs.loc[name, h], 1)}</td>' for h in hs)  # fmt: skip
        mv = mom.get(name, np.nan)
        arrow = "▲" if mv > 0 else "▼" if mv < 0 else ""
        body += f'<tr><td>{esc(name)}</td>{cells}<td class="n">{arrow} {pct(mv, 1)}</td></tr>'
    scatter = rotation_svg(g[g.horizon == "3M"])
    scatter = f'<div class="scroll"><div style="min-width:560px">{scatter}</div></div>'
    return (f'<p class="muted small">Rentabilidad del sector frente a su índice: (1 + sector) / (1 + índice) − 1. '
            f'La última columna es el cambio de la fuerza relativa a 3 meses en el último mes (impulso).</p>'
            f'{legend_div("Mejor que el índice", "Peor que el índice")}'
            f'<div class="scroll"><table><tr><th>Sector</th>{head}<th class="n">Impulso 1M</th></tr>{body}</table></div>'
            f'<h3>Rotación</h3><p class="muted small">Arriba a la derecha, sectores que baten al índice y siguen ganando terreno; '
            f'abajo a la derecha, los que lo baten pero pierden impulso.</p>{scatter}')  # fmt: skip


def section_factors(ix: str, fs: pd.DataFrame) -> str:
    g = fs[fs["index"] == ix] if not fs.empty else fs
    if g.empty:
        return '<p class="muted">Sin datos suficientes todavía (hacen falta 6 meses).</p>'
    last_date = g["date"].max()
    cur = g[g.date == last_date]
    factors = [f for f in FACTOR_LABELS if f in set(cur.factor)]
    targets = ["_index"] + sorted([t for t in cur.target.unique() if t != "_index"], key=lambda t: cur[cur.target == t].target_name.iloc[0])
    head = "".join(f'<th class="n">{esc(FACTOR_LABELS[f])}</th>' for f in factors)
    body = ""
    for t in targets:
        m = cur[cur.target == t].set_index("factor")
        if m.empty:
            continue
        name = m.target_name.iloc[0]
        r2 = m.r2.iloc[0]
        mkt = m.shapley.get("MKT", np.nan)
        cells = ""
        for f in factors:
            share = m.share.get(f, np.nan)
            sign = np.sign(m.std_beta.get(f, 0.0)) or 1.0
            v = share * sign
            tip = f"{name} · {FACTOR_LABELS[f]}: {share:.0%} de lo explicado sin el índice; beta {'positiva' if sign > 0 else 'negativa'}"
            cells += f'<td class="cell"{fill(v, 0.7)} title="{esc(tip)}">{num(share * 100, 0, False, " %") if np.isfinite(share) else "—"}</td>'
        mkt_txt = "—" if t == "_index" else num(mkt * 100, 0, False, " %")
        bold = " style=\"font-weight:600\"" if t == "_index" else ""
        body += f'<tr><td{bold}>{esc(name)}</td><td class="n">{num(r2, 2, False)}</td><td class="n">{mkt_txt}</td>{cells}</tr>'
    # cambios de régimen: factor dominante ahora, hace 6 y 12 meses
    snaps = sorted(g.date.unique())
    def dominant(d, t):
        m = g[(g.date == d) & (g.target == t) & (g.factor != "MKT")]
        if m.empty:
            return "—"
        r = m.sort_values("shapley", ascending=False).iloc[0]
        return f'{FACTOR_LABELS.get(r["factor"], r["factor"])} ({"+" if r["std_beta"] > 0 else "−"})'
    ago = {lbl: next((d for d in reversed(snaps) if d <= last_date - pd.DateOffset(months=k)), None) for lbl, k in (("hace 6 meses", 6), ("hace 12 meses", 12))}
    reg = "".join(f'<tr><td>{esc(cur[cur.target == t].target_name.iloc[0])}</td><td>{esc(dominant(last_date, t))}</td>'
                  + "".join(f"<td>{esc(dominant(d, t)) if d is not None else '—'}</td>" for d in ago.values()) + "</tr>"
                  for t in targets if not cur[cur.target == t].empty)  # fmt: skip
    rd = int(cur.return_days.iloc[0])
    return (f'<p class="muted small">Ventana de 6 meses hasta {pd.Timestamp(last_date):%d-%m-%Y}. Cada sector se explica con su índice y los factores '
            f'(a cada factor se le quita lo que comparte con el índice). R² = parte del movimiento explicada; «Índice» = parte que va con el índice; '
            f'cada factor = su cuota de lo que el sector hace aparte del índice. Azul: se mueve con el factor; rojo: en contra.'
            f'{" Rentabilidades de 2 días (el CAC 40 cierra antes que EE.UU.)." if rd > 1 else ""}</p>'
            f'{legend_div("Beta positiva", "Beta negativa")}'
            f'<div class="scroll"><table><tr><th>Sector</th><th class="n">R²</th><th class="n">Índice</th>{head}</tr>{body}</table></div>'
            f'<h3>Cambios de régimen</h3><p class="muted small">Factor macro dominante (y signo) en cada ventana.</p>'
            f'<div class="scroll"><table><tr><th>Sector</th><th>Ahora</th><th>Hace 6 meses</th><th>Hace 12 meses</th></tr>{reg}</table></div>')  # fmt: skip


def section_data(meta: dict, weights: dict, coverage: pd.DataFrame | None) -> str:
    tr = meta.get("tracking", {})
    rows = "".join(f'<tr><td>{esc(k)}</td><td class="n">{num(v.get("r2"), 4, False)}</td><td class="n">{pct(v.get("te_annual"), 2, False)}</td>'
                   f'<td>{esc((weights.get(k) or {}).get("source", ""))}</td></tr>' for k, v in tr.items())  # fmt: skip
    warns = "".join(f"<li>{esc(w['message'])}</li>" for w in meta.get("warnings", []))
    src = ", ".join(f"{k}: {v}" for k, v in (meta.get("sources") or {}).items())
    cov = ""
    if coverage is not None and not coverage.empty:
        cov = "".join(f'<tr><td>{esc(r["id"])}</td><td>{esc(r["name"])}</td><td>{esc(r["source"])}</td><td>{esc(r["symbol"])}</td>'
                      f'<td>{esc(r["first"])}</td><td>{esc(r["last"])}</td></tr>' for _, r in coverage.iterrows())  # fmt: skip
    return (f'<p class="small">Fuentes: {esc(src)}. Pesos estimados con el propio índice; comprobación en el último año:</p>'
            f'<div class="scroll"><table><tr><th>Índice</th><th class="n">R² sectores → índice</th><th class="n">Error de seguimiento</th><th>Pesos</th></tr>{rows}</table></div>'
            + (f'<h3 class="warn">Avisos</h3><ul class="small">{warns}</ul>' if warns else '<p class="muted small">Sin avisos.</p>')
            + (f'<details><summary>Cobertura de las {len(coverage)} series</summary><div class="scroll"><table><tr><th>Serie</th><th>Nombre</th><th>Fuente</th>'
               f'<th>Símbolo</th><th>Desde</th><th>Hasta</th></tr>{cov}</table></div></details>' if cov else ""))  # fmt: skip


# ----------------------------------------------------------------------------- página


def build_report(out: Path, index_names: dict[str, str], now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    meta = json.loads((out / "data_meta.json").read_text(encoding="utf-8"))
    weights = json.loads((out / "weights_ref.json").read_text(encoding="utf-8")) if (out / "weights_ref.json").exists() else {}
    rd = lambda n: pd.read_parquet(out / n) if (out / n).exists() else pd.DataFrame()  # noqa: E731
    attr, rs, fs = rd("attribution.parquet"), rd("relative_strength.parquet"), rd("factor_shapley.parquet")
    coverage = pd.read_csv(out / "coverage.csv") if (out / "coverage.csv").exists() else None
    indices = [i for i in index_names if not attr.empty and i in set(attr["index"])]
    default_ix = DEFAULT_INDEX if DEFAULT_INDEX in indices else (indices[0] if indices else "")
    controls = (f'<div class="bar">{seg("ix", [(i, index_names[i]) for i in indices], default_ix)}</div>' if indices else "")
    blocks = []
    for ix in indices:
        name = index_names[ix]
        blocks.append(
            f'<section data-ix="{ix}"{"" if ix == default_ix else " hidden"}>'
            f"<h2>{esc(name)}: resumen</h2>{section_summary(ix, name, attr)}"
            f'<h2>Qué sectores lo han movido</h2><div class="bar">{seg("per", [(p, PERIOD_LABELS[p]) for p in PERIODS], DEFAULT_PERIOD)}</div>'
            f'<div class="card">{section_attribution(ix, attr)}</div>'
            f'<h2>Fuerza relativa de los sectores</h2><div class="card">{section_rs(ix, rs)}</div>'
            f'<h2>Qué factor macro mueve cada sector</h2><div class="card">{section_factors(ix, fs)}</div>'
            "</section>"
        )  # fmt: skip
    warn_n = len(meta.get("warnings", []))
    body = (f'<main><h1>Sectores y factores</h1>'
            f'<p class="sub">Datos de cierre hasta <b>{esc(meta.get("data_end"))}</b> · generado {now:%d-%m-%Y %H:%M} UTC · '
            f'{"<span class=warn>" + str(warn_n) + " avisos</span>" if warn_n else "sin avisos"} · mide co-movimiento, no causalidad.</p>'
            f'{controls}{"".join(blocks)}'
            f'<h2>Datos y método</h2><div class="card">{section_data(meta, weights, coverage)}</div>'
            f'<p class="muted small">Fuente: Yahoo Finance (Stooq de respaldo). Se actualiza de lunes a viernes tras el cierre de EE.UU.</p></main>')  # fmt: skip
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Sectores y factores</title><style>{CSS}</style></head>'
            f'<body data-ix="{esc(default_ix)}" data-per="{DEFAULT_PERIOD}">{body}<script>{JS}</script></body></html>')  # fmt: skip


def write_report(out: Path, site: Path, index_names: dict[str, str]) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    path = site / "index.html"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(build_report(out, index_names), encoding="utf-8")
    tmp.replace(path)
    (site / ".nojekyll").write_text("", encoding="utf-8")
    return path
