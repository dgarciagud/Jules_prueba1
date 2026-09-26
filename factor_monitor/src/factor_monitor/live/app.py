"""Dashboard local (SPEC v3 §11). Solo lee lo que escriben el runner y el nocturno.

Uso:  streamlit run src/factor_monitor/live/app.py -- --results results_local --state state
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from factor_monitor.common.events import eia_events, load_events
from factor_monitor.common.universe import load_universe
from factor_monitor.live.dashboard_data import (
    FACTOR_COLORS_DARK,
    FACTOR_COLORS_LIGHT,
    FACTOR_LABELS,
    FACTOR_ORDER,
    Sources,
    cumulative_paths,
    data_status,
    decomposition,
    latest_shares,
    monitor_table,
    regime_table,
    shares_history,
)


def parse_args() -> argparse.Namespace:
    """Rutas por argumentos (`streamlit run app.py -- --state ...`) o variables FM_RESULTS, FM_STATE, FM_UNIVERSE."""
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.environ.get("FM_RESULTS", "results_local"))
    p.add_argument("--state", default=os.environ.get("FM_STATE", "state"))
    p.add_argument("--universe", default=os.environ.get("FM_UNIVERSE"))
    return p.parse_known_args(sys.argv[1:] if "streamlit" in sys.argv[0] else [])[0]


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - versiones sin st.context.theme
        return False


def factor_scale() -> alt.Scale:
    colors = FACTOR_COLORS_DARK if is_dark() else FACTOR_COLORS_LIGHT
    return alt.Scale(domain=[FACTOR_LABELS[f] for f in FACTOR_ORDER], range=colors)


def two_series_scale() -> alt.Scale:
    colors = FACTOR_COLORS_DARK if is_dark() else FACTOR_COLORS_LIGHT
    return alt.Scale(domain=["Real", "Implícito"], range=colors[:2])


def length_scale() -> alt.Scale:
    colors = FACTOR_COLORS_DARK if is_dark() else FACTOR_COLORS_LIGHT
    return alt.Scale(domain=["L = 6", "L = 12"], range=colors[:2])


def sequential_scheme() -> alt.Scale:
    # Una sola tonalidad (azul), de claro a oscuro: magnitud de la cuota.
    return alt.Scale(domain=[0, 1], range=["#cde2fb", "#104281"] if not is_dark() else ["#1c2a3d", "#86b6ef"])


ARGS = parse_args()
SRC = Sources(Path(ARGS.results), Path(ARGS.state))
UNIVERSE = load_universe(ARGS.universe) if ARGS.universe else load_universe()

st.set_page_config(page_title="Monitor de factores", layout="wide")
st.title("Monitor de factores")


def status_banner(now: pd.Timestamp) -> None:
    for level, text in data_status(SRC.run_meta(), SRC.status(), now):
        {"ok": st.caption, "warning": st.warning, "error": st.error}[level](text)


# ---------------------------------------------------------------------------- 1. Régimen


def view_regime() -> None:
    shapley, r2, selection = SRC.shapley(), SRC.daily_r2(), SRC.selection()
    if shapley is None:
        st.info("Todavía no hay resultados de la capa diaria.")
        return
    if selection:
        st.caption(f"Selección vigente del {selection.get('valid_from')} al {selection.get('valid_to')} (datos hasta {selection.get('as_of')}).")
    shares = latest_shares(shapley)
    shares["factor_label"] = shares["factor"].map(lambda f: FACTOR_LABELS.get(f, f))
    heat = (
        alt.Chart(shares)
        .mark_rect(cornerRadius=2, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("factor_label:N", title=None, sort=[FACTOR_LABELS[f] for f in FACTOR_ORDER], axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("target:N", title=None),
            color=alt.Color("share:Q", title="Cuota Shapley", scale=sequential_scheme()),
            tooltip=[alt.Tooltip("target:N", title="Objetivo"), alt.Tooltip("factor_label:N", title="Factor"),
                     alt.Tooltip("share:Q", title="Cuota", format=".0%"), alt.Tooltip("std_beta:Q", title="Beta est.", format="+.2f")],
        )  # fmt: skip
        .properties(height=260)
    )
    labels = heat.mark_text(fontSize=12).encode(
        text=alt.Text("share:Q", format=".0%"),
        color=alt.condition(alt.datum.share > 0.55, alt.value("white"), alt.value("#0b0b0b")),
    )
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Cuota de cada factor (sin el mercado)")
        st.altair_chart(heat + labels, use_container_width=True)
    with c2:
        st.subheader("R² y regresores de la semana")
        table = regime_table(shapley, r2, selection)
        st.dataframe(table[["objetivo", "R²", "seleccionados", "estado"]], hide_index=True, use_container_width=True,
                     column_config={"R²": st.column_config.NumberColumn(format="%.2f")})  # fmt: skip

    st.subheader("Evolución del R² por factor")
    target = st.selectbox("Objetivo", sorted(shapley["target"].unique()), key="regime_target")
    hist = shares_history(shapley, target)
    hist["factor_label"] = hist["factor"].astype(str).map(lambda f: FACTOR_LABELS.get(f, f))
    area = (
        alt.Chart(hist)
        .mark_area(line={"strokeWidth": 0}, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("shapley:Q", title="Contribución al R²", stack="zero", axis=alt.Axis(format=".0%")),
            color=alt.Color("factor_label:N", title="Factor", scale=factor_scale()),
            order=alt.Order("factor:N"),
            tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip("factor_label:N", title="Factor"),
                     alt.Tooltip("shapley:Q", title="Contribución", format=".1%")],
        )  # fmt: skip
        .properties(height=260)
    )
    st.altair_chart(area, use_container_width=True)
    with st.expander("Ver tabla"):
        st.dataframe(table, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------- 2. Monitor y 3. Detalle


def evals_all() -> dict[str, pd.DataFrame | None]:
    return {t: SRC.live_eval(t) for t in UNIVERSE.targets}


def view_monitor(now: pd.Timestamp) -> None:
    table = monitor_table(list(UNIVERSE.targets), evals_all(), SRC.status(), SRC.alerts(), SRC.track_record(), SRC.inventory(), now)
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "z": st.column_config.NumberColumn(format="%+.2f"),
            "z*": st.column_config.NumberColumn(format="%.2f"),
            "|z|/z*": st.column_config.ProgressColumn(min_value=0.0, max_value=1.5, format="%.2f"),
            "gap (pb)": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
    alerts = SRC.alerts()
    st.subheader("Alertas de hoy")
    if alerts is None or alerts.empty:
        st.caption("Sin alertas registradas.")
    else:
        today = alerts[pd.to_datetime(alerts["ts"], utc=True).dt.date == now.date()].copy()
        if today.empty:
            st.caption("Sin alertas hoy.")
        else:
            today["gap (pb)"] = today["gap"] * 1e4
            st.dataframe(today[["ts", "target", "L", "z", "z_star", "gap (pb)", "main_factor", "origin"]].sort_values("ts", ascending=False),
                         hide_index=True, use_container_width=True)  # fmt: skip


def view_detail(now: pd.Timestamp) -> None:
    status = SRC.status() or {}
    target = st.selectbox("Objetivo", list(UNIVERSE.targets), key="detail_target")
    ev = SRC.live_eval(target)
    model = status.get("models", {}).get(target, {})
    if ev is None or ev.empty:
        st.info(status.get("excluded", {}).get(target) or "Sin evaluación en vivo para este objetivo (sesión cerrada o sin datos).")
        return
    z_star = float(model.get("z_star") or 2.5)
    st.caption(f"Factores: {', '.join(FACTOR_LABELS.get(f, f) for f in model.get('factors', [])) or '—'} · R² intradía {model.get('r2', float('nan')):.2f}"
               f" · {'filtro R² superado' if model.get('r2_ok') else 'R² por debajo de su mediana (no alerta)'}"
               + (f" · {model['note']}" if model.get("note") else "") + ". Factores ortogonalizados respecto al mercado.")  # fmt: skip

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Real frente a implícito desde la apertura")
        paths = cumulative_paths(ev)
        line = (
            alt.Chart(paths)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("ts:T", title="hora (UTC)", axis=alt.Axis(format="%H:%M")),
                y=alt.Y("pb:Q", title="Acumulado (pb)"),
                color=alt.Color("serie:N", title=None, scale=two_series_scale(), legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("ts:T", title="Hora", format="%H:%M"), alt.Tooltip("serie:N"), alt.Tooltip("pb:Q", format="+.1f")],
            )  # fmt: skip
            .properties(height=280)
        )
        st.altair_chart(line, use_container_width=True)
    with c2:
        st.subheader("z con bandas ±z*")
        zdf = ev[[c for c in ev.columns if c.startswith("z_")]].reset_index(names="ts").melt(id_vars="ts", var_name="L", value_name="z")
        zdf["L"] = zdf["L"].str.replace("z_", "L = ")
        zline = (
            alt.Chart(zdf.dropna())
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("ts:T", title="hora (UTC)", axis=alt.Axis(format="%H:%M")), y=alt.Y("z:Q", title="z"),
                color=alt.Color("L:N", title=None, scale=length_scale(), legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("ts:T", title="Hora", format="%H:%M"), "L:N", alt.Tooltip("z:Q", format="+.2f")],
            )  # fmt: skip
        )
        bands = alt.Chart(pd.DataFrame({"y": [z_star, -z_star]})).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(y="y:Q")
        st.altair_chart((bands + zline).properties(height=280), use_container_width=True)

    L = st.radio("Ventana del gap", [6, 12], horizontal=True, format_func=lambda v: f"L = {v} velas ({v * 5} min)")
    dec = decomposition(ev, L)
    st.subheader("Descomposición del implícito por factor")
    if dec.empty:
        st.caption("Sin descomposición todavía (hacen falta L velas de sesión).")
    else:
        bars = (
            alt.Chart(dec)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("etiqueta:N", title=None, sort=[FACTOR_LABELS[f] for f in FACTOR_ORDER]),
                x=alt.X("pb:Q", title="Aportación (pb)"),
                color=alt.Color("etiqueta:N", scale=factor_scale(), legend=None),
                tooltip=[alt.Tooltip("etiqueta:N", title="Factor"), alt.Tooltip("pb:Q", format="+.1f", title="pb")],
            )  # fmt: skip
            .properties(height=alt.Step(36))
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8985").encode(x="x:Q")
        st.altair_chart(bars + zero, use_container_width=True)
        st.dataframe(dec[["etiqueta", "pb"]].rename(columns={"etiqueta": "factor"}), hide_index=True,
                     column_config={"pb": st.column_config.NumberColumn(format="%+.1f")})

    st.subheader("Eventos del día")
    events = load_events()
    day = now.normalize()
    eia = eia_events(day.date(), day.date())
    ev_day = pd.concat([events, eia], ignore_index=True) if not events.empty else eia
    ev_day = ev_day[ev_day["datetime_utc"].dt.normalize() == day] if not ev_day.empty else ev_day
    if ev_day.empty:
        st.caption("Sin eventos en el calendario para hoy." + (" (events.csv está vacío)" if events.empty else ""))
    else:
        st.dataframe(ev_day[["datetime_utc", "event_type", "region", "source"]], hide_index=True)


# ---------------------------------------------------------------------------- 4. Estado del sistema


def view_system(now: pd.Timestamp) -> None:
    meta, status = SRC.run_meta(), SRC.status()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Nocturno (GitHub Actions)")
        if meta:
            st.write(f"Última ejecución: {meta.get('run_utc')} · datos hasta **{meta.get('data_end')}** · versión {str(meta.get('code_version'))[:8]}")
            steps = pd.DataFrame([{"paso": k, **v} for k, v in meta.get("steps", {}).items()])
            st.dataframe(steps, hide_index=True, use_container_width=True)
            if meta.get("warnings"):
                with st.expander(f"Avisos del nocturno ({len(meta['warnings'])})"):
                    st.dataframe(pd.DataFrame(meta["warnings"]), hide_index=True, use_container_width=True)
        else:
            st.info("Sin run_meta.json.")
    with c2:
        st.subheader("Runner local")
        if status:
            st.write(f"Última actualización: {status.get('updated_utc')}")
            st.write(f"Hora del servidor MT5: UTC{status.get('server_offset_hours'):+d} ({status.get('offset_source')})"
                     if status.get("server_offset_hours") is not None else "Hora del servidor MT5: sin detectar")  # fmt: skip
            if status.get("excluded"):
                st.write("Objetivos sin modelo hoy:")
                st.dataframe(pd.DataFrame([{"objetivo": k, "motivo": v} for k, v in status["excluded"].items()]), hide_index=True)
            if status.get("errors"):
                with st.expander(f"Errores recientes ({len(status['errors'])})"):
                    st.code("\n".join(status["errors"]))
        else:
            st.info("El runner no está en marcha.")
    st.subheader("Consistencia Dukascopy – MT5")
    cons = SRC.consistency()
    if cons is None:
        st.caption("Sin test de consistencia todavía.")
    else:
        st.dataframe(cons, hide_index=True, use_container_width=True, column_config={"corr": st.column_config.NumberColumn(format="%.3f")})
    st.subheader("Inventario MT5")
    inv = SRC.inventory()
    if inv is None:
        st.caption("Sin inventario todavía.")
    else:
        st.dataframe(inv, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------- disposición


@st.fragment(run_every="60s")
def live_area() -> None:
    now = pd.Timestamp.now(tz="UTC")
    status_banner(now)
    tabs = st.tabs(["Régimen", "Monitor", "Detalle", "Estado del sistema"])
    with tabs[0]:
        view_regime()
    with tabs[1]:
        view_monitor(now)
    with tabs[2]:
        view_detail(now)
    with tabs[3]:
        view_system(now)
    st.caption(f"Actualizado {now:%H:%M:%S} UTC · se refresca cada 60 s · las alertas no son señales de trading.")


live_area()
