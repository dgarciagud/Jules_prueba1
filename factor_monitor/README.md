# factor_monitor

Monitor de drivers y dislocaciones intradía. Especificación: [`../SPEC_v3_monitor_factores.md`](../SPEC_v3_monitor_factores.md).

**No ejecuta operaciones.** Detecta, explica y lleva el historial de cada alerta. Una alerta no es una señal de trading.

## Estado

| Paso | Estado |
|---|---|
| 1. `common/`: sesiones, festivos, horario de verano, factores, modelos | hecho |
| 2. Dukascopy, velas de 5 min, almacén en la release, `backfill` | hecho (backfill en curso) |
| 3. Capa diaria, selección y `nightly.yml` | hecho |
| 4. Capa intradía, alertas, eventos e historial de alertas | hecho |
| 5. Local: MT5, sincronización, consistencia, runner | hecho (probado con mock de MT5) |
| 6. Dashboard Streamlit | hecho |
| Cestas con yfinance/Stooq (pesos, dividendos, ETF) | pendiente |
| Calendario de eventos (`config/events.csv`) | pendiente de rellenar con fuentes oficiales |

## Arquitectura

```
GitHub Actions (nocturno, L–V 22:30 UTC)              PC Windows (en vivo)
  Dukascopy (jetta) → velas 5 min (release)             MT5 Darwinex abierto
  capa diaria → selección semanal                        runner.py  ── único proceso con MT5
  historial de alertas → umbrales z*                        cada 5 min: evalúa y registra alertas
  publica en la rama `results`  ───── git fetch ─────►   app.py (Streamlit) ── solo lee
```

## Estructura

```
factor_monitor/
├── config/
│   ├── universe.yaml               # objetivos, factores, IDs Dukascopy/jetta y símbolos MT5
│   ├── events.csv                  # calendario macro curado (vacío hasta rellenarlo)
│   └── costs.yaml                  # comisión de Darwinex por objetivo (0 = pendiente)
├── src/factor_monitor/
│   ├── common/                     # compartido por el nocturno y el runner
│   │   ├── sessions.py, calendars.py, universe.py, factors.py, models.py, bars.py, runlog.py
│   │   ├── intraday.py             # modelo intradía: ridge, σ por franja, z, descomposición, origen
│   │   ├── alerts.py               # reglas de alerta y enfriamiento
│   │   └── events.py               # calendario, regla EIA, ventanas de bloqueo
│   ├── sources/
│   │   ├── dukascopy.py            # cliente jetta con ritmo fijo (+ dukascopy-node y .bi5)
│   │   └── mt5.py                  # MetaTrader5: inventario, hora del servidor, velas en UTC
│   ├── nightly/
│   │   ├── store.py, build_bars.py # almacén en la release y descargas
│   │   ├── daily_layer.py, selection.py
│   │   ├── track_record.py         # historial de alertas sin look-ahead y umbrales z*
│   │   └── run.py                  # pipeline nocturno
│   └── live/
│       ├── sync_results.py, consistency.py, engine.py, alerts.py, runner.py
│       ├── dashboard_data.py       # preparación de datos del dashboard (con tests)
│       └── app.py                  # dashboard Streamlit
├── scripts/
│   ├── inventario_mt5.py           # inventario de símbolos de Darwinex (solo lectura)
│   └── demo_dashboard.py           # escenario sintético para ver el dashboard sin MT5
└── tests/                          # 90+ tests, con mock de MT5
```

Los workflows están en la raíz del repositorio (`.github/workflows/factor-monitor-*.yml`).

## Convenciones

- **Velas**: marca = **apertura**, en **UTC**; cubre `[marca, marca + 5 min)`.
- **Sesiones**: Europa 09:00–17:30 hora de París; EE.UU. 09:30–16:00 hora de Nueva York, con `zoneinfo`.
- **Precio**: histórico de Dukascopy descargado solo con **bid** (desde 2019); el nocturno incremental descarga bid y ask. Los costes de alertas antiguas son solo la comisión.

## Datos de Dukascopy

- Motor por defecto: cliente propio del API JSON `jetta.dukascopy.com`, una petición cada 10 s. Dukascopy limita a unas 3–6 peticiones por minuto por IP y algunos runners de GitHub llegan ya bloqueados: tras ~7 min de 429 el trabajo falla para que al relanzarlo toque otro runner.
- `backfill --resume` solo pide los días que no estén en el almacén y omite los años completos. **Relanzar el workflow completa los huecos.**
- Historia limitada: T-Bond desde dic-2018; Intesa, UniCredit y Eni desde dic-2020; OMV no existe en Dukascopy.

## Darwinex (MT5)

| Instrumento | Símbolo | Nota |
|---|---|---|
| S&P 500 / Nasdaq 100 | `SP500` / `NDX` | |
| Euro Stoxx 50 / DAX / CAC / IBEX | `STOXX50E` / `GDAXI` / `FCHI40` / `SPA35` | |
| EUR/USD | `EURUSD` | |
| Brent | `XTIUSD` | sustituto en vivo (WTI) |
| T-Bond | `TLT` | sustituto en vivo (ETF), solo horario de EE.UU. |
| Bund, acciones europeas | — | solo capa diaria |

Servidor Darwinex: UTC+3 (verano de EE.UU.) / UTC+2 (invierno). El runner lo detecta solo: con tick reciente, o con el mercado cerrado a partir del cierre del forex del viernes (17:00 NY).

## Instalación en el PC (Windows)

Requisitos: Python 3.11+, Git y MetaTrader 5 de Darwinex con la sesión iniciada.

```powershell
git clone https://github.com/dgarciagud/Jules_prueba1.git
cd Jules_prueba1\factor_monitor
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock.txt
pip install -e .[live]
```

Arranque (dos ventanas de PowerShell, ambas con el entorno activado y en `factor_monitor`):

```powershell
# 1) Runner: sincroniza resultados, inventario MT5, hora del servidor y cálculo cada 5 min
python -m factor_monitor.live.runner --repo .. --results results_local --state state --notify

# 2) Dashboard (se abre en el navegador; se refresca cada 60 s)
streamlit run src\factor_monitor\live\app.py -- --results results_local --state state
```

- `--notify` activa las notificaciones de escritorio de Windows.
- Opcional: una tarea programada de Windows que lance el runner a las 08:45 los días laborables.

### Ver el dashboard sin MT5 (demo)

```powershell
python scripts\demo_dashboard.py demo
$env:FM_RESULTS="demo\results"; $env:FM_STATE="demo\state"; $env:FM_UNIVERSE="demo\universe.yaml"
streamlit run src\factor_monitor\live\app.py
```

## Dashboard

| Pestaña | Contenido |
|---|---|
| Régimen | Heatmap de cuotas Shapley (sin el mercado), R² y regresores de la semana, evolución del R² por factor |
| Monitor | Tabla ordenada por \|z\|/z*: z, z*, gap, factor principal, origen, estado (ALERTA / vigilando / filtrado), historial de la celda, disponibilidad en MT5; alertas de hoy |
| Detalle | Real frente a implícito desde la apertura, z con bandas ±z*, descomposición del implícito por factor (ortogonalizados respecto al mercado), eventos del día |
| Estado del sistema | Nocturno (pasos, avisos, fin de datos), runner (hora del servidor, exclusiones, errores), consistencia Dukascopy–MT5, inventario MT5 |

Colores: un color fijo por factor (paleta validada para daltonismo en modo claro y oscuro); cada gráfico tiene tooltips y su tabla.

## Pipeline nocturno

| Paso | Qué hace | Salida en la rama `results` |
|---|---|---|
| `bars` | incremental de Dukascopy (días nuevos + 3 días hábiles) | assets de la release |
| `daily` | capa diaria (solo fechas nuevas + 3 días) | `shapley.parquet`, `daily_r2.parquet` |
| `selection` | selección semanal; en la ejecución completa reconstruye todas las semanas pasadas | `selection.json`, `selection_history.parquet` |
| `track_record` | historial de alertas; completo el primer día hábil del mes, incremental el resto | `track_record_alerts.parquet`, `track_record.parquet`, `intraday_r2.parquet`, `thresholds.json` |
| `recent_bars` | últimas 15 sesiones de velas de 5 min | `recent_bars.parquet` |
| — | estado, fin de datos, versión, avisos | `run_meta.json` |

Si un paso falla, sus ficheros anteriores no se tocan y el job termina en rojo.

## GitHub Actions

1. **Tests** (`factor-monitor tests`): en cada push que toque `factor_monitor/`.
2. **Carga histórica** (`factor-monitor backfill`): instrumentos, años, motor (`jetta`), ritmo (`pace`, 10 s), lados (`bid` o `bid,ask`) y paralelismo (4). Relanzarlo completa lo que falte.
3. **Nocturno** (`factor-monitor nightly`): automático L–V 22:30 UTC; la primera vez, a mano con `full=true`, `force_selection=true` y `years=8`.

## Desarrollo

```bash
cd factor_monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install --no-deps -e .
pytest
```
