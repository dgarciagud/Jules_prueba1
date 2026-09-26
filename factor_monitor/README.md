# factor_monitor

Monitor de drivers y dislocaciones intradía. Especificación: [`../SPEC_v3_monitor_factores.md`](../SPEC_v3_monitor_factores.md).

## Estado

| Paso (orden de implementación) | Estado |
|---|---|
| 1. `common/`: sesiones, festivos, horario de verano, factores, modelos | hecho |
| 2. Dukascopy, velas de 5 min, almacén en la release, `backfill` | hecho |
| 3. Capa diaria, selección y `nightly.yml` | hecho |
| 4. Capa intradía en `common/` e historial | pendiente |
| 5. Local: inventario MT5, runner, alertas, consistencia | pendiente |
| 6. Dashboard Streamlit | pendiente |

## Estructura actual

```
factor_monitor/
├── config/universe.yaml            # objetivos, factores, candidatos, IDs verificados
├── src/factor_monitor/
│   ├── common/
│   │   ├── sessions.py             # sesiones en hora local → UTC, velas, franjas de 30 min
│   │   ├── calendars.py            # festivos por bolsa (exchange_calendars)
│   │   ├── universe.py             # carga y validación de universe.yaml, regla del 20 %
│   │   ├── factors.py              # retornos, factores con signo, mercado leave-one-out, ortogonalización, cestas
│   │   ├── models.py               # MCO + HAC, Shapley (LMG), cuotas, ridge con CV por sesiones
│   │   ├── bars.py                 # M1 bid/ask → velas de 5 min, fusión, informe de calidad
│   │   └── runlog.py               # avisos y estado de pasos para run_meta.json
│   ├── sources/dukascopy.py        # dukascopy-node (principal) + lector .bi5 (respaldo)
│   └── nightly/
│       ├── store.py                # almacén versionado en assets de la release `data-store`
│       ├── build_bars.py           # CLI: matrix, backfill, publish, incremental, quality, calibrate
│       ├── daily_layer.py          # cierres de sesión alineados, MCO móvil 60 sesiones, Shapley, betas
│       ├── selection.py            # selección semanal (cuota ≥ 15 %, signo estable 4 semanas, máx. 3)
│       └── run.py                  # pipeline nocturno con aislamiento de fallos por paso
├── tests/
├── package.json / package-lock.json  # dukascopy-node fijado
├── pyproject.toml / requirements-lock.txt
```

Los workflows están en la raíz del repositorio (`.github/workflows/factor-monitor-*.yml`), porque GitHub solo los lee ahí.

## Convenciones

- **Velas**: la marca es la **apertura** de la vela, en **UTC**. Una vela cubre `[marca, marca + 5 min)`.
- **Sesiones**: Europa 09:00–17:30 hora de París; EE.UU. 09:30–16:00 hora de Nueva York. La conversión a UTC usa `zoneinfo`, así que las semanas en que EE.UU. y la UE no han cambiado de hora a la vez salen bien.
- **Mid** = (bid + ask) / 2 de cada campo OHLC; `spread` = spread medio de cierre de los minutos de la vela; `n` = minutos con dato. Se descartan los minutos con ask <= bid.

## Fuentes de Dukascopy

`dukascopy-node` 1.50.0 descarga de `jetta.dukascopy.com` (JSON). El lector `.bi5` de respaldo usa `datafeed.dukascopy.com` (velas diarias de minutos, LZMA, registros `>5i1f`), así que el respaldo es una fuente independiente. Si `dukascopy-node` falla, se usa automáticamente el `.bi5`.

Los IDs de `universe.yaml` están verificados contra los metadatos de `dukascopy-node`. Limitaciones de historia:

| Instrumento | Minutos desde |
|---|---|
| Bund | mayo 2016 |
| T-Bond | diciembre 2018 |
| Intesa, UniCredit, Eni | diciembre 2020 |
| OMV | no existe en Dukascopy (entrará solo con yfinance, en la capa diaria) |

**Pendiente antes de usar el `.bi5` como respaldo**: calibrar `point_factor` por instrumento (valores actuales estimados):

```bash
python -m factor_monitor.nightly.build_bars calibrate --instrument SPX --day 2025-06-03
```

Devuelve el factor sugerido y termina con código 1 si no coincide con el configurado.

## Almacén de velas

Assets de la release `data-store`, un parquet por instrumento y año con nombre versionado (`SPX_2024.<fecha>-<hash>.parquet`) y un `manifest.json`. Orden de escritura: subir ficheros nuevos → subir manifiesto → borrar versiones antiguas. Un fallo a mitad deja el manifiesto anterior válido; si se pierde el manifiesto, se reconstruye con la versión más reciente de cada clave.

## Pipeline nocturno

`factor-monitor nightly` corre de lunes a viernes a las 22:30 UTC (y a mano con *Run workflow*). Pasos:

| Paso | Qué hace | Salida en la rama `results` |
|---|---|---|
| `bars` | incremental de Dukascopy (días nuevos + 3 días hábiles) | assets de la release |
| `daily` | capa diaria; recalcula solo las fechas nuevas + 3 días hábiles | `shapley.parquet`, `daily_r2.parquet` |
| `selection` | solo con datos del último día hábil de la semana (o la primera vez) | `selection.json`, `selection_history.parquet` |
| `recent_bars` | últimas 15 sesiones hábiles de velas de 5 min | `recent_bars.parquet` |
| — | estado, fin de datos, versión, avisos, último éxito por paso | `run_meta.json` |

Si un paso falla, sus ficheros anteriores no se tocan, el error queda en `run_meta.json` y el job termina en rojo (GitHub avisa por email) después de publicar lo demás.

Detalles de la capa diaria:
- El cierre de cada día es la última vela de 5 min de la sesión del objetivo; todos los instrumentos se miden en esa misma marca (tolerancia de 30 min).
- Solo se usan fechas hábiles en la bolsa del objetivo. Un miembro del mercado leave-one-out en festivo queda fuera de la media ese día.
- Los candidatos se ortogonalizan respecto al mercado en cada ventana. La cuota Shapley es sobre el R² sin el mercado.
- Las cestas usan pesos iguales hasta que esté el paso de yfinance. Un día con menos del 80 % del peso disponible queda en NaN, así que BANKS y ENERGY solo tienen historia desde diciembre de 2020 (Intesa, UniCredit y Eni). OMV aún no tiene datos.

## Uso

### GitHub Actions

1. **Tests** (`factor-monitor tests`): en cada push que toque `factor_monitor/`.
2. **Carga histórica** (`factor-monitor backfill`): Actions → *factor-monitor backfill* → *Run workflow*. Parámetros: instrumentos (`all` o lista), primer año, último año y motor (`node` o `bi5`). Crea la release `data-store` si no existe y deja `data_quality.csv` como artefacto.

El workflow necesita `contents: write` (ya declarado) y no usa más secretos que el `GITHUB_TOKEN` automático.

### Local (desarrollo)

```bash
cd factor_monitor
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt && pip install --no-deps -e .
npm ci                                                  # solo si se va a descargar con dukascopy-node
pytest
```

Almacén local en lugar de la release (útil para pruebas):

```bash
python -m factor_monitor.nightly.build_bars backfill --instrument SPX --years 2024 --out out
python -m factor_monitor.nightly.build_bars publish --src out --store-dir .cache/remote
python -m factor_monitor.nightly.build_bars incremental --store-dir .cache/remote
python -m factor_monitor.nightly.build_bars quality --store-dir .cache/remote
```
