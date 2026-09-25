# factor_monitor

Monitor de drivers y dislocaciones intradía. Especificación: [`../SPEC_v3_monitor_factores.md`](../SPEC_v3_monitor_factores.md).

## Estado

| Paso (orden de implementación) | Estado |
|---|---|
| 1. `common/`: sesiones, festivos, horario de verano, factores, modelos | hecho |
| 2. Dukascopy, velas de 5 min, almacén en la release, `backfill` | hecho |
| 3. Capa diaria, selección y `nightly.yml` | pendiente |
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
│       └── build_bars.py           # CLI: matrix, backfill, publish, incremental, quality, calibrate
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
