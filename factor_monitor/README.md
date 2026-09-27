# factor_monitor — informe diario de sectores y factores

Qué sectores mueven el **S&P 500** y el **CAC 40**, quién va por delante o por detrás de su índice
y qué factor macro (tipos, bonos, petróleo, dólar, semiconductores/IA, oro, volatilidad) mueve cada
sector. Datos de cierre diario; se calcula cada noche en GitHub Actions.

**No ejecuta operaciones.** Mide co-movimiento, no causalidad.

## Estado

| Paso | Contenido | Estado |
|---|---|---|
| 1 | Cierres diarios (Yahoo, Stooq de respaldo), pesos sectoriales, comprobación | hecho |
| 2 | Atribución sectorial (1D…1A) y fuerza relativa | hecho |
| 3 | Shapley por sector: qué factor macro mueve cada sector | en curso |
| 4 | Informe en GitHub Pages y ejecución nocturna automática | pendiente |

El monitor intradía anterior (Dukascopy, MT5, dashboard Streamlit) se eliminó; está en el historial de git.

## Datos

- **Fuente**: API pública `chart` de Yahoo Finance (todo el histórico en una petición por serie); si falla,
  Stooq; si también falla, la copia de la noche anterior (release `daily-data`), marcada como antigua.
- **Universo** (`config/daily_universe.yaml`):
  - S&P 500 (`^GSPC`) con los 11 ETF Select Sector SPDR (XLK, XLF, XLY, XLC, XLV, XLI, XLP, XLE, XLU, XLRE, XLB).
  - CAC 40 (`^FCHI`) con sus 40 componentes, agrupados por sector (composición vigente desde el 22-12-2025;
    **revisar cada trimestre**: marzo, junio, septiembre, diciembre).
  - Factores: tipo a 10 años de EE.UU. (`^TNX`), bonos alemanes largos (ETF `EXX6.DE`), Brent (`BZ=F`),
    dólar (`DX-Y.NYB`), EUR/USD, semiconductores (`SMH`), oro (`GC=F`) y VIX.

## Método

**Pesos sectoriales.** Un índice ponderado por capitalización es una cesta de acciones fijas entre
revisiones: ΔI_t / I_T = Σ w_i · ΔP_i,t / P_i,T. Los pesos w se estiman con el propio índice (mínimos
cuadrados no negativos que suman 1) en las sesiones desde la última revisión y se arrastran por precio
hacia atrás. Recogen el capital flotante y los topes sin necesitar capitalizaciones. En el S&P 500 se
encogen hacia los pesos sectoriales de SPY, porque los ETF limitan el peso de sus mayores valores.
Comprobación en el último año: el CAC 40 se reproduce con R² 0,9999; el S&P 500 con R² 0,97.

**Atribución.** Aportación diaria c_i,t = w_i,t−1 · r_i,t, encadenada sobre el valor del índice al inicio
de cada periodo: sectores + residuo suman exactamente la rentabilidad del índice.

**Fuerza relativa.** (1 + R_sector) / (1 + R_índice) − 1 por horizonte, y su cambio en un mes.

## Uso

```bash
cd factor_monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install --no-deps -e .
pytest

python -m factor_monitor.daily.run --out daily_out            # descarga y análisis
python -m factor_monitor.daily.run --out daily_out --offline  # solo el análisis, sobre una descarga existente
```

Salida en `--out`: `prices.parquet`, `coverage.csv`, `weights_ref.json`, `sectors.parquet`,
`attribution.parquet`, `relative_strength.parquet`, `contributions_daily.parquet`, `data_meta.json`.

## GitHub Actions

- `factor-monitor tests`: en cada push que toque `factor_monitor/`.
- `factor-monitor daily`: manual por ahora (programado en el paso 4). Guarda la salida en la release
  `daily-data` y resume la cobertura, el seguimiento de los pesos y la atribución semanal.
