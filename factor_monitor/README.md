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
| 3 | Shapley por sector: qué factor macro mueve cada sector | hecho |
| 4 | Informe en GitHub Pages y ejecución nocturna automática | hecho |

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

**Factores por sector.** En ventanas de 126 sesiones: r_sector = a + b·r_índice + Σ g_f·f⊥ + e, con cada
factor ortogonalizado respecto al índice. El R² se reparte con Shapley (LMG); la cuota macro de un factor
es Shapley_f / (R² − Shapley_índice), con el signo de su beta. Para el índice se usan los factores sin
ortogonalizar. En el CAC 40 (cierra antes que EE.UU.) las rentabilidades son de 2 días solapadas.
Estimaciones a fin de cada mes de los últimos 5 años, para ver cambios de régimen.

## Uso

```bash
cd factor_monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install --no-deps -e .
pytest

python -m factor_monitor.daily.run --out daily_out            # descarga y análisis
python -m factor_monitor.daily.run --out daily_out --offline  # solo el análisis, sobre una descarga existente
python -m factor_monitor.daily.run --out daily_out --offline --site site  # y el informe HTML en site/index.html
```

Salida en `--out`: `prices.parquet`, `coverage.csv`, `weights_ref.json`, `sectors.parquet`,
`attribution.parquet`, `relative_strength.parquet`, `contributions_daily.parquet`, `factor_shapley.parquet`,
`data_meta.json`.

## GitHub Actions

- `factor-monitor tests`: en cada push que toque `factor_monitor/`.
- `factor-monitor daily`: de lunes a viernes a las 22:40 UTC (y a mano). Publica el informe en la rama
  `results`, que sirve GitHub Pages: **https://dgarciagud.github.io/Jules_prueba1/**. Guarda los datos en
  la release `daily-data` y resume cobertura, seguimiento de los pesos, atribución semanal y factores.
  La programación solo corre desde la rama principal del repositorio.

## Informe

Por índice (S&P 500 / CAC 40): resumen de rentabilidades; atribución sectorial por periodo (y las
acciones que más suman y restan en el CAC 40); fuerza relativa por horizonte y gráfico de rotación;
mapa de factores por sector y cambios de régimen (factor dominante ahora, hace 6 y 12 meses).
Colores: azul positivo, rojo negativo, intensidad según el valor; los sectores van siempre con su nombre.
