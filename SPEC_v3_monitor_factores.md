# SPEC v3 — Monitor de drivers y dislocaciones intradía (modelo de factores)
## Arquitectura híbrida: GitHub Actions (nocturno, Dukascopy + Yahoo/Stooq) + PC local (en vivo, MT5 Darwinex)

## Cambios respecto a la v2

| # | Cambio | Motivo | Secciones |
|---|---|---|---|
| 1 | La convergencia del historial se mide sobre un **gap anclado** en el momento de la alerta | Con la suma móvil, a los L intervalos la ventana ya no contiene los residuos que dispararon la alerta y el gap vuelve a 0 por construcción | §9, §12 |
| 2 | El factor IA se elimina de los candidatos del S&P 500 | NDX − SPX contiene al objetivo con peso −1 | §4.3, §5.1 |
| 3 | El S&P 500 no lleva factor de mercado | Europa reacciona al SPX, no al revés, y el modelo cambiaba a mitad de sesión | §5.2 |
| 4 | z con σ por franja de 30 min, varianza Newey-West y umbral calibrado con el historial | Los residuos de 5 min tienen patrón intradía y autocorrelación; un umbral fijo de 2 genera decenas de cruces al día | §7, §8 |
| 5 | Proceso `runner` separado de Streamlit, único dueño de la conexión MT5 | Streamlit no es un planificador y MT5 no admite bien varios hilos | §1, §10.2 |
| 6 | `recent_bars.parquet` en la rama `results` | El test de consistencia necesita velas de Dukascopy en local sin token | §3.5, §10.1 |
| 7 | Almacén en Release con subida versionada, manifiesto y `concurrency` | Borrar y resubir no es atómico; dos jobs pueden pisarse | §3.1, §10.1 |
| 8 | Cestas sectoriales con **yfinance** (pesos, dividendos, ETF de referencia) y **Stooq** como respaldo | Los pesos por capitalización y los dividendos no están en Dukascopy | §3.3, §4.2 |
| 9 | Cuota Shapley definida sobre el R² sin el mercado; reglas de selección explícitas | Con la cuota sobre el R² total casi ningún factor pasaba el filtro | §6 |
| 10 | Filtro de alerta con el R² intradía, no con el diario | Coherencia entre el filtro y el modelo que genera la alerta | §7, §8 |
| 11 | Detección del desfase MT5 robusta con mercado cerrado y convención de marca de vela común | El último tick puede ser antiguo; las velas salían desplazadas entre fuentes | §3.2, §12 |
| 12 | Ventana de bloqueo por tipo de evento y datos europeos añadidos | ±15 min no cubre las ruedas de prensa; faltaban datos relevantes para Europa | §3.7 |
| 13 | Calendario de festivos por bolsa | El CFD cotiza con la bolsa cerrada y contamina el factor de mercado | §3.6 |
| 14 | Clases de origen definidas en la propia spec, enfriamiento sin ambigüedad, agregación jerárquica de celdas, validación fuera de muestra y costes de Darwinex | Ambigüedades de la v2 | §7, §8, §9 |

---

## 0. Objetivo

Construir un monitor que responda cada día a dos preguntas:

1. **¿Qué factores están moviendo cada índice y cada sector?** Es la capa diaria y corre en GitHub Actions cada noche.
2. **¿Qué objetivo se ha separado ahora de lo que implican sus factores?** Es la capa intradía, con velas de 5 minutos, y corre en vivo en el PC local con MT5.

Cada objetivo tiene su propio modelo de factores. La dislocación es el residuo acumulado del modelo y se descompone por factor. El monitor **no ejecuta operaciones**: detecta, explica y lleva el historial de cada alerta.

**Solo fuentes gratuitas:**

| Fuente | Uso | Dónde |
|---|---|---|
| Dukascopy | velas de 5 min y diarias de índices, factores y acciones (CFD) | Actions |
| yfinance | acciones en circulación (pesos de las cestas), dividendos y splits, ETF sectoriales de referencia, respaldo diario de acciones | Actions |
| Stooq | respaldo de precios diarios si Yahoo falla | Actions |
| MT5 (Darwinex) | velas de 5 min en vivo | local |

yfinance es una interfaz no oficial: su versión se fija, sus fallos nunca rompen el pipeline y sus datos se cachean (§3.3).

---

## 1. Arquitectura

```
┌──────────────── GitHub (repo privado) ─────────────────┐
│  Código compartido (src/)                              │
│  GitHub Actions, nocturno L–V:                         │
│    Dukascopy → velas 5m/diarias ┐                      │
│    yfinance/Stooq → pesos, div. ├→ capa diaria         │
│    → selección → historial → umbrales → (PCA, fase 2)  │
│    → publica resultados en la rama `results`           │
└──────────────────────────┬─────────────────────────────┘
                           │ git fetch (rama results)
┌──────────────────────────▼─────────────────────────────┐
│  PC Windows local                                      │
│  MT5 (Darwinex) abierto                                │
│  runner.py: único proceso con MT5; cada 5 min calcula  │
│             capa intradía y alertas → parquet local    │
│  app.py (Streamlit): solo lee y muestra                │
└────────────────────────────────────────────────────────┘
```

Reparto de responsabilidades:

- **GitHub Actions**: todo lo que necesita historia larga y no es urgente (capa diaria, selección, historial de alertas, calibración de umbrales y PCA).
- **Local**: solo lo que necesita tiempo real (estimación intradía con el histórico M5 de MT5, gap, alertas y dashboard).
- **El runner calcula; Streamlit solo lee.** Cerrar el navegador no detiene ni el cálculo ni las alertas.
- Si el job nocturno falla, el monitor local sigue funcionando con la última selección y los últimos umbrales válidos, y muestra un aviso de datos desactualizados.

---

## 2. Estructura del repositorio

```
factor_monitor/
├── .github/workflows/
│   ├── nightly.yml
│   ├── backfill.yml          # manual (workflow_dispatch), carga histórica inicial
│   └── tests.yml             # tests en cada push (con mock de MT5)
├── config/
│   ├── universe.yaml         # objetivos, factores, candidatos, mapeo de símbolos, cestas
│   ├── events.csv            # calendario curado y versionado (§3.7)
│   ├── basket_weights.csv    # pesos trimestrales generados con yfinance (§4.2)
│   └── costs.yaml            # spread y comisión de Darwinex por instrumento
├── src/factor_monitor/
│   ├── common/               # sesiones, festivos, horario de verano, factores, modelos, z, métricas
│   ├── sources/
│   │   ├── dukascopy.py      # solo en Actions
│   │   ├── yahoo.py          # solo en Actions
│   │   ├── stooq.py          # solo en Actions (respaldo)
│   │   └── mt5.py            # solo en local
│   ├── nightly/
│   │   ├── build_bars.py
│   │   ├── baskets.py
│   │   ├── daily_layer.py
│   │   ├── track_record.py
│   │   ├── thresholds.py
│   │   ├── pca_control.py    # fase 2
│   │   └── publish.py
│   └── live/
│       ├── sync_results.py
│       ├── runner.py         # proceso de cálculo en vivo
│       ├── intraday_layer.py # envoltorio fino sobre common/
│       ├── alerts.py
│       ├── consistency.py
│       └── app.py            # Streamlit
├── tests/
├── pyproject.toml
└── README.md                 # instalación local y del workflow paso a paso
```

- Python 3.11+ con `pyproject.toml` y dependencias fijadas. Extras:
  - `[nightly]`: `yfinance` (versión fijada), `pandas-datareader` o cliente propio para Stooq;
  - `[live]`: `MetaTrader5`, `streamlit`, notificaciones de Windows.
- **La lógica de modelos, factores, sesiones, z y alertas vive en `common/`** y la usan el historial y el runner, sin duplicarla. `live/intraday_layer.py` solo adapta la fuente de datos.
- Convención única de velas en todo el código: **marca = apertura de la vela, en UTC**.

---

## 3. Datos

### 3.1 Dukascopy (GitHub Actions)

- M1 bid/ask vía `dukascopy-node` (Node en el runner), agregado a velas de 5 minutos (mid OHLC, spread medio) y a diario. **Solo se guardan las velas de 5 minutos**; M1 se descarta tras agregar.
- **Carga inicial** (`backfill.yml`, manual): desde 2017 o la primera fecha disponible, con una matriz de jobs por instrumento y año para no superar el límite de tiempo.
- **Incremental nocturno**: solo los días nuevos, más los 3 últimos días hábiles para corregir datos tardíos.
- **Almacén**: assets de una GitHub Release (`data-store`), en parquet por instrumento y año. Nunca se sube data al historial de git ni se usa `actions/cache` como almacén principal.
  - `manifest.json` en la release lista cada asset con su nombre versionado, rango y hash.
  - Actualización segura: subir el asset nuevo con sufijo de versión, actualizar el manifiesto y **después** borrar el asset antiguo. Un fallo a mitad deja el manifiesto anterior válido.
  - El incremental solo descarga y resube los ficheros del **año en curso**. Los años cerrados solo se descargan en el recálculo mensual del historial.
- Informe de calidad por instrumento y mes: cobertura, spreads y saltos.
- Si GitHub bloquea o limita las descargas de Dukascopy, se aplican reintentos con backoff y el paso se marca como fallido en `run_meta.json`, sin tocar los datos existentes.

### 3.2 MT5 Darwinex (local)

- Paquete `MetaTrader5` con el terminal abierto y la sesión iniciada. **Solo `runner.py` abre la conexión.**
- **Paso 0 obligatorio: inventario.** Con `symbols_get()`, generar `reports/mt5_inventory.csv` y cruzarlo con `universe.yaml`. Todo objetivo o factor sin símbolo en Darwinex queda **excluido de la capa intradía** (sigue en la capa diaria) y aparece señalado en el dashboard.
- Velas M5 con `copy_rates_range`. Pedir al menos 15 sesiones (se usan 10; el resto cubre festivos). Si el terminal no las devuelve, avisar de que hay que ampliar el parámetro "máx. barras en el gráfico".
- **Hora del servidor**: MT5 devuelve las marcas en la hora del servidor del broker.
  - Detección: tomar un símbolo de referencia con tick reciente (por defecto EURUSD; configurable) y calcular `round((hora_tick − utc_sistema) / 1 h)`. Solo es válido si el tick tiene menos de 60 segundos.
  - Si no hay tick reciente (fin de semana, antes de la apertura), se usa el último desfase guardado en `local_state.json` y se vuelve a detectar en cuanto haya ticks.
  - Se recalcula al arrancar, en cada apertura de sesión y en las fechas de cambio de horario de verano de EE.UU. y de la UE.
  - Todo se convierte a UTC con la convención de marca de §2.

### 3.3 yfinance y Stooq (GitHub Actions)

Uso limitado a datos de referencia de las cestas sectoriales:

1. **Acciones en circulación** (`get_shares_full` o `fast_info`) y precio de cierre, para los pesos por capitalización (§4.2).
2. **Dividendos y splits** (`Ticker.actions`), para limpiar los retornos de los CFD de acciones.
3. **ETF sectoriales de referencia**, diarios, para validar las cestas. Por defecto: iShares STOXX Europe 600 Banks (`EXV1.DE`) y Oil & Gas (`EXH1.DE`). Verificar los tickers.
4. **Respaldo diario de acciones**: si una acción no tiene CFD en Dukascopy, su retorno diario se toma del cierre ajustado de yfinance. Esa acción entra solo en la capa diaria de su cesta, no en el historial intradía.

Reglas:

- Si yfinance falla (errores 429, bloqueos de IP de la nube, cambios de interfaz), se reintenta con backoff; después se prueba **Stooq** para los precios diarios; si todo falla, se usan los últimos valores cacheados y se registra el aviso en `run_meta.json`.
- Caché en la rama `results` (`reference/shares.parquet`, `reference/dividends.parquet`, `reference/etf_daily.parquet`).
- yfinance no se usa para datos intradía ni en vivo: su intradía tiene historia corta (5 min: ~60 días) y retraso.

### 3.4 Mapeo de símbolos

`universe.yaml` define por cada instrumento: `id_logico`, `dukascopy_id`, `mt5_symbol`, `yahoo_ticker` (solo acciones y ETF), `stooq_ticker` (opcional), `tipo` (objetivo/factor/componente), `sesion`, `bolsa` (para festivos) y `signo`. Todos los IDs se **verifican** en su fuente. Los no disponibles se excluyen con aviso, sin romper el pipeline.

### 3.5 Consistencia entre fuentes

Ejecutada en local por `consistency.py` al arrancar y una vez por semana, usando `recent_bars.parquet` de la rama `results` (últimas 10 sesiones de velas de 5 minutos de Dukascopy):

- Correlación de retornos de 5 minutos Dukascopy–MT5 por instrumento en las sesiones solapadas:
  - mayor de 0,95: correcto;
  - entre 0,90 y 0,95: aviso;
  - menor de 0,90: instrumento **no consistente**; su historial de alertas se muestra como "no comparable".
- Correlación con desfases de −2 a +2 velas. Si el máximo no está en 0, se avisa de un **posible error de hora del servidor** (§3.2).

### 3.6 Festivos

`common/` incluye un calendario de festivos por bolsa (`pandas_market_calendars` o equivalente, con Xetra, Euronext París, BME, Eurex y NYSE). Un día festivo en la bolsa de un instrumento:

- ese instrumento no se modeliza como objetivo ese día;
- no entra en el factor de mercado leave-one-out (la media se hace con los restantes);
- sus velas no cuentan para las 10 sesiones de estimación.

### 3.7 Calendario de eventos

`config/events.csv`, **curado y versionado**, es la fuente de verdad. Columnas: `datetime_utc, event_type, region, block_before_min, block_after_min, source`.

| Tipo | Bloqueo por defecto (antes / después) |
|---|---|
| FOMC (comunicado + rueda de prensa) | 15 / 75 min |
| BCE (decisión + rueda de prensa) | 15 / 75 min |
| US_CPI, US_NFP | 15 / 30 min |
| EZ_CPI_FLASH, PMI (eurozona y EE.UU.), IFO, ZEW | 10 / 20 min |
| EIA crudo (semanal) | 10 / 20 min |
| OPEP | todo el día para Brent, energía e índices con candidato Brent |

Reglas:

- La EIA se genera por regla (miércoles 10:30 hora de Nueva York, con excepciones por festivo documentadas a mano).
- `nightly` valida el CSV: fechas en UTC, sin duplicados, cobertura del mes en curso y del siguiente por tipo, columna `source` rellena. Si falta cobertura, avisa en `run_meta.json` y en el dashboard. **Nunca se inventan fechas.**

---

## 4. Universo (fase 1)

### 4.1 Objetivos

- Índices: S&P 500, Euro Stoxx 50, DAX, CAC 40 e IBEX 35.
- Sectores europeos: **bancos** y **energía**, como cestas.

### 4.2 Cestas sectoriales (`nightly/baskets.py`)

- Componentes (5 a 8 por cesta) fijados en `universe.yaml`. Lista orientativa, a verificar en cada fuente:
  - bancos: BNP Paribas, Santander, ING, Intesa Sanpaolo, UniCredit, BBVA, Deutsche Bank, Société Générale;
  - energía: TotalEnergies, Eni, Repsol, Galp, OMV.
- **Pesos por capitalización** con yfinance (acciones en circulación × precio, en EUR), revisados el primer día hábil de cada trimestre y guardados en `config/basket_weights.csv` con fecha y fuente. Peso máximo por componente: 30 % (el exceso se reparte proporcionalmente).
- **Historia**: se usan las acciones en circulación históricas de `get_shares_full` cuando estén disponibles. Si no, los periodos anteriores usan pesos iguales y se marcan en el informe.
- Si yfinance no da datos de un componente, peso igual para toda la cesta ese trimestre, con aviso.
- **Dividendos**: el retorno del CFD en la fecha ex-dividendo se corrige sumando dividendo / cierre previo (fechas e importes de yfinance). Si no hay datos de dividendos, se elimina el retorno de la primera vela del día ex-dividendo y el retorno diario de ese día.
- **Validación**: correlación de retornos diarios de la cesta con su ETF de referencia en los últimos 60 días. Si es menor de 0,85, aviso en `run_meta.json` y en el dashboard.
- Las cestas entran en la capa intradía solo si todos sus componentes con peso mayor del 10 % tienen símbolo en Darwinex. En caso contrario, solo en la capa diaria.

### 4.3 Factores

| Factor | Construcción | Objetivos |
|---|---|---|
| Brent | CFD Brent | según §5.1 |
| Bund | CFD Bund, retorno con **signo invertido** (sube = sube la rentabilidad) | según §5.1 |
| T-Bond | CFD T-Bond, con el mismo tratamiento de signo | según §5.1 |
| Dólar | EUR/USD invertido | según §5.1 |
| IA | retorno Nasdaq 100 − retorno S&P 500 | **solo objetivos europeos**; nunca para el S&P 500 |
| Mercado | §5.2 | todos salvo el S&P 500 |

Si Bund o T-Bond no están en Darwinex, el factor de tipos se usa solo en la capa diaria, y el dashboard indica que el modelo intradía de los objetivos afectados (sobre todo bancos) no incluye tipos.

**Fase 2**: BTP–Bund, oro, cobre, más cestas (autos, lujo, tecnología) y control PCA.

### 4.4 Sesiones

- Europa: 09:00–17:30 hora de París (también para el IBEX, en hora de Madrid, que coincide).
- EE.UU.: 09:30–16:00 hora de Nueva York.
- Ambas convertidas a UTC con ajuste de horario de verano. Cada objetivo se modeliza solo en su sesión.

---

## 5. Modelo de factores (`common/`)

### 5.1 Mapa económico de candidatos (editable en `universe.yaml`)

| Objetivo | Candidatos |
|---|---|
| Índices europeos | Brent, Bund, Dólar, IA |
| S&P 500 | Brent, T-Bond, Dólar |
| Bancos | Bund, T-Bond |
| Energía | Brent, Dólar |

### 5.2 Factor de mercado

| Objetivo | Factor de mercado |
|---|---|
| Sectores | Euro Stoxx 50 |
| Índices europeos de país (DAX, CAC, IBEX) | media de los otros índices europeos del universo (leave-one-out), sin los que estén en festivo |
| Euro Stoxx 50 | media de DAX, CAC e IBEX |
| S&P 500 | **ninguno**: el S&P 500 es el mercado |

Con el mercado incluido, el modelo de cada objetivo es el mismo durante toda su sesión.

### 5.3 Ortogonalización y solapamiento

- Cada factor que no es el mercado se ortogonaliza respecto al mercado del objetivo en la misma ventana de estimación. En el S&P 500, sin mercado, no se ortogonaliza.
- Un factor que no es el mercado queda excluido si contiene al objetivo con más de un 20 % de peso. Se documenta cada exclusión.
- El factor de mercado tiene solapamientos **intencionados** (por ejemplo, el Euro Stoxx 50 contiene a los bancos y a la energía, y buena parte del Euro Stoxx 50 son valores alemanes y franceses). No se excluye, pero su solapamiento aproximado se documenta en el README.
- Interpretación: con factores ortogonalizados, "Brent aporta X pb" significa **el Brent después de descontar el mercado**. El dashboard lo indica.

---

## 6. Capa diaria — GitHub Actions (`nightly/daily_layer.py`)

- Retornos diarios de cierre a cierre de la sesión de cada objetivo, con los factores medidos en la última vela de 5 minutos anterior a ese cierre. Para el S&P 500, cierre a las 16:00 hora de Nueva York.
- Ventana móvil de **60 sesiones**: MCO con mercado + candidatos, descomposición **Shapley (LMG)** del R², betas estandarizadas con su signo y errores HAC (informativos, se muestran en el dashboard y no intervienen en la selección).
- **Cuota Shapley de un factor** = Shapley del factor / (R² − Shapley del mercado). En el S&P 500, sin mercado, = Shapley del factor / R².
- **Selección semanal**: se calcula solo en la ejecución con datos hasta el viernes y es válida de lunes a viernes de la semana siguiente. Las demás noches se mantiene.
  - Un factor se selecciona si su cuota es ≥ 15 % **y** su beta tiene el mismo signo en las 4 últimas estimaciones semanales.
  - Como máximo 3 factores, los de mayor cuota, más el mercado.
  - Si ninguno pasa el filtro, el objetivo queda como "sin driver macro".
- La selección se hace con Dukascopy (y yfinance para las cestas) y la aplica el local con MT5, así que se guarda por **ID lógico**, no por símbolo.

---

## 7. Capa intradía (`common/`, usada por el historial y por el runner)

### 7.1 Estimación (al arrancar y en cada apertura de sesión)

1. Sincronizar los resultados de la noche anterior (§10.2).
2. Cargar las últimas 10 sesiones de velas M5 de los instrumentos necesarios (sin festivos).
3. Construir los factores, ortogonalizados según §5.3, y **estandarizarlos**.
4. Estimar **ridge** por objetivo con los regresores seleccionados; λ por validación cruzada por bloques de sesión dentro de la ventana.
5. Calcular sobre los residuos de la ventana:
   - σ_b: desviación típica de los residuos por **franja de 30 minutos** b de la sesión;
   - ρ_1 y ρ_2: autocorrelaciones de los residuos estandarizados (ε / σ_b);
   - **R² intradía** del ajuste.
6. **Fijar betas, σ_b y ρ para toda la sesión.**

### 7.2 Cada 5 minutos (10 segundos después del cierre de la vela)

- Solo velas **cerradas**. La vela en curso nunca entra.
- Retorno implícito r̂_t = Σ β_f · r_f,t y residuo ε_t = r_t − r̂_t.
- Gap acumulado G_t en L = 6 y L = 12 velas.
- **Varianza del gap** con autocorrelación hasta el retardo 2:
  V_t = Σ_{s∈ventana} σ_b(s)² · [1 + 2 · Σ_{k=1..2} (1 − k/L) · ρ_k]

  Son los pesos exactos de la varianza de una suma de L residuos. Los de Bartlett
  (1 − k/3) la infravaloran con autocorrelación positiva e inflan el z (con ρ₁ = 0,2,
  |z| ≥ 2 salía un 6,5 % de las veces en lugar del 4,6 %).
- **z_t = G_t / √V_t**.
- **Descomposición** del implícito acumulado en la ventana por factor, en pb y en %.
- **Origen**, con I = implícito acumulado en la ventana, R = retorno real acumulado y s_I, s_R sus desviaciones típicas para esa longitud L (estimadas en las 10 sesiones):
  - `driver_moved`: |I| > 1,5 · s_I y |R| < 0,5 · |I|;
  - `target_moved`: |I| < 0,5 · s_I y |R| > 1,5 · s_R;
  - `mixed`: cualquier otro caso.

  Umbrales configurables. Nota: ridge encoge las betas hacia 0, lo que infla ligeramente `target_moved` cuando el factor se mueve mucho; se documenta.
- Sin cálculos de alerta en los primeros 30 minutos de sesión.

---

## 8. Alertas (`common/` + `live/alerts.py`)

### 8.1 Umbral calibrado (`nightly/thresholds.py`)

- Para cada objetivo, z*_objetivo = el cuantil de |z| en el historial de los últimos 250 días (§9) que produce la tasa objetivo de alertas (por defecto, **2 alertas por objetivo y semana**, configurable) tras aplicar los demás filtros.
- Suelo: z* ≥ 2.
- Se publica en `thresholds.json`. Si no hay historial suficiente, z* = 2,5.

### 8.2 Condiciones

Se genera una alerta cuando se cumplen **todas**:

- |z_t| ≥ z*_objetivo con L = 6 o L = 12;
- el **R² intradía** (§7.1) supera la mediana de su propia serie en las últimas 250 sesiones (serie publicada por el historial);
- el objetivo no está marcado como "sin driver macro";
- no se está dentro de la ventana de bloqueo de ningún evento aplicable (§3.7);
- no es la primera media hora de sesión.

### 8.3 Enfriamiento y registro

- Tras una alerta, el objetivo no puede generar otra hasta que se cumpla **lo que ocurra más tarde** de: 30 minutos transcurridos, o z haya cruzado 0.
- Registro local en `alerts_live.parquet`: timestamp UTC, objetivo, L, z, z*, gap, factor principal, origen, betas, descomposición y spread.
- **Notificación de escritorio de Windows** (opcional, activable en la configuración).
- Cada alerta muestra el historial de su celda (§9).

---

## 9. Historial de alertas — GitHub Actions (`nightly/track_record.py`)

- Replica sobre el histórico de 5 minutos de Dukascopy la capa intradía (§7) y las alertas (§8), **con el mismo código de `common/`** y sin look-ahead: selección, umbrales, R² y eventos se usan tal como estaban disponibles en cada fecha.
- **Gap anclado**: para una alerta en t0 con gap G_t0 y longitud L, C_k = G_t0 + Σ_{j=1..k} ε_{t0+j}.
- Por cada alerta histórica se mide, a k = 6 y k = 12 velas (30 y 60 minutos):
  - **convergencia**: |C_k| ≤ 0,5 · |G_t0|;
  - retorno del objetivo en el sentido de la convergencia (signo contrario a G_t0);
  - **quién ajusta**: fracción del cierre del gap debida al objetivo (Σ r) frente al implícito (Σ r̂);
  - **reversión neta de costes**: reversión en pb menos el coste de ida y vuelta de Darwinex (spread + comisión de `costs.yaml`).
- **Agregación jerárquica**: objetivo × factor principal × origen; si la celda tiene menos de 30 alertas, se sube a objetivo × origen, y después a origen. Se muestra el nivel usado.
- **Estabilidad**: además del total, cada celda se reporta por año. Una celda solo se marca como "convergencia persistente" si la reversión neta es positiva en al menos 2 de cada 3 años con datos suficientes.
- Se recalcula el primer día hábil de cada mes. Las demás noches solo se añaden las alertas nuevas.
- Publica también la serie de R² intradía por objetivo (usada en §8.2).

---

## 10. Despliegue

### 10.1 GitHub Actions (`nightly.yml`)

- Disparadores: cron de lunes a viernes a las 22:30 UTC y `workflow_dispatch`.
- `permissions: contents: write` (release y rama `results`). Sin más secretos que el `GITHUB_TOKEN` automático.
- `concurrency: group: data-store, cancel-in-progress: false`, compartido con `backfill.yml`.
- Pasos:
  1. checkout;
  2. instalar Python y Node, y las dependencias `[nightly]`;
  3. descargar el manifiesto y las velas del año en curso (todos los años el día del recálculo mensual);
  4. incremental de Dukascopy;
  5. datos de referencia de yfinance/Stooq y cestas (pesos solo el primer día hábil del trimestre);
  6. capa diaria, selección (solo con datos del viernes), historial y umbrales; PCA en fase 2;
  7. subir las velas actualizadas a la release (§3.1);
  8. publicar los resultados.
- **Resultados** en la rama huérfana `results` (solo ficheros pequeños, un commit por noche):
  - `selection.json`, `thresholds.json`, `shapley.parquet`, `daily_r2.parquet`, `intraday_r2.parquet`, `track_record.parquet`;
  - `recent_bars.parquet` (10 sesiones de velas de 5 min, para §3.5);
  - `reference/` (caché de yfinance/Stooq);
  - `run_meta.json`: fecha de fin de datos, estado de cada paso, versión del código y avisos.
- Si falla un paso, no se sobrescriben sus resultados, se deja constancia en `run_meta.json` y el job se marca como fallido para que GitHub avise por email.
- Presupuesto: registrar la duración de cada paso en `run_meta.json`. En un repo privado el plan gratuito incluye un número limitado de minutos al mes; si el recálculo mensual lo compromete, se reduce a la ventana de los últimos 3 años más los años ya cacheados.

### 10.2 Local

- Clon del repo en el PC. `sync_results.py` hace `git fetch` y extrae la rama `results` a una carpeta local. Se ejecuta al arrancar el runner y en cada apertura de sesión.
- **Control de desactualización**: si la fecha de fin de datos de `run_meta.json` es anterior al último día hábil, se muestra un banner y se usan la última selección y los últimos umbrales válidos.
- `runner.py`: bucle con planificación a la vela (cierre + 10 s), escribe `state/live_*.parquet` y `alerts_live.parquet`, guarda el desfase horario en `local_state.json` y registra errores sin detenerse. Si MT5 se desconecta, reintenta y lo muestra en el estado del sistema.
- `app.py` (Streamlit) solo lee esos ficheros y se refresca cada 60 segundos.
- `README.md` con los pasos: entorno virtual, `pip install -e .[live]`, MT5 con sesión iniciada, `python -m factor_monitor.live.runner` y `streamlit run`.
- Opcional: tarea programada de Windows que lance el runner (y la app) a las 08:45 hora local los días laborables.

---

## 11. Dashboard — local (`live/app.py`, Streamlit)

1. **Régimen** (datos nocturnos): heatmap objetivo × factor con las cuotas Shapley, el R² total y los regresores seleccionados esta semana. Evolución de las cuotas en áreas apiladas para un objetivo elegido. Validación de las cestas frente a su ETF.
2. **Monitor** (en vivo): tabla ordenada por |z| / z* con z, z*, gap en pb, factor principal, origen, estado, historial de la celda (con su nivel de agregación) y disponibilidad en MT5.
3. **Detalle del objetivo**:
   - acumulado real frente al implícito desde la apertura;
   - z con bandas en ±z*;
   - barras de descomposición por factor (con la nota de ortogonalización);
   - eventos del día y ventanas de bloqueo.
4. **Estado del sistema**: fecha del último job nocturno y sus avisos, desfase horario de MT5 (detectado o guardado), estado del runner, test de consistencia entre fuentes, instrumentos excluidos y avisos de yfinance/Stooq.

---

## 12. Tests

**Modelo**

- **Modelo sintético** con betas conocidas: se recuperan en la capa intradía y en la diaria.
- **Dislocación inyectada**: genera alerta, clasifica bien el origen (los tres casos) y atribuye bien la descomposición.
- **Shapley**: la suma de las cuotas LMG es igual al R²; la cuota se calcula sobre el R² sin el mercado.
- **Ortogonalización**: correlación ≈ 0 con el mercado.
- **Solapamiento**: el factor IA nunca aparece en el modelo del S&P 500; el S&P 500 no tiene factor de mercado.

**z y alertas**

- Sobre residuos simulados con patrón intradía y AR(1), la frecuencia de |z| ≥ 2 queda dentro de ±1 punto porcentual del 4,6 % teórico.
- El umbral calibrado produce la tasa objetivo de alertas sobre un historial sintético (tolerancia configurable).
- Enfriamiento: no hay una segunda alerta antes de que se cumplan las dos condiciones de §8.3.

**Historial**

- **Gap anclado**: con residuos i.i.d. sin reversión, la tasa de convergencia es la que corresponde al azar (no cercana al 100 %). Con la suma móvil de la v2, el mismo test mostraría una tasa inflada; se incluye como test de regresión.
- **Paridad nocturno–local**: con los mismos datos de entrada, `track_record` y el runner producen el mismo z y las mismas alertas.
- **Sin look-ahead** en la selección, las betas, σ_b, el z, los umbrales, el R² y el historial.

**Datos y tiempo**

- **Horario de verano**: las sesiones son correctas en las fechas de cambio de EE.UU. y de la UE (incluidas las semanas en que no coinciden).
- **Festivos**: un índice en festivo no entra en el mercado leave-one-out.
- **Hora del servidor MT5** (mock): con desfases simulados de +2 h y +3 h, la conversión es correcta; sin tick reciente se usa el desfase guardado.
- **Convención de velas**: velas de Dukascopy agregadas y velas del mock de MT5 con los mismos datos tienen las mismas marcas.
- **Consistencia**: un desfase de una vela inyectado se detecta con la correlación con desfases.
- **Cestas**: pesos con tope del 30 %; corrección de dividendos sobre una serie sintética con fecha ex-dividendo conocida; respaldo a Stooq y a caché con yfinance simulado caído.
- **Almacén**: un fallo simulado entre la subida y el borrado deja el manifiesto anterior válido.
- **Workflow**: con un fallo simulado de Dukascopy, los resultados anteriores quedan intactos y `run_meta.json` refleja el error.
- **Mock de MT5** para ejecutar todos los tests en GitHub Actions sin terminal.

---

## 13. Criterios de aceptación (fase 1)

- `backfill.yml` carga el histórico y `nightly.yml` corre sin intervención y publica en `results`.
- El runner arranca, sincroniza, genera el inventario MT5, detecta (o recupera) el desfase horario y calcula cada 5 minutos aunque el navegador esté cerrado.
- La app muestra las vistas 1 a 4.
- Cada alerta muestra el historial de su celda con el nivel de agregación.
- Todos los tests pasan en GitHub Actions (con el mock de MT5).
- El README permite reinstalar todo desde cero.

---

## 14. Advertencias

- El modelo mide co-movimiento, no causalidad. "Factor principal" significa principal contribuyente al implícito, y con factores ortogonalizados se refiere a su parte no explicada por el mercado.
- **Una alerta no es una señal de trading.** Solo pasa a validación seria (validación purgada, costes reales) si su celda muestra una convergencia neta de costes persistente entre años.
- Buscar "la celda que funciona" entre muchas celdas tiene sesgo de selección; la regla de estabilidad entre años (§9) lo reduce, pero no lo elimina.
- El historial se calcula con CFDs de Dukascopy y las alertas en vivo con CFDs de Darwinex. El test de consistencia (§3.5) acota la diferencia, pero no la elimina.
- yfinance es una interfaz no oficial y puede fallar o cambiar sin aviso; los pesos y dividendos cacheados pueden quedar desactualizados, y el dashboard lo indica.
- La cobertura de Darwinex puede no incluir bonos ni acciones europeas. En ese caso, el monitor en vivo queda limitado a índices con factores de petróleo, dólar e IA, y los bancos y los tipos solo aparecen en la capa diaria.
