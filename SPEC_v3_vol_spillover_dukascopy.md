# SPEC v3 — Transmisión de volatilidad y retornos entre activos con datos intradía de Dukascopy

## Cambios respecto a la v2

| # | Cambio | Motivo | Secciones |
|---|---|---|---|
| 1 | Nueva métrica **conectividad dinámica** (solo retardos) junto a la GFEVD total; la conclusión sobre retornos se basa en ella | En retornos sin dinámica, la NPDC normalizada por filas surge de la correlación contemporánea, no de la transmisión | §5.1, §5.2, §8, §9, §11 |
| 2 | `exclude` redefinido: el día de evento se excluye **solo como variable dependiente** y se mantiene como retardo | Con la exclusión total se pierde entre el 36 % y el 59 % de cada ventana | §5.3, §9 |
| 3 | `varx` se declara informativo en volatilidad; en retornos la variante de referencia es `exclude` | Una dummy en la media no absorbe shocks de signo aleatorio | §5.3, §11 |
| 4 | El lead-lag se estima con **ticks** en el episodio y se añade un control de precios rancios | Con barras M1 el retardo casi siempre sale 0, y el precio rancio del CFD crea lead-lag espurio | §2.4, §6.3, §9 |
| 5 | Calendario de festivos por bolsa y filtro de spread por minuto del día | Días festivos con RV casi nula; el filtro rolling simple borra sistemáticamente las franjas de spread ancho | §3 |
| 6 | Rolls por calendario de vencimientos, validados con el salto | La v2 no definía la regla de detección | §3 |
| 7 | Desestacionalización robusta en hora local, con rejilla de 15 min | Perfil hora × día de la semana con ~12 obs. por celda y difuminado por el horario de verano | §4.2, §9 |
| 8 | Presupuesto de cómputo del cuantílico (paso semanal, λ fijo por ecuación) y τ = 0,05 en retornos | ~4 millones de regresiones por variante con paso diario y CV por ventana | §5.4 |
| 9 | Horizonte espectral propio para la conectividad en frecuencia | H = 10 no resuelve la banda de más de 20 días | §5.4 |
| 10 | HAR-VAR como robustez; política explícita ante rechazo de KPSS | log RV tiene memoria larga | §4.1, §5.5 |
| 11 | Malla de robustez de un factor cada vez sobre un caso base | La malla completa es combinatoriamente inviable | §5.5 |
| 12 | Calendario de eventos curado y versionado como fuente de verdad | El scraping de la OPEP o del BoJ es frágil y rompe `run-all` | §2.5, §10 |
| 13 | Regla de retroceso para la ventana de control; tests corregidos (parámetros verdaderos, κ = 1, 52 intervalos, ruido de microestructura) | Ambigüedades o tests inconsistentes en la v2 | §6.1, §9 |
| 14 | Política de NaN conjunta, muestra por variante, semillas, log estructurado y manifiesto de datos | Reproducibilidad | §1, §3, §10 |

---

## 0. Objetivo

Construir un pipeline reproducible que responda a dos preguntas:

- **Marco de fondo (diario)**: qué activos son emisores netos estructurales de volatilidad y de retornos hacia los índices bursátiles, y cómo cambia eso en la cola (volatilidad anormal).
- **Episodio en curso (intradía)**: cuando un activo entra en volatilidad anormal, quién lidera la transmisión **ahora**, usando datos de 15 minutos y ticks dentro del episodio.

El pipeline debe:

1. Descargar datos M1 bid/ask de Dukascopy (y ticks para la ventana del episodio).
2. Limpiar y sincronizar los datos, y construir RV diaria, RV intradía desestacionalizada y retornos diarios.
3. Estimar la conectividad con los métodos rolling Diebold-Yilmaz, TVP-VAR, cuantiles y frecuencia, **con y sin control de eventos macro**, sobre volatilidad y sobre retornos, distinguiendo conectividad **total** y **dinámica**.
4. Ejecutar un módulo de episodio intradía con VAR de alta frecuencia y lead-lag.
5. Generar informes y validar todo con datos sintéticos.

Lenguaje de resultados: "transmite/emite". Nunca "causa" (ver §11).

---

## 1. Stack y estructura

- Python 3.11+ con `pyproject.toml` y dependencias fijadas (lockfile).
- Node.js ≥ 18 con versión fijada de **`dukascopy-node`** (`package.json` + `package-lock.json`).
- Librerías Python: `pandas`, `numpy`, `pyarrow`, `statsmodels`, `scipy`, `scikit-learn` (LASSO), `pandas_market_calendars` (festivos), `matplotlib`, `pyyaml`, `typer`, `pytest`.
- Descarga con `dukascopy-node` vía `subprocess`. Si falla, fallback con lector propio de `.bi5` (§2.4).

```
vol_spillover/
├── config/
│   ├── config.yaml
│   └── events.csv           # calendario macro curado y versionado (§2.5)
├── data/{raw,clean,features}/
├── data/manifest.json       # hash y rango por fichero descargado
├── src/vol_spillover/
│   ├── download.py
│   ├── events_calendar.py
│   ├── calendars.py         # festivos y sesiones por bolsa, horario de verano
│   ├── clean.py
│   ├── realized.py          # RV diaria, RV intradía desestacionalizada, retornos
│   ├── var_utils.py         # VAR/VARX/HAR-VAR, GFEVD total y dinámica, GIRF, retardos
│   ├── connectedness/
│   │   ├── dy_rolling.py
│   │   ├── tvp_var.py
│   │   ├── quantile.py
│   │   └── frequency.py
│   ├── episode/
│   │   ├── detect.py
│   │   ├── intraday_var.py
│   │   └── leadlag.py
│   ├── events_study.py
│   ├── plots.py
│   └── cli.py
├── tests/
└── reports/
```

CLI:

```
vs download | vs events | vs clean | vs realized
vs connect  --method {dy,tvp,quantile,freq,all} --target {vol,ret,both} --events {none,varx,exclude,all}
vs robustness                                     # malla de un factor cada vez (§5.5)
vs episode  [--asset WTI] [--start YYYY-MM-DD]    # autodetecta si no se indica
vs report | vs run-all
```

Requisitos transversales:

- Todo es idempotente, con caché en parquet y configuración única en YAML.
- Semilla global configurable para bootstrap, CV y simulaciones.
- Log estructurado (JSON por línea) en `reports/run_log.jsonl`, con un evento por exclusión, filtro, roll, estimación no fiable y fuente de calendario.
- `data/manifest.json` registra, por fichero bruto, instrumento, rango, fecha de descarga, versión de `dukascopy-node` y hash.

---

## 2. Datos

### 2.1 Universo

**Universo núcleo** (el que se usa en todos los modelos). Tiene un activo por bloque para evitar que la conectividad intra-bloque domine.

| Bloque | Activo | ID Dukascopy (verificar) | Rol hipotético |
|---|---|---|---|
| Índice EE.UU. | S&P 500 | `usa500idxusd` | receptor |
| Índice Europa | Euro Stoxx 50 | `eusidxeur` | receptor |
| Índice Japón | Nikkei 225 | `jpnidxjpy` | receptor |
| Energía | WTI | `lightcmdusd` | candidato |
| Tipos | US T-Bond | `ustbondtrusd` (probable) | candidato |
| Dólar | Índice dólar | `dollaridxusd` | candidato |
| Refugio | Oro | `xauusd` | candidato |

El "rol" es una hipótesis de trabajo que organiza los informes (candidato → índice). **El método no la impone** y los resultados pueden contradecirla; si ocurre, se informa tal cual.

**Universo extendido** (solo para robustez y **por sustitución**, nunca a la vez que su equivalente del núcleo):

- Nasdaq 100 (`usatechidxusd`) en lugar de SPX;
- DAX (`deuidxeur`) en lugar de SX5E;
- Brent (`brentcmdusd`) en lugar de WTI;
- Bund (`bundtreur`, probable) en lugar de T-Bond;
- EUR/USD o USD/JPY en lugar del índice dólar.

Reglas:

- El agente DEBE verificar todos los IDs contra la lista de instrumentos de la versión fijada de `dukascopy-node` antes de descargar.
- Si el índice dólar no está disponible, se usa EUR/USD invertido como proxy y se documenta.
- Un instrumento inexistente o con historia insuficiente se registra en el log y se excluye, sin romper el pipeline.
- **Muestra por variante**: cada combinación de universo usa su propia primera fecha común. Se reporta el rango efectivo de cada variante junto a sus resultados.
- Se espera que la historia común empiece hacia 2017–2018. Con los periodos de arranque (200 días DY/TVP, 500 cuantílico) los resultados útiles empiezan 1–2 años después; el informe lo indica.

### 2.2 Parámetros de descarga

- Timeframe `m1`, precios **bid y ask** (dos descargas por bloque, una por tipo de precio, que luego se combinan por marca temporal), marcas temporales en UTC y almacenamiento en parquet.
- Se guardan OHLC de bid y de ask. El mid del minuto se construye con los cierres: `mid = (bid_close + ask_close) / 2`.
- Rango: desde la primera fecha disponible por activo hasta ayer. Reportar el inicio efectivo por activo.
- Descarga por bloques mensuales, con reintentos (backoff exponencial, máximo 5) y pausa configurable entre peticiones.
- Sin relleno artificial de huecos.
- **Ticks**: solo para la ventana del episodio y su ventana de control (§6), por horas, desde los ficheros `.bi5` de ticks (§2.4).

```
npx dukascopy-node -i usa500idxusd -from 2016-01-01 -to 2016-01-31 -t m1 -p bid -f csv -dir data/raw/usa500idxusd/bid
npx dukascopy-node -i usa500idxusd -from 2016-01-01 -to 2016-01-31 -t m1 -p ask -f csv -dir data/raw/usa500idxusd/ask
```

(Ajustar a la sintaxis real de la versión fijada.)

### 2.3 Calidad de datos

Generar `reports/data_quality.csv` por instrumento y año con:

- % de minutos cubiertos y días con cobertura inferior al 50 %;
- spread mediano y p99 en puntos básicos;
- saltos mayores de 10 MAD en retornos de 1 minuto;
- tramos de precio constante de más de 30 minutos en horario líquido;
- **fracción de minutos con retorno cero en horario líquido** (indicador de precio rancio, usado en §6.3).

### 2.4 Fallback `.bi5`

Base: `https://datafeed.dukascopy.com/datafeed/{SYMBOL}/...`. El **mes va indexado desde 0**.

- **Velas M1 (fallback de la descarga histórica)**: `{YYYY}/{MM}/{DD}/BID_candles_min_1.bi5` y `ASK_candles_min_1.bi5`. Un fichero por día. Registros de 24 bytes con formato `>5i1f`: segundos desde el inicio del día, open, close, low y high como enteros, y volumen. Verificar el orden de los campos con un día conocido.
- **Ticks (módulo de episodio)**: `{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5`. Registros de 20 bytes con formato `>3i2f`: ms desde el inicio de la hora, ask, bid, volumen ask y volumen bid.
- Compresión LZMA en ambos casos. Un fichero vacío es una hora o día sin datos, no un error.
- Precio = entero / factor de escala del instrumento. Calibrar el factor contra un precio conocido y guardarlo en `config.yaml`.

### 2.5 Calendario de eventos macro (`events_calendar.py`)

`config/events.csv`, **curado y versionado en el repositorio**, es la fuente de verdad. Columnas: `datetime_utc, event_type, region, tier, source`.

Tipos de evento:

| Tipo | Contenido | Tier |
|---|---|---|
| `FOMC` | comunicado y rueda de prensa | 1 |
| `US_CPI` | IPC de EE.UU. | 1 |
| `US_NFP` | nóminas no agrícolas | 1 |
| `ECB` | decisión y rueda de prensa | 2 |
| `BOJ` | decisión de política monetaria | 2 |
| `OPEC` | reuniones ministeriales OPEP/OPEP+ | 2 |

Reglas:

- `vs events` **valida** el CSV: fechas en UTC, sin duplicados, cobertura del rango de la muestra por tipo (número de eventos por año dentro de un rango esperado configurable), columna `source` rellena.
- `vs events --update` intenta **ampliar** el CSV con fuentes oficiales (Fed, BLS, BCE, BoJ, OPEP). Nunca sobrescribe filas existentes. Es una utilidad auxiliar, no forma parte de `run-all`.
- Si falta cobertura para parte de la muestra, se emite un aviso claro y ese tramo se marca como "sin control de eventos" en los informes. **Nunca se inventan fechas.**
- La hora en UTC de cada fila debe reflejar el horario de verano del emisor (CPI y NFP: 13:30 UTC en invierno, 12:30 UTC en verano).
- Se derivan:
  - dummies **diarias** por tipo, asignadas al día común de 22:00 a 22:00 UTC que contiene el evento;
  - dummies **intradía** por intervalo de 15 minutos (el intervalo del anuncio y el siguiente).

---

## 3. Limpieza y sincronización

1. **Filtro de spread**: descartar minutos con `ask <= bid` o con un spread mayor que k × la **mediana del spread para ese mismo minuto del día (en hora UTC) en los 20 días hábiles previos** (k = 10, configurable). Así no se borran sistemáticamente las franjas de spread ancho (cambio de sesión, 22:00–23:00 UTC).
2. **Día común de 22:00 a 22:00 UTC** para todos los activos. Como los CFD cotizan casi 24 h, el mid a las 22:00 es un precio sincrónico. Esto elimina el lead-lag espurio por cierres asíncronos **en los retornos diarios**. No lo elimina en la RV de sesión (§5.5).
3. **Máscaras de sesión** por activo (`calendars.py`), definidas en **hora local** y convertidas a UTC con horario de verano:

   | Activo | Sesión (hora local) |
   |---|---|
   | SPX, NDX | 09:30–16:00 Nueva York |
   | SX5E, DAX | 09:00–17:30 Fráncfort |
   | Nikkei | 09:00–15:00 Tokio (con pausa de mediodía según la bolsa) |
   | WTI, Brent | 09:00–14:30 Nueva York (pit de energía) |
   | T-Bond | 08:20–15:00 Nueva York |
   | Bund | 08:00–17:15 Fráncfort |
   | Oro | 08:20–13:30 Nueva York |
   | Índice dólar, EUR/USD, USD/JPY | 08:00–17:00 Londres |

   Se usan para la RV de sesión (robustez) y para el módulo intradía (§6).
4. **Días excluidos**:
   - fines de semana (el tramo del domingo desde las 22:00 UTC pertenece al lunes);
   - días con cobertura inferior al umbral (50 %, configurable);
   - 24–26 de diciembre y 31 de diciembre–1 de enero (configurable);
   - **festivos de la bolsa de referencia de cada activo** (`pandas_market_calendars`): ese activo queda en NaN ese día aunque el CFD haya cotizado.
5. **Rolls** de CFDs sobre futuros (WTI, Brent, T-Bond, Bund):
   - Se genera el **calendario de rolls esperado** a partir del calendario de vencimientos (mensual para crudo, trimestral para bonos) y se busca el salto en una ventana de ±2 días hábiles alrededor de cada fecha.
   - Se acepta como roll el mayor salto de 1 minuto de esa ventana si supera 10 MAD; si no hay salto, se registra "roll no detectado" en el log.
   - Se **elimina solo el retorno de 1 minuto del roll**: la RV del día se calcula sin él y el retorno diario se recalcula restando ese salto. La serie de precios no se ajusta.
   - Si la configuración de Dukascopy no produce saltos de roll para un instrumento (ajuste en caja), se documenta y el paso se omite para ese instrumento.
   - Las fechas detectadas se guardan en `reports/rolls.csv`.
6. **Política de NaN conjunta**: los modelos multivariantes usan solo filas en las que todos los activos del modelo tienen dato (en `exclude`, ver §5.3). Las filas descartadas se cuentan por motivo en el log.

---

## 4. Variables

### 4.1 Diarias

- **RV submuestreada**: media de las RV a 5 minutos con los 5 desfases posibles sobre el mid de 1 minuto (con precio previo si falta un minuto). Guardar también la bipower variation (BV) y el componente de saltos J = max(RV − BV, 0).
- Variable de volatilidad: `log(RV)` anualizada.
- Tests ADF y KPSS reportados por activo. **Política**: el modelo se estima en niveles de log RV aunque KPSS rechace (práctica estándar en la literatura de spillovers). Si ADF no rechaza la raíz unitaria para algún activo, se documenta y se reporta como robustez la variante HAR-VAR (§5.5).
- **Retornos diarios**: log-retorno de cierre a cierre sobre el día común de 22:00 a 22:00 UTC (mid), con el ajuste de roll de §3.5. El retorno del lunes incluye el fin de semana.
- **Flag de vol anormal**: `log RV` > percentil 95 rolling de 250 días, calculado solo con datos hasta t − 1.

### 4.2 Intradía (para el §6)

- Intervalos de **15 minutos**.
- RV del intervalo = suma de los retornos de 1 minuto al cuadrado. Variable: `log(RV_15m + ε)`, con ε igual al percentil 1 de las RV positivas del activo en la ventana de estimación del perfil.
- **Desestacionalización obligatoria** (sin ella, el patrón diurno domina la conectividad):
  - Rejilla en **hora local de la sesión de referencia del activo**, para que el cambio de horario de verano dentro de la ventana no difumine el perfil.
  - Perfil = **mediana** de `log(RV_15m + ε)` por intervalo de 15 minutos + efecto aditivo por día de la semana (mediana de las desviaciones por día de la semana).
  - Alternativa configurable: forma Fourier flexible (Andersen-Bollerslev) con dummies de día de la semana.
  - Estimado con los 60 días hábiles previos al inicio del episodio, sin incluirlo, excluyendo días de evento de tier 1.
  - Se resta el perfil tanto en el episodio como en la ventana de control.
- Retornos de ticks (mid) en el episodio y la ventana de control para el lead-lag (§6.3). Retornos de 1 minuto como alternativa si no hay ticks.

---

## 5. Conectividad diaria (marco de fondo)

### 5.1 Base común (`var_utils.py`)

- VAR(p) o **VARX(p)** con dummies de eventos como variables exógenas. p se elige por BIC entre 1 y 5 sobre la muestra completa y se mantiene fijo en todas las ventanas (configurable).
- **GFEVD** (Pesaran-Shin) a H = 10.
- **Dos versiones de la GFEVD**:
  - **Total**: la GFEVD estándar a H, normalizada por filas.
  - **Dinámica**: la contribución de los horizontes 1…H−1, excluyendo el impacto contemporáneo (h = 0). En el numerador se suma solo sobre h ≥ 1 y se divide por la **misma suma de fila que la GFEVD total** (no se renormaliza por su propia fila). Así, total = impacto + dinámica elemento a elemento, y la dinámica es ≈ 0 cuando no hay transmisión con retardo. Mide la parte de la transmisión que viaja con retardo.
- Métricas para ambas versiones: TCI, TO, FROM, NET y **NPDC_ij** (conectividad direccional neta por pares), que es la métrica central hacia cada índice receptor.
- **Signo con GIRF**: la GFEVD no tiene signo. Para cada par candidato → índice se reportan:
  - la GIRF en h = 0 (impacto);
  - la GIRF acumulada de h = 1 a H (dinámica).

  Esto es clave en retornos, donde importa si el petróleo arrastra al índice al alza o a la baja.

### 5.2 Dos objetivos

`--target vol` estima sobre `log RV` y `--target ret` sobre retornos diarios. Ambos usan exactamente la misma maquinaria.

**Regla de interpretación**:

- En retornos, las conclusiones sobre quién emite hacia quién se basan en la **NPDC dinámica**. La NPDC total de retornos se reporta pero se etiqueta como "dominada por la correlación contemporánea": con A_h ≈ 0 para h ≥ 1, la GFEVD total es ≈ ρ²_ij y la normalización por filas genera una NPDC distinta de cero a favor del activo más correlacionado con el resto, sin que haya transmisión temporal.
- En volatilidad se reportan ambas. Si la NPDC total y la dinámica discrepan en signo, el informe lo señala expresamente.

### 5.3 Tratamiento de eventos macro

Tres variantes, todas reportadas:

- `none`: VAR sin controles.
- `varx`: VARX con dummies de eventos por tipo.
- `exclude`: los días de evento (tier configurable, por defecto todos) **no se usan como variable dependiente**, pero **sí como retardos** de los días siguientes. Se implementa eliminando filas de la matriz de regresión ya construida, no poniendo NaN en la serie. Para la covarianza de los residuos se usan solo las filas restantes.

Alcance de cada variante:

| Objetivo | Variante de referencia para el control macro | Motivo |
|---|---|---|
| Volatilidad | `varx` (y `exclude` como contraste) | El evento desplaza el nivel de log RV de todos los activos a la vez; una dummy en la media lo absorbe |
| Retornos | `exclude` | El efecto del evento tiene signo aleatorio; una dummy en la media apenas lo absorbe y el shock común queda en la covarianza de los residuos |

Métrica de diagnóstico: el cambio en NET y NPDC de cada activo entre `none` y la variante de referencia de su objetivo. Un activo cuya transmisión cae mucho con el control está captando shocks comunes de calendario, no transmisión propia.

Se reporta, por ventana, el número de filas usadas en cada variante.

### 5.4 Métodos

- **Rolling Diebold-Yilmaz**: ventana de 200 días y paso de 1 día. Es la referencia.
- **TVP-VAR** (Antonakakis, Chatziantoniou y Gabauer, 2020): filtro de Kalman con κ1 = κ2 = 0,99 y prior de las primeras 200 observaciones, que se descartan. Es el método principal del marco de fondo.
  - Dummies exógenas: por defecto, **residualización previa**: cada serie se regresa sobre las dummies con una ventana rolling de 500 días que solo usa datos pasados, y el TVP-VAR se estima sobre los residuos. Alternativa configurable: dummies dentro del estado.
- **Cuantiles** (Ando, Greenwood-Nimmo y Shin, 2022):
  - τ ∈ {0,05; 0,50; 0,95} para ambos objetivos. Casos de interés: **τ = 0,95 en volatilidad** y **τ = 0,05 en retornos**.
  - Por defecto, regresión cuantílica penalizada con LASSO y ventana de 500 días. Alternativa: sin penalización con ventana mínima de 500.
  - **Presupuesto de cómputo**:
    - λ se calibra **una vez por ecuación y cuantil** con validación cruzada en bloques temporales sobre las primeras 750 observaciones y se mantiene fijo; se recalibra cada 250 días.
    - Paso de la ventana: **5 días** (configurable) y warm start desde la ventana anterior.
    - `vs connect --method quantile` informa del número estimado de regresiones antes de empezar.
  - **Control de observaciones de cola**: si 500 × min(τ, 1 − τ) dividido por los parámetros no nulos de la ecuación es menor que 3, la estimación se marca como **no fiable** en los resultados y en los gráficos (trama rayada). Con la especificación por defecto (7 variables, p = 2, 25 observaciones de cola) este control solo se supera si LASSO deja 8 parámetros no nulos o menos; es esperable que se marque a menudo, y el informe lo dice.
- **Frecuencia** (Baruník y Křehlík, 2018): bandas de 1–5, 5–20 y más de 20 días. Horizonte espectral **H_freq = 100** (configurable, independiente del H de la GFEVD), sobre las mismas ventanas que el rolling DY.

### 5.5 Robustez

Se define un **caso base**:

> rolling DY y TVP-VAR, H = 10, ventana 200, RV total de 22:00 a 22:00 UTC, universo núcleo, VAR(p) en niveles, variantes de eventos `none` + referencia.

Y se varía **un único factor cada vez**:

| Factor | Valores alternativos |
|---|---|
| Horizonte | H ∈ {5, 20} |
| Ventana (DY) | {100, 300} |
| Medida de RV | RV de sesión (máscaras de §3.3); BV (RV continua) en lugar de RV total |
| Especificación | HAR-VAR (retardos 1, 5 y 22 de cada variable) |
| Universo | cada sustitución del §2.1, una a una |

La RV de sesión reintroduce la asincronía entre regiones (la sesión del Nikkei del día t precede a la de EE.UU. del mismo día). Su resultado se lee como contraste, no como sustituto.

`vs robustness` produce una tabla con el emisor neto principal hacia cada índice en cada fila de la malla y marca las filas en las que cambia respecto al caso base.

---

## 6. Módulo de episodio intradía

### 6.1 Detección (`episode/detect.py`)

- Episodio activo: un candidato tiene flag de vol anormal en al menos 3 de los últimos 5 días.
- Inicio: el primer día del tramo continuo con flag. También se admite un inicio manual por CLI.
- Ventana de estimación: desde el inicio hasta el último día disponible, con un mínimo de 10 días hábiles. Si hay menos, se avisa y se usan 10 días de todas formas (incluyendo días previos al inicio, lo que se indica en el informe).
- **Ventana de control**: N días hábiles (N = longitud de la ventana de estimación) **sin flags de ningún activo**:
  1. Se busca el bloque contiguo de N días sin flags que termine lo más cerca posible del inicio del episodio, retrocediendo hasta 120 días hábiles.
  2. Si no existe, se acepta un bloque no contiguo: los N días sin flags más cercanos dentro de esos 120 días.
  3. Si tampoco hay N días sin flags, se usan los N días con menos activos con flag y se marca la comparación como **control contaminado** en el informe.

### 6.2 VAR intradía (`episode/intraday_var.py`)

- Variables: `log RV_15m` desestacionalizada de SPX, SX5E y los candidatos del núcleo.
- **El Nikkei se excluye del módulo intradía**, porque su sesión no se solapa con la de EE.UU. Se analiza solo en el marco diario.
- Horario común: de 07:00 a 20:00 UTC en invierno, desplazado una hora con el horario de verano (52 intervalos por día).
- **Sin retardos entre días**: la matriz de regresión se construye día a día; las primeras p observaciones de cada día no se usan como variable dependiente.
- p por BIC entre 1 y 4.
- VARX con las dummies intradía de eventos (§2.5).
- GFEVD total y dinámica con H = 8 intervalos (2 horas), junto con NET, NPDC y signo GIRF (impacto y acumulado).
- Se estima en el episodio y en la ventana de control, y se reporta la diferencia. Intervalos de confianza por bootstrap en bloques de días.

### 6.3 Lead-lag de alta frecuencia (`episode/leadlag.py`)

- Datos: **ticks** (mid) del episodio y de la ventana de control. Si los ticks no están disponibles, se usan retornos de 1 minuto y el informe avisa de que la resolución es de 1 minuto.
- Estimador de Hoffmann, Rosenbaum y Yoshida (2013), basado en Hayashi-Yoshida. Es robusto a la asincronía de los ticks.
- Pares: cada candidato frente a SPX y frente a SX5E, solo en el horario común de §6.2.
- Rejilla de retardos:
  - con ticks: de −60 a +60 s en pasos de 1 s, y de −30 a +30 min en pasos de 1 min;
  - con M1: de −30 a +30 min en pasos de 1 min.
- Reportar:
  - el retardo que maximiza la correlación absoluta;
  - el **ratio de asimetría** (suma de correlaciones al cuadrado en retardos positivos frente a negativos), que es la métrica principal. Se espera que el retardo óptimo sea 0 o casi 0 en la mayoría de pares;
  - ambos en el episodio y en la ventana de control.
- **Control de precio rancio**: para cada activo, fracción de ticks o minutos sin cambio de mid en el horario común. Si el activo "seguidor" de un par tiene una fracción claramente mayor (umbral configurable) que el "líder", el resultado se marca como **posiblemente debido a precio rancio del CFD**.
- Intervalos de confianza por bootstrap en bloques de días. Con 10 días el número de bloques es pequeño; el informe indica que los intervalos son orientativos.

---

## 7. Estudio de episodios históricos (`events_study.py`)

- Para cada inicio de episodio histórico de un candidato (índices sin flag en t − 1), calcular la respuesta de `log RV` y del retorno acumulado de cada índice en t, …, t + 10, frente a días de control emparejados por nivel de vol previo.
- Separar los episodios que coinciden con eventos macro de los que no.
- Esta sección solo produce las respuestas medias y sus intervalos. El ranking mensual del emisor dominante se construye una sola vez en §8.3.

---

## 8. Salidas (`reports/`)

1. TCI en el tiempo: TVP frente a DY, cuantil 0,50 frente a cola (0,95 en vol y 0,05 en retornos), `none` frente a variante de referencia, y total frente a dinámica.
2. NET por activo en el tiempo, en volatilidad y en retornos (total y dinámica).
3. **NPDC de cada candidato hacia cada índice**, con el signo GIRF (impacto y acumulado): áreas apiladas y **tabla del emisor dominante por mes** (única fuente del ranking mensual). En retornos, la tabla se basa en la NPDC dinámica.
4. Heatmaps de conectividad: muestra completa, últimos 60 días y por bandas de frecuencia.
5. **Informe de episodio** (`episode_YYYYMMDD.md` y gráficos):
   - NPDC intradía en el episodio frente a la ventana de control, con intervalos de confianza;
   - perfiles de lead-lag con sus intervalos de confianza y el control de precio rancio;
   - calidad de la ventana de control (contigua, no contigua o contaminada);
   - conclusión en texto sobre quién lidera y con qué signo.
6. Tabla de impacto de los controles macro (§5.3), con el número de filas usadas por variante.
7. Calidad de datos, fechas de roll, festivos aplicados y calendario de eventos usado (con cobertura por tipo y año).
8. Tabla de robustez de un factor cada vez (§5.5).
9. Estudio de episodios históricos (§7).
10. **`summary.md`** con dos bloques:
    - **Fondo**: principal emisor neto hacia SPX, SX5E y Nikkei, en volatilidad (total y dinámica) y en retornos (dinámica), en mediana y cola, con y sin control macro. Incluye cuántas filas de la malla de robustez confirman el resultado.
    - **Ahora**: si hay un episodio activo, quién lidera intradía y con qué signo, con los avisos que procedan (control contaminado, precio rancio, resolución de 1 minuto). Si no lo hay, indicarlo.

---

## 9. Tests

**Maquinaria VAR**

- **GFEVD analítica**: con los **parámetros verdaderos** de un VAR(1) donde A transmite a B y C, la función de GFEVD reproduce la solución analítica (tolerancia 1e-10).
- **Recuperación estadística**: sobre datos simulados de ese VAR(1) (T = 2.000), NET_A > 0 y NPDC_A→B > 0 en al menos el 95 % de 200 réplicas.
- Las filas de la GFEVD total suman 1; total = impacto + dinámica elemento a elemento; la suma de los NET es 0 en ambas versiones.
- **Artefacto de normalización**: con un VAR sin dinámica (A = 0) y covarianza en la que un activo está más correlacionado con el resto, la NPDC total es distinta de cero y la NPDC dinámica es ≈ 0 (tolerancia configurable).
- **Signo GIRF**: con un coeficiente de transmisión negativo conocido, el signo acumulado reportado es negativo.

**Eventos**

- **Shock común en la media** (volatilidad): un shock exógeno simultáneo en el nivel de todas las series. Sin VARX aparece conectividad espuria; con la dummy como exógena, la conectividad cae a su valor verdadero (tolerancia configurable).
- **`exclude`**: con p = 2 y un 20 % de días de evento, la fracción de filas usadas es ≈ 80 %, no ≈ 49 %; ninguna fila con dependiente en día de evento entra en la regresión.

**Métodos**

- TVP-VAR con **κ1 = κ2 = 1** sobre datos de un VAR con parámetros constantes converge al VAR estático (tolerancia configurable). Con κ = 0,99, los coeficientes permanecen dentro de una banda configurable alrededor de los verdaderos.
- El cuantil 0,5 **sin penalización** sobre datos gaussianos se aproxima al rolling DY (tolerancia configurable).
- La conectividad en frecuencia suma, entre bandas, la conectividad total a H_freq (tolerancia 1e-6).

**Variables**

- RV submuestreada sobre un browniano simulado: sesgo < 2 %. Con ruido de microestructura i.i.d. añadido (nivel configurable), el sesgo de la RV submuestreada es menor que el de la RV a 1 minuto.
- **Desestacionalización**: una serie con patrón diurno sintético en hora local (que cruza un cambio de horario de verano) más ruido i.i.d. debe quedar sin autocorrelación significativa en el retardo de un día de la rejilla (**52 intervalos**) y sin diferencia de medias por intervalo.

**Episodio**

- **Lead-lag con ticks**: dos procesos con retardo conocido de 5 s y observación asíncrona (llegadas de Poisson). El estimador recupera 5 ± 1 s.
- **Lead-lag con M1**: retardo conocido de 5 min; se recupera 5 ± 1 min.
- **Precio rancio**: un proceso observado con actualización lenta frente a su propia versión sin retraso se marca como "posiblemente debido a precio rancio".
- No hay retardos entre días en el VAR intradía.
- La ventana de control aplica la regla de retroceso de §6.1 en los tres casos (contigua, no contigua, contaminada).

**Look-ahead y datos**

- Sin look-ahead en los flags, el perfil de estacionalidad, la residualización del TVP-VAR, la calibración de λ y las ventanas rolling.
- Lector `.bi5` de velas y de ticks sobre ficheros de ejemplo con valores conocidos.
- Detección de rolls sobre una serie sintética con saltos en fechas conocidas.
- Opcional: una ventana del rolling DY coincide con el paquete `ConnectednessApproach` de R con diferencias menores de 1e-4.

---

## 10. Criterios de aceptación

- `vs run-all` se ejecuta de principio a fin sin intervención manual usando el `config/events.csv` versionado. Solo avisa (sin detenerse) si el calendario no cubre parte de la muestra.
- La ejecución es reproducible con la misma semilla, el mismo `manifest.json` y las mismas versiones fijadas. Todos los tests pasan.
- `summary.md` responde a los dos bloques del §8.10 en la última fecha disponible.
- `reports/run_log.jsonl` documenta: exclusiones de instrumentos, días filtrados por motivo, festivos aplicados, rolls detectados y no detectados, estimaciones no fiables por cuantiles, tramos sin cobertura del calendario, calidad de la ventana de control y avisos de precio rancio.

---

## 11. Advertencias de interpretación

- La GFEVD y la GIRF miden contribución predictiva, no causalidad estructural.
- **En retornos, la conectividad total refleja sobre todo la correlación contemporánea**, y la normalización por filas favorece como emisor al activo más correlacionado con el resto. La transmisión en retornos se lee en la conectividad dinámica.
- **El control `varx` solo es informativo en volatilidad.** En retornos, que la transmisión no cambie con `varx` no demuestra que no sea un efecto de calendario; hay que mirar `exclude`.
- Si un activo transmite en `none` y deja de hacerlo con la variante de referencia de su objetivo, el resultado se atribuye al calendario macro, no a ese activo.
- Los datos son CFDs de Dukascopy: el volumen no es de mercado, la liquidez fuera de sesión es menor y la cotización puede ir con retraso respecto al futuro subyacente. Un lead-lag a favor del activo más líquido puede deberse a eso.
- Con resolución de 1 minuto, el retardo óptimo entre activos líquidos será casi siempre 0; la asimetría es más informativa que el retardo.
- Las estimaciones cuantílicas en cola marcadas como no fiables son indicativas.
- El módulo de episodio tiene pocas semanas de datos y pocos bloques para el bootstrap. Sus resultados son indicativos y deben leerse junto al marco de fondo, no en lugar de él.
