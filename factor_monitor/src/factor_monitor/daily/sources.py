"""Cierres diarios de Yahoo Finance (API `chart`) con Stooq de respaldo.

Sin yfinance: dos endpoints públicos y una sesión con cabecera de navegador.
- `history`: todo el histórico diario de un símbolo en una sola petición.
- `quotes` / `profile` / `fund_sectors`: capitalización, sector y pesos sectoriales
  de un fondo; necesitan la cookie y el "crumb" de Yahoo, que se obtienen una vez.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.cookiejar import CookieJar
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
YAHOO = "https://query2.finance.yahoo.com"
STOOQ = "https://stooq.com/q/d/l/"

# Claves de `sectorWeightings` de Yahoo -> claves del universo
FUND_SECTOR_KEYS = {
    "realestate": "real_estate",
    "consumer_cyclical": "consumer_cyclical",
    "basic_materials": "basic_materials",
    "consumer_defensive": "consumer_defensive",
    "technology": "technology",
    "communication_services": "communication_services",
    "financial_services": "financial_services",
    "utilities": "utilities",
    "industrials": "industrials",
    "energy": "energy",
    "healthcare": "healthcare",
}
# Sector de `assetProfile` -> clave del universo
PROFILE_SECTOR_KEYS = {
    "Technology": "technology",
    "Financial Services": "financial_services",
    "Consumer Cyclical": "consumer_cyclical",
    "Communication Services": "communication_services",
    "Healthcare": "healthcare",
    "Industrials": "industrials",
    "Consumer Defensive": "consumer_defensive",
    "Energy": "energy",
    "Utilities": "utilities",
    "Real Estate": "real_estate",
    "Basic Materials": "basic_materials",
}


class SourceError(RuntimeError):
    pass


Getter = Callable[[str], bytes]


@dataclass
class Http:
    """GET con cabecera de navegador, cookies, pausa entre peticiones y reintentos ante 429/5xx."""

    pause: float = 0.4
    retries: int = 4
    backoff: float = 5.0
    sleep: Callable[[float], None] = time.sleep
    jar: CookieJar = field(default_factory=CookieJar)
    _last: float = 0.0

    def __post_init__(self):
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [("User-Agent", UA), ("Accept", "*/*")]

    def get(self, url: str) -> bytes:
        for attempt in range(self.retries + 1):
            wait = self.pause - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            try:
                with self.opener.open(url, timeout=30) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise SourceError(f"404 en {url}") from e
                if e.code not in (401, 429, 500, 502, 503, 504) or attempt == self.retries:
                    raise SourceError(f"HTTP {e.code} en {url}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise SourceError(f"{type(e).__name__} en {url}: {e}") from e
            self.sleep(self.backoff * 2**attempt)
        raise SourceError(url)  # pragma: no cover


# ----------------------------------------------------------------------------- Yahoo


def parse_chart(raw: bytes, symbol: str) -> pd.DataFrame:
    """JSON de `v8/finance/chart` -> DataFrame (date, close, adjclose) con la fecha local del mercado."""
    data = json.loads(raw)
    chart = data.get("chart") or {}
    if chart.get("error"):
        raise SourceError(f"{symbol}: {chart['error'].get('description') or chart['error']}")
    res = (chart.get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        raise SourceError(f"{symbol}: sin datos")
    tz = res["meta"].get("exchangeTimezoneName") or "UTC"
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    quote = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(ts.date).as_unit("ns"),
            "close": pd.to_numeric(pd.Series(quote.get("close")), errors="coerce").to_numpy(),
            "adjclose": pd.to_numeric(pd.Series(adj if adj is not None else quote.get("close")), errors="coerce").to_numpy(),
        }
    )
    df = df.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
    if df.empty:
        raise SourceError(f"{symbol}: sin cierres válidos")
    return df.reset_index(drop=True)


@dataclass
class Yahoo:
    http: Http = field(default_factory=Http)
    crumb: str | None = None

    def history(self, symbol: str, start: str = "1990-01-01") -> pd.DataFrame:
        p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        p2 = int(time.time()) + 86400
        q = urllib.parse.urlencode({"period1": p1, "period2": p2, "interval": "1d", "events": "div,split", "includeAdjustedClose": "true"})
        url = f"{YAHOO}/v8/finance/chart/{urllib.parse.quote(symbol)}?{q}"
        return parse_chart(self.http.get(url), symbol)

    def _crumb(self) -> str:
        if self.crumb is None:
            try:
                self.http.get("https://fc.yahoo.com/")
            except SourceError:
                pass  # responde 404 pero deja la cookie
            crumb = self.http.get(f"{YAHOO}/v1/test/getcrumb").decode().strip()
            if not crumb or "<" in crumb:
                raise SourceError("Yahoo: no se obtuvo el crumb")
            self.crumb = crumb
        return self.crumb

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Capitalización, acciones en circulación y nombre por símbolo."""
        out: dict[str, dict] = {}
        for i in range(0, len(symbols), 40):
            chunk = symbols[i : i + 40]
            q = urllib.parse.urlencode({"symbols": ",".join(chunk), "crumb": self._crumb()})
            data = json.loads(self.http.get(f"{YAHOO}/v7/finance/quote?{q}"))
            for r in data.get("quoteResponse", {}).get("result", []):
                out[r["symbol"]] = {k: r.get(k) for k in ("marketCap", "sharesOutstanding", "regularMarketPrice", "currency", "longName")}
        return out

    def _summary(self, symbol: str, module: str) -> dict:
        q = urllib.parse.urlencode({"modules": module, "crumb": self._crumb()})
        data = json.loads(self.http.get(f"{YAHOO}/v10/finance/quoteSummary/{urllib.parse.quote(symbol)}?{q}"))
        res = (data.get("quoteSummary", {}).get("result") or [None])[0]
        if not res or module not in res:
            raise SourceError(f"{symbol}: sin módulo {module}")
        return res[module]

    def profile(self, symbol: str) -> dict:
        """Sector e industria de una acción."""
        p = self._summary(symbol, "assetProfile")
        return {"sector": PROFILE_SECTOR_KEYS.get(p.get("sector")), "sector_raw": p.get("sector"), "industry": p.get("industry")}

    def fund_sectors(self, symbol: str) -> dict[str, float]:
        """Pesos sectoriales de un fondo (p. ej. SPY), normalizados a 1."""
        th = self._summary(symbol, "topHoldings")
        w: dict[str, float] = {}
        for item in th.get("sectorWeightings", []):
            for k, v in item.items():
                key = FUND_SECTOR_KEYS.get(k)
                val = v.get("raw") if isinstance(v, dict) else v
                if key and val is not None:
                    w[key] = w.get(key, 0.0) + float(val)
        total = sum(w.values())
        if total <= 0:
            raise SourceError(f"{symbol}: sin pesos sectoriales")
        return {k: v / total for k, v in w.items()}


# ----------------------------------------------------------------------------- Stooq


def parse_stooq(raw: bytes, symbol: str) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text or text.lower().startswith("no data") or "Date" not in text.splitlines()[0]:
        raise SourceError(f"stooq {symbol}: sin datos")
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns=str.lower)
    if "close" not in df or "date" not in df:
        raise SourceError(f"stooq {symbol}: formato inesperado")
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]).dt.as_unit("ns"), "close": pd.to_numeric(df["close"], errors="coerce")})
    out["adjclose"] = out["close"]
    out = out.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
    if out.empty:
        raise SourceError(f"stooq {symbol}: sin cierres válidos")
    return out.reset_index(drop=True)


@dataclass
class Stooq:
    http: Http = field(default_factory=lambda: Http(pause=1.0))

    def history(self, symbol: str) -> pd.DataFrame:
        q = urllib.parse.urlencode({"s": symbol, "i": "d"})
        return parse_stooq(self.http.get(f"{STOOQ}?{q}"), symbol)
