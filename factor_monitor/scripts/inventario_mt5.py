"""Inventario de símbolos de MT5 (Darwinex) para rellenar `mt5_symbol` en universe.yaml.

Uso (en el PC con Windows y el terminal MT5 abierto y con sesión iniciada):
    pip install MetaTrader5
    python inventario_mt5.py

Solo LEE información: lista de símbolos, hora del servidor y nombre del broker.
No lee saldo, posiciones ni número de cuenta, y no envía órdenes.
Genera `mt5_simbolos.csv` y muestra un resumen para pegar en el chat.
"""

import csv
import sys
import time

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("Falta el paquete: pip install MetaTrader5")

# Palabras clave para proponer el símbolo de cada instrumento del universo.
CANDIDATES = {
    "SPX": ["SPX", "US500", "SP500", "USA500", "S&P"],
    "NDX": ["NDX", "US100", "NAS100", "USTEC", "NASDAQ"],
    "SX5E": ["STOXX50", "EU50", "ESTX50", "SX5E", "EUSTX", "EURO STOXX"],
    "DAX": ["DAX", "GER40", "DE40", "GDAXI", "GER30"],
    "CAC": ["CAC 40", "FRA40", "FR40", "FCHI", "FRANCE"],
    "IBEX": ["IBEX", "ESP35", "SPA35", "ES35"],
    "BRENT": ["BRENT", "UKOIL", "XBR", "BRN", "OIL", "CRUDE", "WTI", "XTI"],
    "BUND": ["BUND", "FGBL", "DE10Y", "GER10", "GERMAN BOND", "BOBL", "SCHATZ"],
    "TBOND": ["TBOND", "T-BOND", "USTBOND", "US30Y", "US10Y", "TREASURY", "T-NOTE", "TNOTE", " BOND"],
    "EURUSD": ["EURUSD"],
    "BNP": ["BNP"], "SAN": ["SANTANDER", "SAN.MC", "SAN."], "INGA": ["ING"], "ISP": ["INTESA", "ISP"],
    "UCG": ["UNICREDIT", "UCG"], "BBVA": ["BBVA"], "DBK": ["DEUTSCHE BANK", "DBK"], "GLE": ["SOCIETE", "GLE"],
    "TTE": ["TOTAL", "TTE"], "ENI": ["ENI"], "REP": ["REPSOL", "REP"], "GALP": ["GALP"], "OMV": ["OMV"],
}


def main():
    if not mt5.initialize():
        sys.exit(f"No se pudo conectar con MT5 (¿está abierto y con sesión iniciada?): {mt5.last_error()}")
    try:
        term = mt5.terminal_info()
        acc = mt5.account_info()
        print(f"Terminal: {getattr(term, 'name', '?')} · build {getattr(term, 'build', '?')}")
        print(f"Broker: {getattr(acc, 'company', '?')} · servidor {getattr(acc, 'server', '?')}")

        symbols = mt5.symbols_get() or []
        print(f"Símbolos disponibles: {len(symbols)}")
        with open("mt5_simbolos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "description", "path", "digits", "point", "trade_mode"])
            for s in symbols:
                w.writerow([s.name, s.description, s.path, s.digits, s.point, s.trade_mode])
        print("Lista completa guardada en mt5_simbolos.csv")

        # Hora del servidor frente a UTC (con un símbolo que tenga ticks recientes).
        print("\n--- Hora del servidor ---")
        now = time.time()
        tick = mt5.symbol_info_tick("EURUSD")
        offset = round((tick.time - now) / 3600) if tick and tick.time else None
        if offset is not None and -12 <= offset <= 14 and 0 <= now - (tick.time - offset * 3600) < 60:
            print(f"Servidor UTC{offset:+d} (tick reciente de EURUSD)")
        else:
            # Mercado cerrado: el forex cierra el viernes a las 17:00 de Nueva York.
            rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 1)
            if rates is not None and len(rates):
                from datetime import datetime, timedelta
                from zoneinfo import ZoneInfo
                ny = datetime.fromtimestamp(now, ZoneInfo("America/New_York"))
                friday = (ny - timedelta(days=(ny.weekday() - 4) % 7)).replace(hour=17, minute=0, second=0, microsecond=0)
                if friday > ny:
                    friday -= timedelta(days=7)
                inferred = round((int(rates[-1]["time"]) + 300 - friday.timestamp()) / 3600)
                print(f"Servidor UTC{inferred:+d} (deducido del cierre del forex del viernes; mercado cerrado)")
            else:
                print("No se pudo determinar: prueba entre semana con el mercado abierto.")

        print("\n--- Propuesta de mt5_symbol (revisar) ---")
        for iid, keys in CANDIDATES.items():
            hits = []
            for s in symbols:
                text = f"{s.name} {s.description}".upper()
                if any(k in text for k in keys):
                    hits.append(f"{s.name} ({s.description}; {s.path})")
            shown = " | ".join(hits[:6]) if hits else "— sin coincidencias —"
            print(f"{iid:7s}: {shown}")
        print("\nPega en el chat todo lo que hay desde 'Terminal:' hasta aquí.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
