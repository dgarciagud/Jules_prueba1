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
    "CAC": ["CAC", "FRA40", "FR40"],
    "IBEX": ["IBEX", "ESP35", "SPA35", "ES35"],
    "BRENT": ["BRENT", "UKOIL", "XBR", "BRN"],
    "BUND": ["BUND", "FGBL", "DE10Y", "GER10"],
    "TBOND": ["TBOND", "T-BOND", "USTBOND", "ZB", "US30Y", "UST"],
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
        for ref in ("EURUSD", "EURUSD.", "EURUSDm"):
            tick = mt5.symbol_info_tick(ref)
            if tick and tick.time:
                now = time.time()
                offset = round((tick.time - now) / 3600)
                age = now - (tick.time - offset * 3600)
                fresh = "reciente" if 0 <= age < 60 else f"antiguo ({age / 60:.0f} min; mercado cerrado?)"
                print(f"{ref}: servidor UTC{offset:+d} · último tick {fresh}")
                break
        else:
            print("No hay tick de EURUSD: prueba entre semana con el mercado abierto.")

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
