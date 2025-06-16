import requests
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert

# Wir importieren unsere SQLAlchemy Engine und die Tabellen-Definition
from database import engine, historical_data

# --- API-SCHLÜSSEL ---
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', '2a152b2fb77743cda7c0066278e4ef37')

def backfill_historical_data(symbol, output_size=2000):
    """
    Holt eine große Menge historischer Daten und speichert sie in der Datenbank.
    """
    print(f"Starte Backfill für {symbol} mit {output_size} Datenpunkten...")
    try:
        # Wir fragen eine große Menge an Daten an
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize={output_size}&apikey={TWELVEDATA_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'ok' and 'values' in data:
            values = data['values']
            print(f"{len(values)} Datenpunkte von API für {symbol} erhalten.")
            
            # Wir bereiten alle Daten für einen schnellen Insert vor
            records_to_insert = []
            for entry in values:
                records_to_insert.append({
                    "symbol": symbol.replace('/', ''),
                    "timestamp": datetime.strptime(entry['datetime'], '%Y-%m-%d %H:%M:%S'),
                    "open": float(entry['open']),
                    "high": float(entry['high']),
                    "low": float(entry['low']),
                    "close": float(entry['close']),
                    "volume": float(entry.get('volume', 0))
                })
            
            # Speichere alle auf einmal in der DB
            if records_to_insert:
                with engine.connect() as conn:
                    # Diese spezielle Anweisung fügt Daten ein und ignoriert Konflikte,
                    # falls ein Datenpunkt (selber Timestamp/Symbol) schon existiert.
                    stmt = insert(historical_data).values(records_to_insert)
                    stmt = stmt.on_conflict_do_nothing()
                    conn.execute(stmt)
                    conn.commit()
                print(f"{len(records_to_insert)} Datensätze erfolgreich in die DB für {symbol} geschrieben.")
            return True
        else:
            print(f"Konnte keine gültigen Daten für {symbol} von der API bekommen: {data.get('message')}")
            return False
            
    except Exception as e:
        print(f"Ein schwerer Fehler beim Backfill für {symbol} ist aufgetreten: {e}")
        return False

# --- Haupt-Logik ---
if __name__ == "__main__":
    print(f"--- Starte historisches Daten-Backfill um {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # Zuerst die Tabelle in der leeren DB erstellen!
    from database import create_tables
    print("Erstelle Tabellenstruktur in der neuen Datenbank...")
    create_tables()

    # Jetzt das Backfill für jedes Symbol durchführen
    symbols_to_backfill = ['BTC/USD', 'XAU/USD']
    for symbol in symbols_to_backfill:
        backfill_historical_data(symbol)
    
    print(f"--- Daten-Backfill beendet ---")