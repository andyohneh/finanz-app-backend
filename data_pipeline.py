# backend/data_pipeline.py
import requests
import os
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv

# Eigene Module importieren
from database import engine

# --- KONFIGURATION ---
load_dotenv()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

# Symbole, die wir von der API abrufen
SYMBOLS = {
    "BTC/USD": "BTC/USD",
    "XAU/USD": "XAU/USD" 
}

def load_historical_data():
    """
    Holt eine große Menge historischer Daten von der Twelvedata API
    und speichert sie in der Datenbank.
    """
    if not TWELVEDATA_API_KEY:
        print("FEHLER: TWELVEDATA_API_KEY nicht in den Umgebungsvariablen gefunden.")
        return

    print("=== STARTE HISTORISCHEN DATEN-IMPORT (TWELVEDATA) ===")
    
    with engine.connect() as conn:
        for api_symbol, db_symbol in SYMBOLS.items():
            print(f"\n--- Lade Daten für {api_symbol} ---")
            try:
                # Wir holen bis zu 5000 Datenpunkte für eine solide Historie
                url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval=1day&outputsize=5000&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()

                if data.get('status') == 'ok' and 'values' in data:
                    records = []
                    for v in data['values']:
                        records.append({
                            'timestamp': datetime.strptime(v['datetime'], '%Y-%m-%d'),
                            'symbol': db_symbol,
                            'open': float(v['open']),
                            'high': float(v['high']),
                            'low': float(v['low']),
                            'close': float(v['close']),
                            'volume': int(v.get('volume', 0))
                        })
                    
                    if not records:
                        print("Keine Datensätze zum Einfügen gefunden.")
                        continue
                        
                    print(f"Füge {len(records)} Datensätze für {db_symbol} ein...")
                    trans = conn.begin()
                    try:
                        # Wir verwenden einen direkten SQL-Befehl für maximale Stabilität
                        for record in records:
                            stmt = text("""
                                INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume) 
                                VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume) 
                                ON CONFLICT (timestamp, symbol) DO NOTHING
                            """)
                            conn.execute(stmt, record)
                        trans.commit()
                        print(f"✅ Daten für {db_symbol} erfolgreich importiert.")
                    except Exception as e_insert:
                        trans.rollback()
                        print(f"FEHLER beim Einfügen: {e_insert}")
                else:
                    print(f"Fehlerhafte API-Antwort: {data.get('message')}")
            
            except Exception as e:
                print(f"Ein FEHLER ist bei {api_symbol} aufgetreten: {e}")

    print("\n=== DATEN-IMPORT ABGESCHLOSSEN ===")

if __name__ == '__main__':
    load_historical_data()