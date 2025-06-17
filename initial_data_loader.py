# initial_data_loader.py (Version mit Batch-Verarbeitung)
import requests
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data
from dotenv import load_dotenv

load_dotenv()
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
SYMBOLS = ['BTC/USD', 'XAU/USD']

def load_initial_data():
    """
    Lädt eine große Menge an historischen Daten von der API und speichert sie in Batches,
    um die Datenbank nicht zu überlasten. Bestehende Daten werden ignoriert.
    """
    print("Starte initialen Daten-Ladevorgang...")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"Lade große Datenmenge für {symbol} von der API...")
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=5000&apikey={TWELVEDATA_API_KEY}"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'ok' and 'values' in data:
                    records_to_insert = []
                    for v in data['values']:
                        records_to_insert.append({
                            'symbol': symbol,
                            'timestamp': datetime.fromisoformat(v['datetime']),
                            'open': float(v['open']),
                            'high': float(v['high']),
                            'low': float(v['low']),
                            'close': float(v['close']),
                            'volume': float(v.get('volume')) if v.get('volume') else 0
                        })
                    
                    if records_to_insert:
                        # --- NEUE LOGIK: BATCH-VERARBEITUNG ---
                        batch_size = 500  # Wir senden Päckchen mit 500 Datensätzen
                        print(f"Füge {len(records_to_insert)} Datensätze in Päckchen von {batch_size} ein...")
                        
                        for i in range(0, len(records_to_insert), batch_size):
                            batch = records_to_insert[i:i + batch_size]
                            
                            stmt = insert(historical_data).values(batch)
                            stmt = stmt.on_conflict_do_nothing(
                                index_elements=['symbol', 'timestamp']
                            )
                            conn.execute(stmt)
                            conn.commit() # Wichtig: Nach jedem Päckchen speichern
                            print(f"--> Päckchen {i//batch_size + 1} für {symbol} erfolgreich gespeichert.")
                        
                        print(f"Alle Daten für {symbol} erfolgreich gespeichert.")
                        # --- ENDE NEUE LOGIK ---
                    else:
                        print(f"Keine Daten zum Speichern für {symbol} gefunden.")
                else:
                    print(f"Fehlerhafte API-Antwort für {symbol}: {data.get('message', 'Keine Details')}")
            except Exception as e:
                print(f"Ein Fehler ist bei der Verarbeitung von {symbol} aufgetreten: {e}")

if __name__ == '__main__':
    load_initial_data()