# initial_data_loader_4h.py
import requests
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data_4h # Wichtig: die neue Tabelle importieren
from dotenv import load_dotenv

load_dotenv()
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
SYMBOLS = ['BTC/USD', 'XAU/USD']

def load_initial_4h_data():
    print("Starte initialen Ladevorgang für 4-STUNDEN-DATEN...")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"Lade große Datenmenge für {symbol} von der API (Intervall: 4h)...")
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=4h&outputsize=5000&apikey={TWELVEDATA_API_KEY}"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'ok' and 'values' in data:
                    records_to_insert = []
                    for v in data['values']:
                        dt_object = datetime.strptime(v['datetime'], '%Y-%m-%d %H:%M:%S')
                        records_to_insert.append({
                            'symbol': symbol, 'timestamp': dt_object,
                            'open': float(v['open']), 'high': float(v['high']),
                            'low': float(v['low']), 'close': float(v['close']),
                            'volume': float(v.get('volume', 0))
                        })
                    
                    if records_to_insert:
                        stmt = insert(historical_data_4h).values(records_to_insert)
                        stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                        conn.execute(stmt)
                        conn.commit()
                        print(f"--> {len(records_to_insert)} 4-Stunden-Datensätze für {symbol} erfolgreich gespeichert.")
                else:
                    print(f"Fehlerhafte API-Antwort für {symbol}: {data.get('message', 'Keine Details')}")
            except Exception as e:
                print(f"Ein Fehler ist bei der Verarbeitung von {symbol} aufgetreten: {e}")

if __name__ == '__main__':
    load_initial_4h_data()