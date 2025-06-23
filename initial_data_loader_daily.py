# initial_data_loader.py (Swing-Trading-Version)
import requests
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data_daily # Wichtig: die neue Tabelle importieren
from dotenv import load_dotenv

load_dotenv()
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
SYMBOLS = ['BTC/USD', 'XAU/USD']

def load_initial_daily_data():
    print("Starte initialen Ladevorgang für TAGES-DATEN (Swing Trading)...")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"Lade große Datenmenge für {symbol} von der API (Intervall: 1day)...")
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={TWELVEDATA_API_KEY}"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'ok' and 'values' in data:
                    records_to_insert = []
                    for v in data['values']:
                        # Konvertiere das Datum (ohne Uhrzeit) in ein DateTime-Objekt
                        dt_object = datetime.strptime(v['datetime'], '%Y-%m-%d')
                        records_to_insert.append({
                            'symbol': symbol, 'timestamp': dt_object,
                            'open': float(v['open']), 'high': float(v['high']),
                            'low': float(v['low']), 'close': float(v['close']),
                            'volume': float(v.get('volume', 0))
                        })
                    
                    if records_to_insert:
                        # Schreibe die Daten in die neue Tabelle
                        stmt = insert(historical_data_daily).values(records_to_insert)
                        stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                        conn.execute(stmt)
                        conn.commit()
                        print(f"--> {len(records_to_insert)} Tages-Datensätze für {symbol} erfolgreich gespeichert.")
                else:
                    print(f"Fehlerhafte API-Antwort für {symbol}: {data.get('message', 'Keine Details')}")
            except Exception as e:
                print(f"Ein Fehler ist bei der Verarbeitung von {symbol} aufgetreten: {e}")

if __name__ == '__main__':
    load_initial_daily_data()