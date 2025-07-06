# data_updater_daily.py (Finale, korrigierte Version)
import os
import requests
from datetime import datetime
from sqlalchemy import text, select, func
# KORREKTUR: Wir importieren 'insert' direkt aus dem PostgreSQL-Dialekt
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data_daily
from dotenv import load_dotenv

load_dotenv()
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
SYMBOLS = ['BTC/USD', 'XAU/USD']

def update_daily_data():
    print("Starte tägliches Update der Preisdaten...")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            try:
                last_date_query = select(func.max(historical_data_daily.c.timestamp)).where(historical_data_daily.c.symbol == symbol)
                last_date = conn.execute(last_date_query).scalar_one_or_none()
                
                print(f"Letzter bekannter Datenpunkt für {symbol}: {last_date}")
                
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={TWELVEDATA_API_KEY}&outputsize=5000"
                # Wenn wir schon Daten haben, fragen wir nur die neuesten an.
                # Ansonsten holen wir uns die volle Historie.
                if last_date:
                    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={TWELVEDATA_API_KEY}&outputsize=30&start_date={last_date.strftime('%Y-%m-%d')}"

                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'ok' and 'values' in data:
                    records_to_insert = []
                    for v in data['values']:
                        dt_object = datetime.strptime(v['datetime'], '%Y-%m-%d')
                        if last_date is None or dt_object.date() > last_date.date():
                            records_to_insert.append({
                                'symbol': symbol, 'timestamp': dt_object,
                                'open': float(v['open']), 'high': float(v['high']),
                                'low': float(v['low']), 'close': float(v['close']),
                                'volume': float(v.get('volume', 0))
                            })
                    
                    if records_to_insert:
                        # Jetzt funktioniert der Befehl, da wir den richtigen 'insert' verwenden
                        stmt = insert(historical_data_daily).values(records_to_insert)
                        stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                        conn.execute(stmt)
                        conn.commit()
                        print(f"--> {len(records_to_insert)} neue Tages-Datensätze für {symbol} hinzugefügt.")
                    else:
                        print(f"Keine neuen Daten für {symbol} gefunden.")
                else:
                    print(f"Fehlerhafte API-Antwort für {symbol}: {data.get('message', 'Keine Details')}")

            except Exception as e:
                print(f"Ein Fehler ist beim Update von {symbol} aufgetreten: {e}")

if __name__ == '__main__':
    update_daily_data()