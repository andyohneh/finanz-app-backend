# data_updater_daily.py (Finale Version mit Zeitzonen-Fix)
import os
import requests
from datetime import datetime, date
import pytz # Wichtig für Zeitzonen
from sqlalchemy import text, select, func
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
                # Finde das letzte Datum in der Datenbank
                last_date_query = select(func.max(historical_data_daily.c.timestamp)).where(historical_data_daily.c.symbol == symbol)
                last_db_date = conn.execute(last_date_query).scalar_one_or_none()
                
                # Konvertiere das letzte DB-Datum (UTC) in ein reines Datum-Objekt
                last_date_obj = last_db_date.date() if last_db_date else None
                print(f"Letzter bekannter Kalendertag für {symbol}: {last_date_obj}")

                # Heutiges Datum in UTC, um sicher mit der DB zu vergleichen
                today_utc = datetime.now(pytz.utc).date()

                # Wenn der letzte bekannte Tag gestern oder heute ist, sind wir aktuell.
                if last_date_obj and last_date_obj >= (today_utc - pd.Timedelta(days=1)):
                    print(f"Daten für {symbol} sind bereits aktuell.")
                    continue
                
                # API-Anfrage für die volle Historie, falls die DB leer ist, sonst für die letzten 30 Tage
                output_size = 5000 if not last_date_obj else 30
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={TWELVEDATA_API_KEY}&outputsize={output_size}"

                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'ok' and 'values' in data:
                    records_to_insert = []
                    for v in data['values']:
                        dt_object = datetime.strptime(v['datetime'], '%Y-%m-%d')
                        # Füge nur wirklich neue Daten hinzu
                        if last_date_obj is None or dt_object.date() > last_date_obj:
                            records_to_insert.append({
                                'symbol': symbol, 'timestamp': dt_object,
                                'open': float(v['open']), 'high': float(v['high']),
                                'low': float(v['low']), 'close': float(v['close']),
                                'volume': float(v.get('volume', 0))
                            })
                    
                    if records_to_insert:
                        # Wir sortieren, um sicherzustellen, dass die ältesten zuerst kommen
                        records_to_insert.reverse()
                        stmt = insert(historical_data_daily).values(records_to_insert)
                        stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                        conn.execute(stmt)
                        conn.commit()
                        print(f"--> {len(records_to_insert)} neue Tages-Datensätze für {symbol} hinzugefügt.")
                    else:
                        print(f"Keine neuen Daten zum Hinzufügen für {symbol} gefunden.")
                else:
                    print(f"Fehlerhafte API-Antwort für {symbol}: {data.get('message', 'Keine Details')}")

            except Exception as e:
                print(f"Ein Fehler ist beim Update von {symbol} aufgetreten: {e}")

if __name__ == '__main__':
    update_daily_data()