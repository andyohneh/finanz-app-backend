# data_collector.py
import requests
import os
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus der .env Datei (wichtig für lokale Tests)
load_dotenv()

TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
# Die Symbole im Format, das die Twelve Data API erwartet
SYMBOLS = ['BTC/USD', 'XAU/USD'] 

def fetch_and_store_latest_data():
    """
    Holt für jedes Symbol den neuesten Datenpunkt und speichert ihn in der Datenbank,
    aber nur, wenn ein identischer Datensatz (Symbol + Timestamp) noch nicht vorhanden ist.
    """
    print(f"Starte Datensammlung um {datetime.now()}...")
    
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            try:
                # 1. Daten von der API holen (nur den allerneuesten Punkt)
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=1&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=10)
                response.raise_for_status() # Löst einen Fehler aus, wenn die API nicht mit 200 OK antwortet
                data = response.json()

                if data.get('status') == 'ok' and 'values' in data:
                    latest_point = data['values'][0]
                    
                    record_timestamp = datetime.strptime(latest_point['datetime'], '%Y-%m-%d %H:%M:%S')

                    # Erstellen des Datensatzes für die Datenbank
                    record = {
                        "symbol": symbol.replace('/', ''), # 'BTC/USD' -> 'BTCUSD'
                        "timestamp": record_timestamp,
                        "open": float(latest_point['open']),
                        "high": float(latest_point['high']),
                        "low": float(latest_point['low']),
                        "close": float(latest_point['close']),
                        "volume": float(latest_point.get('volume', 0.0))
                    }
                    
                    # 2. Daten in die Datenbank einfügen
                    # on_conflict_do_nothing sorgt dafür, dass kein Fehler auftritt,
                    # wenn wir versuchen, einen bereits existierenden Datenpunkt einzufügen.
                    # Dies benötigt einen Unique Constraint in der DB-Tabelle.
                    # Für den Moment fügen wir eine einfache Prüfung ein.
                    
                    # Prüfen, ob der Datensatz schon existiert
                    from sqlalchemy import select
                    check_stmt = select(historical_data).where(
                        historical_data.c.symbol == record['symbol'],
                        historical_data.c.timestamp == record['timestamp']
                    )
                    result = conn.execute(check_stmt).first()

                    if result is None:
                        # Datensatz existiert nicht, also einfügen
                        insert_stmt = historical_data.insert().values(record)
                        conn.execute(insert_stmt)
                        conn.commit()
                        print(f"-> Neuer Datenpunkt für {record['symbol']} um {record['timestamp']} gespeichert.")
                    else:
                        # Datensatz existiert bereits
                        print(f"-> Datenpunkt für {record['symbol']} um {record['timestamp']} bereits vorhanden, übersprungen.")

                else:
                    print(f"Fehlerhafte Antwort von API für {symbol}: {data.get('message')}")

            except Exception as e:
                print(f"Ein Fehler ist bei der Verarbeitung von {symbol} aufgetreten: {e}")

    print("Datensammlung beendet.")


if __name__ == "__main__":
    fetch_and_store_latest_data()