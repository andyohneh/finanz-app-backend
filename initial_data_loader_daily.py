# backend/initial_data_loader_daily.py (Die absolut finale Version)
import yfinance as yf
from sqlalchemy import text
import pandas as pd

from database import engine

SYMBOLS_TO_FETCH = {
    "BTC-USD": "BTC/USD",
    "GC=F": "XAU/USD"
}

def load_all_historical_data():
    print("Starte den Download der historischen Daten (finaler, robuster Modus)...")

    with engine.connect() as conn:
        for ticker, db_symbol in SYMBOLS_TO_FETCH.items():
            print(f"\n--- Verarbeite Symbol: {ticker} ---")

            try:
                data = yf.download(ticker, period="max", interval="1d", progress=False)

                if data.empty:
                    print(f"Keine Daten für {ticker} gefunden.")
                    continue

                # FINALE, ENTSCHEIDENDE KORREKTUR:
                # Wir greifen auf die erste Ebene der Spaltennamen zu, um Tupel zu vermeiden.
                data.columns = data.columns.get_level_values(0)
                # Und JETZT konvertieren wir sie zu Kleinbuchstaben.
                data.columns = [col.lower() for col in data.columns]


                data.reset_index(inplace=True)
                data.rename(columns={'date': 'timestamp'}, inplace=True)
                
                data['symbol'] = db_symbol
                data = data[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                data.dropna(inplace=True)

                records = data.to_dict(orient='records')
                if not records:
                    continue

                print(f"Füge {len(records)} Datensätze für {db_symbol} mit direktem SQL-Befehl ein...")

                trans = conn.begin()
                try:
                    for record in records:
                        stmt = text("""
                            INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume)
                            VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume)
                            ON CONFLICT (timestamp, symbol) DO NOTHING
                        """)
                        conn.execute(stmt, record)
                    trans.commit()
                except:
                    trans.rollback()
                    raise

                print(f"Daten für {db_symbol} erfolgreich importiert.")

            except Exception as e:
                print(f"Ein FEHLER ist bei der Verarbeitung von {ticker} aufgetreten: {e}")

    print("\n✅ Alle historischen Daten wurden erfolgreich geladen!")

if __name__ == "__main__":
    load_all_historical_data()