# backend/initial_data_loader_daily.py (Die finale, korrigierte Logik)
import yfinance as yf
from sqlalchemy import text
import pandas as pd

from database import engine

SYMBOLS_TO_FETCH = {
    "BTC-USD": "BTC/USD",
    "GC=F": "XAU/USD"
}

def load_all_historical_data():
    """
    Lädt historische Daten und schreibt sie mit der korrekten Logik
    in die Datenbank, um alle Fehler zu umgehen.
    """
    print("Starte den Download der historischen Daten (finale Logik)...")

    with engine.connect() as conn:
        for ticker, db_symbol in SYMBOLS_TO_FETCH.items():
            print(f"\n--- Verarbeite Symbol: {ticker} ---")

            try:
                # Schritt 1: Daten laden
                data = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)

                if data.empty:
                    print(f"Keine Daten für {ticker} gefunden.")
                    continue

                # Schritt 2: Den Datums-Index in eine Spalte umwandeln
                # yfinance erstellt eine Spalte 'Date'
                data.reset_index(inplace=True)

                # Schritt 3: Spaltennamen bereinigen und umbenennen
                data.rename(columns={
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
                
                # Schritt 4: Das interne Symbol zuweisen
                data['symbol'] = db_symbol
                
                # Schritt 5: Sicherstellen, dass alle nötigen Spalten da sind
                required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                data = data[required_columns]
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