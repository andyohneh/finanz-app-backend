# backend/initial_data_loader_daily.py (Finale Experten-Version)
import yfinance as yf
from sqlalchemy import text
import pandas as pd

from database import engine

# Definition der Ticker, die yfinance kennt
SYMBOLS_TO_FETCH = {
    "BTC-USD": "BTC/USD",
    "GC=F": "XAU/USD"
}

def load_all_historical_data():
    """
    Lädt historische Daten und schreibt sie mit einem direkten SQL-Befehl
    in die Datenbank, um Treiberprobleme zu umgehen.
    """
    print("Starte den Download der historischen Daten (Experten-Modus)...")

    with engine.connect() as conn:
        for ticker, db_symbol in SYMBOLS_TO_FETCH.items():
            print(f"\n--- Verarbeite Symbol: {ticker} ---")

            try:
                data = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)

                if data.empty:
                    print(f"Keine Daten für {ticker} gefunden.")
                    continue

                data.reset_index(inplace=True)
                data.rename(columns={
                    'Date': 'timestamp', 'Open': 'open', 'High': 'high',
                    'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                }, inplace=True)

                data['symbol'] = db_symbol
                data = data[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                data.dropna(inplace=True)

                records = data.to_dict(orient='records')
                if not records:
                    continue

                print(f"Füge {len(records)} Datensätze für {db_symbol} mit direktem SQL-Befehl ein...")

                # Transaktion beginnen
                trans = conn.begin()
                try:
                    for record in records:
                        # Manueller SQL-Befehl für maximale Kompatibilität
                        stmt = text("""
                            INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume)
                            VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume)
                            ON CONFLICT (timestamp, symbol) DO NOTHING
                        """)
                        conn.execute(stmt, record)
                    # Transaktion erfolgreich abschließen
                    trans.commit()
                except:
                    # Bei einem Fehler alles zurückrollen
                    trans.rollback()
                    raise

                print(f"Daten für {db_symbol} erfolgreich importiert.")

            except Exception as e:
                print(f"Ein FEHLER ist bei der Verarbeitung von {ticker} aufgetreten: {e}")

    print("\n✅ Alle historischen Daten wurden erfolgreich geladen!")


if __name__ == "__main__":
    load_all_historical_data()