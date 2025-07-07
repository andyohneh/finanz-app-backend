# backend/initial_data_loader_daily.py (Finale Version für BTC & XAU)
import yfinance as yf
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
import pandas as pd

# Eigene Module importieren
from database import engine, historical_data_daily

# --- EINSTELLUNGEN ---
# KORREKTUR: Wir laden nur noch die gewünschten Symbole
SYMBOLS = ["BTC-USD", "XAU-USD"]

def load_all_historical_data():
    """
    Lädt historische TAGES-Daten für die definierten Symbole von yfinance
    und speichert sie in der Datenbank.
    """
    print("Starte den Download der historischen Daten von der yfinance API...")
    
    with engine.connect() as conn:
        for ticker in SYMBOLS:
            db_symbol = ticker.replace('-', '/')
            print(f"\n--- Verarbeite Symbol: {ticker} ---")
            
            try:
                data = yf.download(ticker, period="max", interval="1d")
                
                if data.empty:
                    print(f"Keine Daten für {ticker} gefunden. Überspringe.")
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
                    
                print(f"Füge {len(records)} Datensätze für {db_symbol} in die Datenbank ein...")
                
                stmt = insert(historical_data_daily).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=['timestamp', 'symbol'])
                conn.execute(stmt)
                conn.commit()
                
                print(f"Daten für {db_symbol} erfolgreich importiert.")

            except Exception as e:
                print(f"Ein Fehler ist bei der Verarbeitung von {ticker} aufgetreten: {e}")

    print("\n✅ Alle historischen Daten wurden erfolgreich geladen und gespeichert!")

if __name__ == "__main__":
    load_all_historical_data()