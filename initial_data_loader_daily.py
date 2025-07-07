# backend/initial_data_loader_daily.py (Finale, robuste Version)
import yfinance as yf
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
import pandas as pd

from database import engine, historical_data_daily

SYMBOLS_TO_FETCH = {
    "BTC-USD": "BTC/USD",
    "GC=F": "XAU/USD"
}

def load_all_historical_data():
    print("Starte den Download der historischen Daten von der yfinance API...")
    
    with engine.connect() as conn:
        for ticker, db_symbol in SYMBOLS_TO_FETCH.items():
            print(f"\n--- Verarbeite Symbol: {ticker} (wird als {db_symbol} gespeichert) ---")
            
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
                    
                print(f"Füge {len(records)} Datensätze für {db_symbol} Zeile für Zeile ein (das kann dauern)...")
                
                # FINALE KORREKTUR: Wir fügen die Daten Zeile für Zeile ein, um Treiber-Probleme zu umgehen.
                for record in records:
                    stmt = insert(historical_data_daily).values(record)
                    stmt = stmt.on_conflict_do_nothing(index_elements=['timestamp', 'symbol'])
                    conn.execute(stmt)
                
                # Wichtig: Die Transaktion am Ende committen
                conn.commit()
                
                print(f"Daten für {db_symbol} erfolgreich importiert.")

            except Exception as e:
                print(f"Ein Fehler ist bei der Verarbeitung von {ticker} aufgetreten: {e}")

    print("\n✅ Alle historischen Daten wurden erfolgreich geladen!")

if __name__ == "__main__":
    load_all_historical_data()