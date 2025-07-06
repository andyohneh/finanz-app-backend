# backend/initial_data_loader_daily.py (Die finale, korrekte API-Version)
import yfinance as yf
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
import pandas as pd

# Eigene Module importieren
from database import engine, historical_data_daily

# --- EINSTELLUNGEN ---
# Hier kannst du alle Symbole eintragen, die du für deine App benötigst.
# yfinance verwendet Bindestriche.
SYMBOLS = [
    "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "AVAX-USD",
    "MATIC-USD", "DOT-USD", "LINK-USD", "UNI-USD", "XRP-USD", "XAU-USD"
]

def load_all_historical_data():
    """
    Lädt eine große Menge an historischen TÄGLICHEN Daten für alle Symbole
    von yfinance und speichert sie in der Datenbank.
    Bestehende Einträge werden dank 'ON CONFLICT DO NOTHING' ignoriert.
    """
    print("Starte den Download aller historischen Daten von der yfinance API...")
    
    with engine.connect() as conn:
        for ticker in SYMBOLS:
            # Das Datenbankformat verwendet einen Schrägstrich
            db_symbol = ticker.replace('-', '/')
            print(f"\n--- Verarbeite Symbol: {ticker} ---")
            
            try:
                # Lade die maximal verfügbare Menge an täglichen Daten von der API
                data = yf.download(ticker, period="max", interval="1d")
                
                if data.empty:
                    print(f"Keine Daten für {ticker} gefunden. Überspringe.")
                    continue

                # Daten für die Datenbank vorbereiten
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
                    print(f"Keine validen Datensätze für {ticker} zum Einfügen.")
                    continue
                    
                print(f"Füge {len(records)} Datensätze für {db_symbol} in die Datenbank ein...")
                
                # Benutze eine Bulk-Insert-Methode für bessere Performance
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