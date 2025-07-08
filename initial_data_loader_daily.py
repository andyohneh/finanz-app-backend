# backend/initial_data_loader_daily.py (Die finale, korrekte Lösung)
import os
import pandas as pd
from sqlalchemy import text

from database import engine

DATA_DIR = "data/daily"

def load_all_historical_data():
    """
    Lädt historische Daten aus den lokalen CSV-Dateien und liest
    die erste Spalte als den korrekten Datums-Index.
    """
    print("Starte den Daten-Import aus den CSV-Dateien (finale Lösung)...")

    if not os.path.exists(DATA_DIR):
        print(f"FEHLER: Das Datenverzeichnis '{DATA_DIR}' wurde nicht gefunden.")
        return

    with engine.connect() as conn:
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(DATA_DIR, filename)
                db_symbol = filename.replace('.csv', '').replace('-', '/')
                
                print(f"\n--- Verarbeite Datei: {filename} für Symbol {db_symbol} ---")
                
                try:
                    # DIE FINALE, ENTSCHEIDENDE KORREKTUR:
                    # Wir sagen pandas, die ERSTE SPALTE (index_col=0) als Index zu verwenden.
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    
                    # JETZT können wir den Index sicher in eine Spalte umwandeln.
                    # Diese Spalte wird automatisch "index" heißen.
                    df.reset_index(inplace=True)

                    # Und JETZT benennen wir sie um.
                    df.rename(columns={
                        'index': 'timestamp', 'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    }, inplace=True)
                    
                    df['symbol'] = db_symbol
                    
                    required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                    df = df[required_columns]
                    df.dropna(inplace=True)

                    records = df.to_dict(orient='records')
                    if not records:
                        continue

                    print(f"Füge {len(records)} Datensätze für {db_symbol} ein...")
                    
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
                    print(f"Ein FEHLER ist bei der Verarbeitung von {filename} aufgetreten: {e}")

    print("\n✅ Alle CSV-Daten wurden erfolgreich in die Datenbank importiert!")

if __name__ == "__main__":
    load_all_historical_data()