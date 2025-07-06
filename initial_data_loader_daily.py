# initial_data_loader_daily.py (Das einzige Skript für Daten-Imports)
import os
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data_daily

# --- EINSTELLUNGEN ---
DATA_DIR = "data/daily" # Der Ordner mit deinen CSV-Dateien

def load_initial_data():
    """
    Lädt historische Daten aus CSV-Dateien im angegebenen Verzeichnis
    in die Datenbank. Bestehende Einträge (basierend auf Zeitstempel und Symbol)
    werden dank 'ON CONFLICT DO NOTHING' ignoriert.
    Dieses Skript kann sicher mehrfach ausgeführt werden, um neue Daten
    aus den aktualisierten CSVs hinzuzufügen.
    """
    print("Starte Daten-Import aus CSV-Dateien...")
    
    # Überprüfen, ob das Datenverzeichnis existiert
    if not os.path.exists(DATA_DIR):
        print(f"FEHLER: Das Datenverzeichnis '{DATA_DIR}' wurde nicht gefunden.")
        return

    # Alle CSV-Dateien im Verzeichnis durchgehen
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            symbol_name = filename.replace('.csv', '').replace('-', '/') # z.B. BTC-USD.csv -> BTC/USD
            filepath = os.path.join(DATA_DIR, filename)
            
            print(f"Verarbeite Datei: {filename} für Symbol {symbol_name}")
            
            try:
                # CSV-Datei mit Pandas laden
                df = pd.read_csv(filepath)
                
                # Spaltennamen an die Datenbank anpassen
                df.rename(columns={
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)

                # Notwendige Spalten für die Datenbank
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"FEHLER in {filename}: Notwendige Spalten fehlen. Überspringe.")
                    continue

                # Symbol-Spalte hinzufügen
                df['symbol'] = symbol_name
                
                # Datentyp der Timestamp-Spalte korrigieren
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Daten in die Datenbank einfügen
                with engine.connect() as conn:
                    for index, row in df.iterrows():
                        # Ignoriere Zeilen mit NaN-Werten
                        if row.isnull().any():
                            continue

                        insert_stmt = insert(historical_data_daily).values(
                            timestamp=row['timestamp'],
                            symbol=row['symbol'],
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=row['volume']
                        ).on_conflict_do_nothing(
                            index_elements=['timestamp', 'symbol']
                        )
                        conn.execute(insert_stmt)
                    conn.commit()

                print(f"Daten für {symbol_name} erfolgreich importiert/aktualisiert.")

            except Exception as e:
                print(f"Ein Fehler ist bei der Verarbeitung von {filename} aufgetreten: {e}")

    print("Daten-Import abgeschlossen.")

if __name__ == "__main__":
    load_initial_data()