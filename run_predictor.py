# backend/run_predictor.py
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, timezone

# Eigene Module importieren
from database import engine, historical_data_daily, predictions
import predictor_daily # Wir nutzen die "daily" Strategie für die Live-Signale

# --- KONFIGURATION ---
SYMBOLS = ['BTC/USD', 'XAU/USD']

def generate_and_store_predictions():
    """
    Generiert für jedes Symbol eine neue Vorhersage und speichert sie
    in der 'predictions'-Tabelle in der Datenbank.
    """
    print("=== Starte Generator für Live-Signale ===")
    
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Verarbeite {symbol} ---")
            
            # 1. Lade die neuesten Daten für das Symbol
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 200")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            
            if df.empty or len(df) < 50: # Wir brauchen genug Daten
                print(f"Nicht genügend historische Daten für {symbol}, überspringe.")
                continue

            # Daten aufsteigend sortieren für die Indikatoren-Berechnung
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            # 2. Hole die Vorhersage vom Predictor-Modul
            prediction_result = predictor_daily.get_prediction(df, symbol)
            
            if 'error' in prediction_result:
                print(f"Fehler bei der Vorhersage für {symbol}: {prediction_result['error']}")
                continue
            
            # 3. Bereite den Datensatz für die Datenbank vor
            update_data = {
                'symbol': symbol,
                'signal': prediction_result['signal'],
                'entry_price': prediction_result['entry_price'],
                'take_profit': prediction_result['take_profit'],
                'stop_loss': prediction_result['stop_loss'],
                'last_updated': datetime.now(timezone.utc)
            }

            # 4. Speichere das Ergebnis in der 'predictions'-Tabelle (Upsert)
            stmt = insert(predictions).values(update_data)
            
            # Definiere, was bei einem Konflikt (Symbol existiert bereits) passieren soll:
            # Aktualisiere einfach alle Felder.
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol'],
                set_=update_data
            )
            
            conn.execute(stmt)
            conn.commit()
            print(f"Signal für {symbol} erfolgreich in der Datenbank gespeichert: {prediction_result['signal']}")

    print("\n=== Signal-Generator abgeschlossen ===")


if __name__ == "__main__":
    generate_and_store_predictions()