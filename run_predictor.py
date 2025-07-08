# backend/run_predictor.py (Die finale, korrekte Version)
import argparse
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, timezone

# Eigene Module importieren
from database import engine, historical_data_daily, predictions
import predictor_daily
import predictor_swing
import predictor_genius

# Die Symbole, die wir verarbeiten wollen
SYMBOLS_TO_PROCESS = ['BTC/USD', 'XAU/USD']

# Eine einfache Zuordnung, welche Strategie welches Skript verwendet
PREDICTOR_MAPPING = {
    'daily': predictor_daily,
    'swing': predictor_swing,
    'genius': predictor_genius
}

def generate_and_store_predictions(strategy):
    """
    Startet den Generator für eine bestimmte Strategie.
    """
    print(f"=== Starte Generator für '{strategy.upper()}' Live-Signale ===")
    
    predictor_module = PREDICTOR_MAPPING.get(strategy)
    if not predictor_module:
        print(f"FEHLER: Unbekannte Strategie '{strategy}'.")
        return

    with engine.connect() as conn:
        for symbol in SYMBOLS_TO_PROCESS:
            print(f"\n--- Verarbeite {symbol} für Strategie '{strategy}' ---")
            
            symbol_filename = symbol.replace('/', '')
            model_path = f"models/model_{strategy}_{symbol_filename}.pkl"
            print(f"Verwende Modell: {model_path}")

            # Wir laden genügend Daten für alle möglichen Indikatoren
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            
            if df.empty or len(df) < 250:
                print(f"Nicht genügend Daten für {symbol} in der Datenbank.")
                continue

            # Die Daten müssen für die Indikatoren-Berechnung zeitlich aufsteigend sein
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            
            # Das jeweilige Predictor-Modul weiß selbst, wie es die Vorhersage machen muss
            prediction_result = predictor_module.get_prediction(df, model_path)
            
            if 'error' in prediction_result:
                print(f"Fehler: {prediction_result['error']}")
                continue
            
            # Die fertigen Daten für die Datenbank vorbereiten
            update_data = {
                'symbol': symbol,
                'strategy': strategy,
                'signal': prediction_result['signal'],
                'entry_price': prediction_result['entry_price'],
                'take_profit': prediction_result['take_profit'],
                'stop_loss': prediction_result['stop_loss'],
                'last_updated': datetime.now(timezone.utc)
            }

            # Datensatz in die 'predictions'-Tabelle einfügen oder aktualisieren
            stmt = insert(predictions).values(update_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'strategy'],
                set_=update_data
            )
            
            conn.execute(stmt)
            conn.commit()
            print(f"Signal für {symbol} ({strategy}) erfolgreich gespeichert.")

if __name__ == "__main__":
    # Dieser Teil sorgt dafür, dass wir das Skript mit 'daily', 'swing' oder 'genius' aufrufen können
    parser = argparse.ArgumentParser(description="KI-Signal-Generator.")
    parser.add_argument("strategy", type=str, choices=['daily', 'swing', 'genius'], help="Die auszuführende Strategie.")
    args = parser.parse_args()
    generate_and_store_predictions(args.strategy)