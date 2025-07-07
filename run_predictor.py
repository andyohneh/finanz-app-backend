# backend/run_predictor.py (Finale Version mit Strategie-Speicherung)
import argparse
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, timezone

from database import engine, historical_data_daily, predictions
import predictor_daily
import predictor_swing
import predictor_genius

SYMBOLS_TO_PROCESS = ['BTC/USD', 'XAU/USD']
PREDICTOR_MAPPING = {
    'daily': predictor_daily, 'swing': predictor_swing, 'genius': predictor_genius
}

def generate_and_store_predictions(strategy):
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

            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 200")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            
            if df.empty or len(df) < 50:
                print(f"Nicht genügend Daten für {symbol}.")
                continue

            df = df.sort_values(by='timestamp').reset_index(drop=True)
            prediction_result = predictor_module.get_prediction(df, model_path)
            
            if 'error' in prediction_result:
                print(f"Fehler: {prediction_result['error']}")
                continue
            
            # NEU: 'strategy'-Feld zum Datensatz hinzufügen
            update_data = {
                'symbol': symbol,
                'strategy': strategy,
                'signal': prediction_result['signal'],
                'entry_price': prediction_result['entry_price'],
                'take_profit': prediction_result['take_profit'],
                'stop_loss': prediction_result['stop_loss'],
                'last_updated': datetime.now(timezone.utc)
            }

            # Upsert basierend auf dem neuen Unique Key (symbol, strategy)
            stmt = insert(predictions).values(update_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'strategy'],
                set_=update_data
            )
            
            conn.execute(stmt)
            conn.commit()
            print(f"Signal für {symbol} ({strategy}) erfolgreich gespeichert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KI-Signal-Generator.")
    parser.add_argument("strategy", type=str, choices=['daily', 'swing', 'genius'], help="Die Strategie.")
    args = parser.parse_args()
    generate_and_store_predictions(args.strategy)