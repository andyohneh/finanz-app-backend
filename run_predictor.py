# backend/run_predictor.py (Finale Version mit Sentiment-Daten-Ladung)
import argparse
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, timezone

from database import engine, predictions
import predictor_daily
import predictor_swing
import predictor_genius

SYMBOLS_TO_PROCESS = ['BTC/USD', 'XAU/USD']
PREDICTOR_MAPPING = {
    'daily': predictor_daily,
    'swing': predictor_swing,
    'genius': predictor_genius
}

def load_live_data_with_sentiment(symbol: str, conn) -> pd.DataFrame:
    """
    Lädt die neuesten Kursdaten und den letzten Sentiment-Score für die Live-Vorhersage.
    """
    # KORREKTUR: Wir laden Preisdaten UND Sentiment-Daten mit einem JOIN
    query = text("""
        SELECT
            h.timestamp, h.symbol, h.open, h.high, h.low, h.close, h.volume,
            COALESCE(s.sentiment_score, 0.0) as sentiment_score
        FROM (
            SELECT * FROM historical_data_daily
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT 400
        ) h
        LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = s.date
        ORDER BY h.timestamp ASC
    """)
    df = pd.read_sql_query(query, conn, params={'symbol': symbol})
    return df

def generate_and_store_predictions(strategy):
    print(f"=== Starte Generator für '{strategy.upper()}' Live-Signale ===")
    
    predictor_module = PREDICTOR_MAPPING.get(strategy)
    if not predictor_module:
        print(f"FEHLER: Unbekannte Strategie '{strategy}'.")
        return

    with engine.connect() as conn:
        for symbol in SYMBOLS_TO_PROCESS:
            print(f"\n--- Verarbeite {symbol} für Strategie '{strategy}' ---")
            
            model_path = f"models/model_{strategy}_{symbol.replace('/', '')}.pkl"
            print(f"Verwende Modell: {model_path}")

            # Lade die kombinierten Daten
            df_live = load_live_data_with_sentiment(symbol, conn)
            
            if df_live.empty or len(df_live) < 250:
                print(f"Nicht genügend kombinierte Daten für {symbol} vorhanden.")
                continue

            # Übergebe die vollständigen Daten an den Predictor
            prediction_result = predictor_module.get_prediction(df_live, model_path)
            
            if 'error' in prediction_result:
                print(f"Fehler vom Predictor: {prediction_result['error']}")
                continue
            
            update_data = {
                'symbol': symbol, 'strategy': strategy,
                'signal': prediction_result['signal'],
                'entry_price': prediction_result['entry_price'],
                'take_profit': prediction_result['take_profit'],
                'stop_loss': prediction_result['stop_loss'],
                'last_updated': datetime.now(timezone.utc)
            }

            stmt = insert(predictions).values(update_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'strategy'], set_=update_data
            )
            
            conn.execute(stmt)
            conn.commit()
            print(f"✅ Signal für {symbol} ({strategy}) erfolgreich gespeichert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KI-Signal-Generator.")
    parser.add_argument("strategy", type=str, choices=['daily', 'swing', 'genius'], help="Die auszuführende Strategie.")
    args = parser.parse_args()
    generate_and_store_predictions(args.strategy)