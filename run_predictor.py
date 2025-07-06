# backend/run_predictor.py (Finale, intelligente Version)
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

# --- KONFIGURATION ---
SYMBOLS_TO_PROCESS = ['BTC/USD', 'XAU/USD']

# Mapping von Strategie-Argument zu Predictor-Modul
PREDICTOR_MAPPING = {
    'daily': predictor_daily,
    'swing': predictor_swing,
    'genius': predictor_genius
}

def generate_and_store_predictions(strategy):
    """
    Generiert für eine gegebene Strategie die Signale und speichert sie.
    """
    print(f"=== Starte Generator für '{strategy.upper()}' Live-Signale ===")
    
    predictor_module = PREDICTOR_MAPPING.get(strategy)
    if not predictor_module:
        print(f"FEHLER: Unbekannte Strategie '{strategy}'. Verfügbar: {list(PREDICTOR_MAPPING.keys())}")
        return

    with engine.connect() as conn:
        for symbol in SYMBOLS_TO_PROCESS:
            print(f"\n--- Verarbeite {symbol} für Strategie '{strategy}' ---")
            
            # 1. Konstruiere den korrekten Modell-Pfad
            symbol_filename = symbol.replace('/', '') # BTC/USD -> BTCUSD
            model_path = f"models/model_{strategy}_{symbol_filename}.pkl"
            print(f"Verwende Modell: {model_path}")
            
            # 2. Lade die neuesten Daten für das Symbol
            # HINWEIS: Wir verwenden hier immer die TAGESDATEN als Basis.
            # Wenn du 4H-Daten für Swing hast, müsstest du hier die Tabelle anpassen.
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 200")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            
            if df.empty or len(df) < 50:
                print(f"Nicht genügend historische Daten für {symbol}, überspringe.")
                continue

            df = df.sort_values(by='timestamp').reset_index(drop=True)

            # 3. Hole die Vorhersage vom jeweiligen Predictor-Modul
            prediction_result = predictor_module.get_prediction(df, model_path)
            
            if 'error' in prediction_result:
                print(f"Fehler bei der Vorhersage für {symbol}: {prediction_result['error']}")
                continue
            
            # 4. Speichere das Ergebnis in der 'predictions'-Tabelle
            # HINWEIS: Dies überschreibt Signale von anderen Strategien für dasselbe Symbol!
            # Deine App zeigt immer das Signal an, das als letztes generiert wurde.
            update_data = {
                'symbol': symbol,
                'signal': prediction_result['signal'],
                'entry_price': prediction_result['entry_price'],
                'take_profit': prediction_result['take_profit'],
                'stop_loss': prediction_result['stop_loss'],
                'last_updated': datetime.now(timezone.utc)
            }

            stmt = insert(predictions).values(update_data)
            stmt = stmt.on_conflict_do_update(index_elements=['symbol'], set_=update_data)
            
            conn.execute(stmt)
            conn.commit()
            print(f"Signal für {symbol} ({strategy}) erfolgreich gespeichert: {prediction_result['signal']}")

    print(f"\n=== Signal-Generator für '{strategy.upper()}' abgeschlossen ===")

if __name__ == "__main__":
    # Erlaubt das Ausführen mit Kommandozeilen-Argument, z.B. python run_predictor.py daily
    parser = argparse.ArgumentParser(description="KI-Signal-Generator für verschiedene Handelsstrategien.")
    parser.add_argument(
        "strategy", 
        type=str, 
        choices=['daily', 'swing', 'genius'], 
        help="Die auszuführende Strategie (daily, swing, genius)."
    )
    args = parser.parse_args()
    generate_and_store_predictions(args.strategy)