# backend/run_backtester.py (Finale, synchronisierte Version)
import pandas as pd
import numpy as np
import joblib
import json
import os
from sqlalchemy import text

# Wichtig: Wir importieren die Konfiguration direkt aus dem Master-Controller,
# um sicherzustellen, dass die Logik zu 100% identisch ist.
from master_controller import STRATEGIES, SYMBOLS
from database import engine

def run_backtest_for_strategy(df_symbol, strategy_name, config, conn):
    """Führt einen Backtest für eine einzelne Strategie und ein Symbol durch."""
    print(f"-- Starte Backtest für {strategy_name.upper()}...")
    
    # 1. Lade das passende Modell
    symbol = df_symbol['symbol'].iloc[0]
    model_path = f"models/model_{strategy_name}_{symbol.replace('/', '')}.pkl"
    if not os.path.exists(model_path):
        print(f"Modell {model_path} nicht gefunden.")
        return None
    
    # Hier laden wir das Modell und die dazugehörigen Daten
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']

    # 2. Berechne die Features für den gesamten Datensatz
    # Wir verwenden exakt dieselbe Funktion wie im Master-Controller
    df_features = config['feature_func'](df_symbol.copy())
    df_features.dropna(inplace=True)

    if not all(f in df_features.columns for f in features):
        print(f"Fehlende Features im Datensatz. Benötigt: {features}")
        return None

    # 3. Generiere Signale für den gesamten Zeitraum
    X = df_features[features]
    X_scaled = scaler.transform(X)
    df_features['signal'] = model.predict(X_scaled)

    # 4. Simuliere die Trades und berechne die Renditen
    df_features['daily_return'] = df_features['close'].pct_change()

    df_features['strategy_return'] = np.where(df_features['signal'] == 1, df_features['daily_return'].shift(-1), 0)
    df_features['strategy_return'] = np.where(df_features['signal'] == 0, -df_features['daily_return'].shift(-1), df_features['strategy_return'])

    # 5. Berechne die Performance-Metriken
    trades = df_features[df_features['signal'] != 2]
    if trades.empty:
        return {'Symbol': symbol, 'Gesamtrendite_%': 0, 'Gewinnrate_%': 0, 'Anzahl_Trades': 0}

    number_of_trades = len(trades)
    winning_trades = trades[trades['strategy_return'] > 0]
    win_rate = (len(winning_trades) / number_of_trades) * 100 if number_of_trades > 0 else 0
    
    cumulative_return = (1 + df_features['strategy_return']).cumprod()
    # Sicherstellen, dass wir keine NaN-Werte haben
    last_valid_return = cumulative_return.dropna().iloc[-1] if not cumulative_return.dropna().empty else 1
    total_return_pct = (last_valid_return - 1) * 100
    
    print(f"Ergebnis: {total_return_pct:.2f}% Rendite, {win_rate:.2f}% Gewinnrate")

    return {
        'Symbol': symbol,
        'Gesamtrendite_%': round(total_return_pct, 2),
        'Gewinnrate_%': round(win_rate, 2),
        'Anzahl_Trades': number_of_trades
    }

def main():
    """Hauptfunktion zum Ausführen aller Backtests."""
    all_results = {'daily': [], 'swing': [], 'genius': []}
    
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n{'='*20}\nLade Daten für Backtest von {symbol}...\n{'='*20}")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp")
            df_symbol = pd.read_sql_query(query, conn, params={'symbol': symbol})

            if df_symbol.empty:
                print(f"Keine Daten für {symbol} gefunden.")
                continue
            
            # Wichtig: Dem DataFrame das korrekte Symbol zuweisen
            df_symbol['symbol'] = symbol

            for strategy_name, config in STRATEGIES.items():
                result = run_backtest_for_strategy(df_symbol.copy(), strategy_name, config, conn)
                if result:
                    all_results[strategy_name].append(result)

    with open('backtest_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\n✅ Backtest abgeschlossen und Ergebnisse in 'backtest_results.json' gespeichert.")

if __name__ == '__main__':
    main()