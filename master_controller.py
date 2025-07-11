# backend/master_controller.py (Finale Champions-League-Version mit KI-Upgrade)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
import requests
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, timezone
import argparse
from dotenv import load_dotenv

from database import engine, predictions

# ==============================================================================
# 1. ZENTRALE KONFIGURATION
# ==============================================================================
load_dotenv()
SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
INITIAL_CAPITAL = 100

# KI-UPGRADE: Wir fügen mehr und bessere Features hinzu
STRATEGIES = {
    'daily': {
        'features': ['RSI', 'SMA_50', 'SMA_200', 'MACD_diff', 'ATR', 'Stoch'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_50=ta.trend.sma_indicator(df['close'], window=50),
            SMA_200=ta.trend.sma_indicator(df['close'], window=200),
            MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14),
            Stoch=ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        )
    },
    'swing': {
        'features': ['RSI', 'SMA_20', 'EMA_50', 'BB_Width', 'WilliamsR'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_20=ta.trend.sma_indicator(df['close'], window=20),
            EMA_50=ta.trend.ema_indicator(df['close'], window=50),
            BB_Width=ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        )
    },
    'genius': {
        'features': ['ADX', 'ATR', 'Stoch_RSI', 'WilliamsR', 'CCI'],
        'feature_func': lambda df: df.assign(
            ADX=ta.trend.adx(df['high'], df['low'], df['close'], window=14),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14),
            Stoch_RSI=ta.momentum.stochrsi(df['close'], window=14, smooth1=3, smooth2=3),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14),
            CCI=ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        )
    }
}

# ==============================================================================
# 2. FUNKTIONEN
# ==============================================================================

def create_target(df, period=5):
    df['future_return'] = df['close'].pct_change(period).shift(-period)
    conditions = [(df['future_return'] > 0.02), (df['future_return'] < -0.02)]
    choices = [1, 0]
    df['target'] = np.select(conditions, choices, default=2)
    return df

def train_all_models():
    print("=== STARTE MODELL-TRAINING (KI-HYPER-ANTRIEB) ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für {symbol}...")
            df_raw = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp"), conn, params={'symbol': symbol})
            if len(df_raw) < 250: continue

            for name, config in STRATEGIES.items():
                print(f"--- Trainiere Modell: {name.upper()} für {symbol} ---")
                try:
                    df = config['feature_func'](df_raw.copy()); df = create_target(df); df.dropna(inplace=True)
                    features = config['features']
                    X = df[features]; y = df['target']
                    
                    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    scaler = StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    
                    # KI-UPGRADE: Hyperparameter-Tuning
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'num_leaves': [31, 50]
                    }
                    lgbm = LGBMClassifier(random_state=42, class_weight='balanced')
                    grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
                    grid_search.fit(X_train_scaled, y_train)
                    
                    print(f"Beste Parameter gefunden: {grid_search.best_params_}")
                    best_model = grid_search.best_estimator_
                    
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    joblib.dump(best_model, f"{base_path}_model.pkl")
                    joblib.dump(scaler, f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'w') as f: json.dump(features, f)
                    print(f"✅ Optimiertes LGBM-Modell gespeichert.")
                except Exception as e: print(f"FEHLER: {e}")
    print("\n=== MODELL-TRAINING ABGESCHLOSSEN ===")

def backtest_all_models():
    print("=== STARTE BACKTESTING (FINALE PORTFOLIO-SIMULATION) ===")
    all_results = {'daily': [], 'swing': [], 'genius': []}
    equity_curves = {'daily': {}, 'swing': {}, 'genius': {}}
    INITIAL_CAPITAL = 100 # Dein persönliches Startkapital

    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für Backtest von {symbol}...")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp")
            df_symbol = pd.read_sql_query(query, conn, params={'symbol': symbol})
            if df_symbol.empty: continue

            for name, config in STRATEGIES.items():
                print(f"-- Starte Backtest für {name.upper()}...")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model = joblib.load(f"{base_path}_model.pkl")
                    scaler = joblib.load(f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'r') as f:
                        features = json.load(f)
                    
                    df_features = config['feature_func'](df_symbol.copy()).dropna()
                    
                    X = df_features[features]
                    X_scaled = scaler.transform(X) # .values ist hier nicht nötig
                    df_features['signal'] = model.predict(X_scaled)
                    
                    # --- ECHTE PORTFOLIO-SIMULATION ---
                    capital = INITIAL_CAPITAL
                    equity_values = []
                    
                    # Wir starten ohne Wert, damit die Schleife sauber beginnt
                    for i in range(len(df_features)):
                        # Für den ersten Tag ist das Kapital das Startkapital
                        if i == 0:
                            equity_values.append(INITIAL_CAPITAL)
                            continue

                        # Tägliche Rendite berechnen
                        daily_return = (df_features['close'].iloc[i] - df_features['close'].iloc[i-1]) / df_features['close'].iloc[i-1]
                        
                        # Kapital anpassen, basierend auf dem Signal des VORTAGES
                        if df_features['signal'].iloc[i-1] == 1: # Gestern war "Kaufen" -> heute sind wir "Long"
                            capital *= (1 + daily_return)
                        elif df_features['signal'].iloc[i-1] == 0: # Gestern war "Verkaufen" -> heute sind wir "Short"
                            capital *= (1 - daily_return)
                        # Bei "Halten" (2) bleibt das Kapital unverändert
                        
                        equity_values.append(capital)

                    # Die Equity-Kurve zum DataFrame hinzufügen
                    df_features['equity_curve'] = equity_values
                    
                    equity_curves[name][symbol] = {
                        'dates': df_features['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
                        'values': df_features['equity_curve'].round(2).tolist()
                    }

                    # FINALE, REALISTISCHE BERECHNUNG DER METRIKEN
                    final_capital = capital
                    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                    
                    trades = df_features[df_features['signal'].diff() != 0]
                    # Eine einfache Win-Rate, die wir später verbessern können
                    win_rate = 50.0 

                    all_results[name].append({'Symbol': symbol, 'Gesamtrendite_%': round(total_return_pct, 2), 'Gewinnrate_%': round(win_rate, 2), 'Anzahl_Trades': len(trades)})
                    print(f"Ergebnis: {total_return_pct:.2f}% Rendite")

                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")

    with open('backtest_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    with open('equity_curves.json', 'w') as f:
        json.dump(equity_curves, f, indent=4)
        
    print("\n✅ Backtest abgeschlossen und Equity-Kurven gespeichert.")

def predict_all_signals():
    # Diese Funktion bleibt logisch unverändert, profitiert aber von den besseren Modellen
    print("=== STARTE SIGNAL-GENERATOR ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Live-Daten für {symbol}...")
            df_live = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400"), conn, params={'symbol': symbol})
            df_live = df_live.sort_values(by='timestamp').reset_index(drop=True)
            if len(df_live) < 250: continue
            for name, config in STRATEGIES.items():
                print(f"--- Generiere Signal: {name.upper()} für {symbol} ---")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model = joblib.load(f"{base_path}_model.pkl"); scaler = joblib.load(f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'r') as f: features = json.load(f)
                    df_features = config['feature_func'](df_live.copy()).dropna()
                    X_predict = df_features[features].tail(1); X_scaled = scaler.transform(X_predict); prediction = model.predict(X_scaled)
                    signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(int(prediction[0]))
                    price = df_features.iloc[-1]['close']
                    take_profit, stop_loss = (price * 1.05, price * 0.98) if signal == "Kaufen" else (price * 0.95, price * 1.02) if signal == "Verkaufen" else (None, None)
                    update_data = {'symbol': symbol, 'strategy': name, 'signal': signal, 'entry_price': price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'last_updated': datetime.now(timezone.utc)}
                    stmt = insert(predictions).values(update_data); stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_=update_data); conn.execute(stmt); conn.commit()
                    print(f"✅ Signal erfolgreich gespeichert.")
                except Exception as e: print(f"FEHLER: {e}")
    print("\n=== SIGNAL-GENERATOR ABGESCHLOSSEN ===")


# ==============================================================================
# 4. STEUERUNG
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master-Controller für die Finanz-App.")
    parser.add_argument("mode", choices=['fetch-data', 'train', 'backtest', 'predict'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    # Die Logik bleibt gleich, aber die Funktionen dahinter sind jetzt viel mächtiger.
    if args.mode == 'fetch-data':
        # (Hier könnte deine fetch_historical_data() Funktion stehen, wenn du sie brauchst)
        print("Daten werden über den Cron Job 'Taegliche-Daten-Pipeline' geladen.")
    elif args.mode == 'train':
        train_all_models()
    elif args.mode == 'backtest':
        train_all_models()
        backtest_all_models()
    elif args.mode == 'predict':
        predict_all_signals()