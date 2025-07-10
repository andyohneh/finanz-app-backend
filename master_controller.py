# backend/master_controller.py (Finale, unzerstörbare Version)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, timezone
import argparse

from database import engine, predictions

# ==============================================================================
# 1. ZENTRALE KONFIGURATION
# ==============================================================================
SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"

STRATEGIES = {
    'daily': {
        'features': ['RSI', 'SMA_50', 'SMA_200', 'MACD_diff'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_50=ta.trend.sma_indicator(df['close'], window=50),
            SMA_200=ta.trend.sma_indicator(df['close'], window=200),
            MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
        )
    },
    'swing': {
        'features': ['RSI', 'SMA_20', 'EMA_50', 'BB_Width'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_20=ta.trend.sma_indicator(df['close'], window=20),
            EMA_50=ta.trend.ema_indicator(df['close'], window=50),
            BB_Width=ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
        )
    },
    'genius': {
        'features': ['ADX', 'ATR', 'Stoch_RSI', 'WilliamsR'],
        'feature_func': lambda df: df.assign(
            ADX=ta.trend.adx(df['high'], df['low'], df['close'], window=14),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14),
            Stoch_RSI=ta.momentum.stochrsi(df['close'], window=14, smooth1=3, smooth2=3),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
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
    print("=== STARTE MODELL-TRAINING (ROBUSTER MODUS) ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für {symbol}...")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp")
            df_raw = pd.read_sql_query(query, conn, params={'symbol': symbol})
            if len(df_raw) < 250: continue

            for name, config in STRATEGIES.items():
                print(f"--- Trainiere Modell: {name.upper()} für {symbol} ---")
                try:
                    df = config['feature_func'](df_raw.copy())
                    df = create_target(df)
                    df.dropna(inplace=True)
                    features = config['features']
                    X = df[features]
                    y = df['target']
                    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    scaler = StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train_scaled, y_train)
                    
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    joblib.dump(model, f"{base_path}_model.pkl")
                    joblib.dump(scaler, f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'w') as f:
                        json.dump(features, f)
                        
                    print(f"✅ Modell, Scaler und Features erfolgreich gespeichert.")
                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")
    print("\n=== MODELL-TRAINING ABGESCHLOSSEN ===")

def backtest_all_models():
    print("=== STARTE BACKTESTING (ROBUSTER MODUS) ===")
    all_results = {'daily': [], 'swing': [], 'genius': []}
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
                    
                    df_features = config['feature_func'](df_symbol.copy())
                    df_features.dropna(inplace=True)
                    
                    X = df_features[features]
                    X_scaled = scaler.transform(X)
                    df_features['signal'] = model.predict(X_scaled)
                    
                    df_features['daily_return'] = df_features['close'].pct_change()
                    df_features['strategy_return'] = np.where(df_features['signal'] == 1, df_features['daily_return'].shift(-1), np.where(df_features['signal'] == 0, -df_features['daily_return'].shift(-1), 0))
                    
                    trades = df_features[df_features['signal'] != 2]
                    total_return_pct = (df_features['strategy_return'].sum() * 100)
                    win_rate = (len(trades[trades['strategy_return'] > 0]) / len(trades) * 100) if not trades.empty else 0
                    
                    all_results[name].append({'Symbol': symbol, 'Gesamtrendite_%': round(total_return_pct, 2), 'Gewinnrate_%': round(win_rate, 2), 'Anzahl_Trades': len(trades)})
                    print(f"Ergebnis: {total_return_pct:.2f}% Rendite, {win_rate:.2f}% Gewinnrate")
                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")

    with open('backtest_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n✅ Backtest abgeschlossen und Ergebnisse gespeichert.")

def predict_all_signals():
    print("=== STARTE SIGNAL-GENERATOR (ROBUSTER MODUS) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Live-Daten für {symbol}...")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400")
            df_live = pd.read_sql_query(query, conn, params={'symbol': symbol})
            df_live = df_live.sort_values(by='timestamp').reset_index(drop=True)
            if len(df_live) < 250: continue

            for name, config in STRATEGIES.items():
                print(f"--- Generiere Signal: {name.upper()} für {symbol} ---")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model = joblib.load(f"{base_path}_model.pkl")
                    scaler = joblib.load(f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'r') as f:
                        features = json.load(f)
                    
                    df_features = config['feature_func'](df_live.copy())
                    df_features.dropna(inplace=True)
                    
                    X_predict = df_features[features].tail(1)
                    X_scaled = scaler.transform(X_predict)
                    prediction = model.predict(X_scaled)
                    
                    signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(int(prediction[0]))
                    price = df_features.iloc[-1]['close']
                    
                    update_data = {'symbol': symbol, 'strategy': name, 'signal': signal, 'entry_price': price, 'take_profit': price * 1.05, 'stop_loss': price * 0.98, 'last_updated': datetime.now(timezone.utc)}
                    
                    stmt = insert(predictions).values(update_data)
                    stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_=update_data)
                    conn.execute(stmt)
                    conn.commit()
                    print(f"✅ Signal erfolgreich gespeichert.")
                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")
    print("\n=== SIGNAL-GENERATOR ABGESCHLOSSEN ===")

# ==============================================================================
# 4. STEUERUNG
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master-Controller für Training, Backtesting und Vorhersage.")
    parser.add_argument("mode", choices=['train', 'backtest', 'predict'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_all_models()
    elif args.mode == 'backtest':
        backtest_all_models()
    elif args.mode == 'predict':
        predict_all_signals()