# backend/master_controller.py (Finale All-in-One-Lösung)
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
# 1. ZENTRALE KONFIGURATION (ALLES AN EINEM ORT)
# ==============================================================================
SYMBOLS = ["BTC/USD", "XAU/USD"]

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
# 2. FUNKTIONEN (TRAIN, BACKTEST, PREDICT)
# ==============================================================================

def create_target(df, period=5):
    df['future_return'] = df['close'].pct_change(period).shift(-period)
    conditions = [(df['future_return'] > 0.02), (df['future_return'] < -0.02)]
    choices = [1, 0]
    df['target'] = np.select(conditions, choices, default=2)
    return df

def train_all_models():
    print("=== STARTE MODELL-TRAINING ===")
    os.makedirs('models', exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für {symbol}...")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp")
            df_raw = pd.read_sql_query(query, conn, params={'symbol': symbol})
            if len(df_raw) < 250:
                print(f"Nicht genügend Daten für {symbol}.")
                continue

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
                    model_path = f"models/model_{name}_{symbol.replace('/', '')}.pkl"
                    joblib.dump({'model': model, 'scaler': scaler, 'features': features}, model_path)
                    print(f"✅ Modell erfolgreich gespeichert: {model_path}")
                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")
    print("\n=== MODELL-TRAINING ABGESCHLOSSEN ===")

def predict_all_signals():
    print("=== STARTE SIGNAL-GENERATOR ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Live-Daten für {symbol}...")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400")
            df_live = pd.read_sql_query(query, conn, params={'symbol': symbol})
            df_live = df_live.sort_values(by='timestamp').reset_index(drop=True)
            if len(df_live) < 250:
                print(f"Nicht genügend Live-Daten für {symbol}.")
                continue

            for name, config in STRATEGIES.items():
                print(f"--- Generiere Signal: {name.upper()} für {symbol} ---")
                try:
                    model_path = f"models/model_{name}_{symbol.replace('/', '')}.pkl"
                    model_data = joblib.load(model_path)
                    model, scaler, features = model_data['model'], model_data['scaler'], model_data['features']
                    
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