# backend/master_controller.py (Die finale, vollständige Version)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
import requests
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

# Machine Learning Bibliotheken
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Dropout

# Datenbank-Anbindung
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, predictions

# ==============================================================================
# 1. ZENTRALE KONFIGURATION
# ==============================================================================
load_dotenv()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
INITIAL_CAPITAL = 100
SEQUENCE_LENGTH = 60

STRATEGIES = {
    'daily_lstm': { 'features': ['close', 'SMA_50', 'RSI'], 'feature_func': lambda df: df.assign(SMA_50=ta.trend.sma_indicator(df['close'], window=50), RSI=ta.momentum.rsi(df['close'], window=14)) },
    'genius_lstm': { 'features': ['close', 'RSI', 'MACD_diff', 'WilliamsR', 'ATR'], 'feature_func': lambda df: df.assign(RSI=ta.momentum.rsi(df['close'], window=14), MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9), WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14), ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)) }
}

# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================
def load_historical_data(symbol, conn):
    query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
    return pd.read_sql_query(query, conn, params={'symbol': symbol})

def prepare_data_for_lstm(df, features):
    future_pct_change = df['close'].pct_change(7).shift(-7)
    df['target'] = 2
    df.loc[future_pct_change > 0.03, 'target'] = 1
    df.loc[future_pct_change < -0.03, 'target'] = 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + ['target'], inplace=True)
    if len(df) < SEQUENCE_LENGTH + 1: return None, None, None
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQUENCE_LENGTH:i])
        y.append(df['target'].iloc[i])
    return np.array(X), np.array(y), scaler

# ==============================================================================
# 3. KERNFUNKTIONEN
# ==============================================================================
def train_all_models():
    print("=== STARTE MODELL-TRAINING (LOKAL AUF DEINEM PC) ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für {symbol}...")
            df_raw = load_historical_data(symbol, conn)
            if len(df_raw) < SEQUENCE_LENGTH + 50: continue
            for name, config in STRATEGIES.items():
                print(f"--- Trainiere '{name}' für {symbol} ---")
                try:
                    df_features = config['feature_func'](df_raw.copy())
                    X, y, scaler = prepare_data_for_lstm(df_features, config['features'])
                    if X is None: continue
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
                    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
                    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=3)
                    model = Sequential([Input(shape=(X.shape[1], X.shape[2])), LSTM(units=50, return_sequences=True), Dropout(0.2), LSTM(units=50), Dropout(0.2), Dense(units=25), Dense(units=3, activation='softmax')])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    model.fit(X_train, y_train_cat, epochs=100, batch_size=32, validation_data=(X_val, y_val_cat), callbacks=[early_stopping], verbose=1)
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model.save(f"{base_path}.keras")
                    joblib.dump(scaler, f"{base_path}_scaler.pkl")
                    print(f"✅ Modell für {name} erfolgreich gespeichert.")
                except Exception as e: print(f"FEHLER: {e}")
    print("\n=== LOKALES MODELL-TRAINING ABGESCHLOSSEN ===")

def backtest_all_models():
    print("=== STARTE BACKTESTING ===")
    all_results = {'daily_lstm': [], 'genius_lstm': []}
    equity_curves = {'daily_lstm': {}, 'genius_lstm': {}}
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für Backtest von {symbol}...")
            df_full = load_historical_data(symbol, conn)
            if df_full.empty: continue
            for name, config in STRATEGIES.items():
                print(f"-- Starte Backtest für {name.upper()}...")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    if not os.path.exists(f"{base_path}.keras"):
                        print(f"Modell für {name} nicht gefunden. Bitte zuerst trainieren.")
                        continue
                        
                    model = load_model(f"{base_path}.keras")
                    scaler = joblib.load(f"{base_path}_scaler.pkl")
                    df_features = config['feature_func'](df_full.copy()).dropna()
                    
                    all_scaled_data = scaler.transform(df_features[config['features']])
                    X_backtest = []
                    for i in range(SEQUENCE_LENGTH, len(all_scaled_data)):
                        X_backtest.append(all_scaled_data[i-SEQUENCE_LENGTH:i])
                    
                    if not X_backtest: continue
                    
                    predicted_classes = np.argmax(model.predict(np.array(X_backtest)), axis=1)
                    df_trade = df_features.iloc[SEQUENCE_LENGTH:].copy()
                    df_trade['signal'] = predicted_classes
                    
                    df_trade['daily_return'] = df_trade['close'].pct_change()
                    df_trade['strategy_return'] = np.where(df_trade['signal'].shift(1) == 1, df_trade['daily_return'], np.where(df_trade['signal'].shift(1) == 0, -df_trade['daily_return'], 0))
                    
                    df_trade['equity_curve'] = INITIAL_CAPITAL * (1 + df_trade['strategy_return']).cumprod()
                    equity_curves[name][symbol] = {'dates': df_trade['timestamp'].dt.strftime('%Y-%m-%d').tolist(),'values': df_trade['equity_curve'].fillna(INITIAL_CAPITAL).round(2).tolist()}
                    
                    total_return_pct = ((df_trade['equity_curve'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                    trades = df_trade[df_trade['signal'] != 2]
                    win_rate = (len(trades[trades['strategy_return'] > 0]) / len(trades) * 100) if not trades.empty else 0
                    all_results[name].append({'Symbol': symbol, 'Gesamtrendite_%': round(total_return_pct, 2), 'Gewinnrate_%': round(win_rate, 2), 'Anzahl_Trades': len(trades)})
                    print(f"Ergebnis: {total_return_pct:.2f}% Rendite")
                except Exception as e: print(f"FEHLER: {e}")
    with open('backtest_results.json', 'w') as f: json.dump(all_results, f, indent=4)
    with open('equity_curves.json', 'w') as f: json.dump(equity_curves, f, indent=4)
    print("\n✅ Backtest abgeschlossen.")

def predict_all_signals():
    print("=== STARTE SIGNAL-GENERATOR (AUF RENDER) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Verarbeite {symbol} ---")
            try:
                df_live = load_historical_data(symbol, conn).tail(120).copy()
                if len(df_live) < SEQUENCE_LENGTH: continue
                for name, config in STRATEGIES.items():
                    print(f"-> Generiere '{name}' Signal...")
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    if not os.path.exists(f"{base_path}.keras"):
                        print(f"Modell für {name} nicht gefunden. Bitte zuerst auf dem PC trainieren und hochladen.")
                        continue
                        
                    model = load_model(f"{base_path}.keras")
                    scaler = joblib.load(f"{base_path}_scaler.pkl")
                    df_features = config['feature_func'](df_live).dropna()
                    last_sequence = df_features[config['features']].tail(SEQUENCE_LENGTH)
                    if len(last_sequence) < SEQUENCE_LENGTH: continue
                    last_sequence_scaled = scaler.transform(last_sequence)
                    X_predict = np.array([last_sequence_scaled])
                    prediction_proba = model.predict(X_predict)[0]
                    confidence = round(np.max(prediction_proba) * 100, 2)
                    signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(np.argmax(prediction_proba))
                    price = df_features.iloc[-1]['close']
                    update_data = {'symbol': symbol, 'strategy': name, 'signal': signal, 'confidence': confidence, 'entry_price': price, 'last_updated': datetime.now(timezone.utc)}
                    stmt = insert(predictions).values(update_data)
                    stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_=update_data)
                    conn.execute(stmt)
                    conn.commit()
                    print(f"✅ Signal für '{name}' gespeichert: {signal} ({confidence}%)")
            except Exception as e:
                print(f"Ein FEHLER bei {symbol}: {e}")
    print("\n=== SIGNAL-GENERATOR ABGESCHLOSSEN ===")

# ==============================================================================
# 4. STEUERUNG
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master-Controller für die Finanz-App.")
    # Der Backtest-Modus ist wieder da!
    parser.add_argument("mode", choices=['train', 'backtest', 'predict'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_all_models()
    elif args.mode == 'backtest':
        backtest_all_models()
    elif args.mode == 'predict':
        predict_all_signals()