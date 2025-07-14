# backend/master_controller.py (Die finale, unzerstörbare LSTM-Version)
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
from sklearn.model_selection import train_test_split # HIER IST DIE KORREKTUR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Dropout

# Datenbank-Anbindung & Cloud
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from google.cloud import storage
from database import engine, predictions

# ==============================================================================
# 1. ZENTRALE KONFIGURATION
# ==============================================================================
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
SEQUENCE_LENGTH = 60 

LSTM_STRATEGY = {
    'name': 'genius_lstm',
    'features': ['close', 'RSI', 'MACD_diff', 'WilliamsR', 'ATR'],
    'feature_func': lambda df: df.assign(
        RSI=ta.momentum.rsi(df['close'], window=14),
        MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9),
        WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14),
        ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    )
}

# ==============================================================================
# 2. HILFSFUNKTIONEN (CLOUD & DATEN)
# ==============================================================================

def upload_to_gcs(source_file_name, destination_blob_name):
    if not GCS_BUCKET_NAME: print("GCS_BUCKET_NAME nicht konfiguriert."); return
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"✅ Datei {source_file_name} nach gs://{GCS_BUCKET_NAME}/{destination_blob_name} hochgeladen.")
    except Exception as e:
        print(f"❌ Fehler beim GCS-Upload: {e}")

def download_from_gcs(source_blob_name, destination_file_name):
    if not GCS_BUCKET_NAME: print("GCS_BUCKET_NAME nicht konfiguriert."); return
    try:
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    except Exception as e:
        print(f"❌ Fehler beim GCS-Download von {source_blob_name}: {e}")
        raise

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
# 3. KERNFUNKTIONEN (TRAIN, BACKTEST, PREDICT)
# ==============================================================================

# In backend/master_controller.py -> die Funktion train_lstm_model ersetzen

def train_lstm_model():
    print("=== STARTE LSTM MODELL-TRAINING (MIT EARLY STOPPING) ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für {symbol}...")
            df_raw = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp"), conn, params={'symbol': symbol})
            if len(df_raw) < SEQUENCE_LENGTH + 50: continue

            print(f"--- Trainiere LSTM für {symbol} ---")
            try:
                df_features = LSTM_STRATEGY['feature_func'](df_raw.copy())
                X, y, scaler = prepare_data_for_lstm(df_features, LSTM_STRATEGY['features'])
                if X is None: 
                    print("Nicht genügend Daten nach Bereinigung.")
                    continue
                
                # Wir teilen die Daten auf, um einen "Prüfungsbogen" (Validierungsset) zu haben
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
                y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=3)

                model = Sequential([
                    Input(shape=(X.shape[1], X.shape[2])),
                    LSTM(units=50, return_sequences=True), Dropout(0.2),
                    LSTM(units=50, return_sequences=False), Dropout(0.2),
                    Dense(units=25), 
                    Dense(units=3, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                # NEU: Der "Gedulds-Wächter" (Early Stopping)
                # Er stoppt das Training, wenn sich der Fehler auf dem Prüfungsbogen (val_loss) 
                # für 10 Runden (patience=10) nicht verbessert.
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                )
                
                print("Starte intelligentes Training... (stoppt automatisch, wenn optimal)")
                # Wir erhöhen die Epochen auf eine hohe Zahl (z.B. 200), da die KI sowieso vorher aufhört.
                model.fit(
                    X_train, 
                    y_train_cat, 
                    epochs=200, 
                    batch_size=32, 
                    validation_data=(X_val, y_val_cat),
                    callbacks=[early_stopping], # Hier wird der Wächter aktiviert
                    verbose=1
                )
                
                base_path = f"{MODELS_DIR}/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                model.save(f"{base_path}.keras")
                joblib.dump(scaler, f"{base_path}_scaler.pkl")
                print(f"✅ Optimales LSTM-Modell für {symbol} gespeichert.")
                
                # Optional: Lade die fertigen Modelle in die Cloud
                # upload_to_gcs(...)

            except Exception as e: 
                print(f"FEHLER: {e}")
                
    print("\n=== MODELL-TRAINING ABGESCHLOSSEN ===")

def backtest_lstm_model():
    print("=== STARTE LSTM BACKTEST (VON GCS) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für Backtest von {symbol}...")
            df_full = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp"), conn, params={'symbol': symbol})
            if df_full.empty: continue
            print(f"--- Backteste LSTM für {symbol} ---")
            try:
                base_path_gcs = f"models/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                base_path_local = f"{MODELS_DIR}/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                download_from_gcs(f"{base_path_gcs}.keras", f"{base_path_local}.keras")
                download_from_gcs(f"{base_path_gcs}_scaler.pkl", f"{base_path_local}_scaler.pkl")
                
                model = load_model(f"{base_path_local}.keras")
                scaler = joblib.load(f"{base_path_local}_scaler.pkl")
                
                df_features = LSTM_STRATEGY['feature_func'](df_full.copy()).dropna()
                all_scaled_data = scaler.transform(df_features[LSTM_STRATEGY['features']])
                X_backtest = []
                for i in range(SEQUENCE_LENGTH, len(all_scaled_data)):
                    X_backtest.append(all_scaled_data[i-SEQUENCE_LENGTH:i])
                
                if not X_backtest: print("Keine Daten für Backtest-Vorhersage."); continue
                    
                predictions_proba = model.predict(np.array(X_backtest))
                predicted_classes = np.argmax(predictions_proba, axis=1)
                
                df_trade = df_features.iloc[SEQUENCE_LENGTH:].copy()
                df_trade['signal'] = predicted_classes
                
                df_trade['daily_return'] = df_trade['close'].pct_change()
                long_trades = df_trade[df_trade['signal'] == 1]
                short_trades = df_trade[df_trade['signal'] == 0]
                long_returns = long_trades['daily_return'].shift(-1)
                short_returns = -short_trades['daily_return'].shift(-1)
                winning_longs = long_returns[long_returns > 0]
                winning_shorts = short_returns[short_returns > 0]
                long_win_rate = (len(winning_longs) / len(long_trades) * 100) if not long_trades.empty else 0
                short_win_rate = (len(winning_shorts) / len(short_trades) * 100) if not short_trades.empty else 0
                
                print(f"  -> KAUF-Signale: {len(long_trades)} Trades | Gewinnrate: {long_win_rate:.2f}%")
                print(f"  -> VERKAUF-Signale: {len(short_trades)} Trades | Gewinnrate: {short_win_rate:.2f}%")
            except Exception as e:
                print(f"Ein FEHLER ist aufgetreten: {e}")
    print("\n✅ Backtest abgeschlossen.")

def predict_with_lstm():
    print("=== STARTE LSTM SIGNAL-GENERATOR (VON GCS) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Live-Daten für {symbol}...")
            df_live = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 100"), conn, params={'symbol': symbol})
            df_live = df_live.sort_values(by='timestamp').reset_index(drop=True)
            if len(df_live) < SEQUENCE_LENGTH: continue
            print(f"--- Generiere LSTM-Signal für {symbol} ---")
            try:
                base_path_gcs = f"models/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                base_path_local = f"{MODELS_DIR}/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                download_from_gcs(f"{base_path_gcs}.keras", f"{base_path_local}.keras")
                download_from_gcs(f"{base_path_gcs}_scaler.pkl", f"{base_path_local}_scaler.pkl")
                
                model = load_model(f"{base_path_local}.keras")
                scaler = joblib.load(f"{base_path_local}_scaler.pkl")
                
                df_features = LSTM_STRATEGY['feature_func'](df_live.copy()).dropna()
                last_sequence = df_features[LSTM_STRATEGY['features']].tail(SEQUENCE_LENGTH)
                last_sequence_scaled = scaler.transform(last_sequence)
                X_predict = np.array([last_sequence_scaled])
                prediction_proba = model.predict(X_predict)[0]
                confidence = round(np.max(prediction_proba) * 100, 2)
                predicted_class = np.argmax(prediction_proba)
                signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(predicted_class, "Unbekannt")
                price = df_features.iloc[-1]['close']
                atr_value = df_features.iloc[-1]['ATR']
                take_profit, stop_loss = (price + (2.5 * atr_value), price - (1.5 * atr_value)) if signal == "Kaufen" else (price - (2.5 * atr_value), price + (1.5 * atr_value)) if signal == "Verkaufen" else (None, None)
                update_data = {'symbol': symbol, 'strategy': 'genius_lstm', 'signal': signal, 'confidence': confidence, 'entry_price': price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'last_updated': datetime.now(timezone.utc)}
                stmt = insert(predictions).values(update_data); stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_=update_data); conn.execute(stmt); conn.commit()
                print(f"✅ Signal gespeichert: {signal} (Konfidenz: {confidence}%)")
            except Exception as e: print(f"FEHLER: {e}")
    print("\n=== SIGNAL-GENERATOR ABGESCHLOSSEN ===")

# ==============================================================================
# 4. STEUERUNG
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master-Controller für die Finanz-App.")
    parser.add_argument("mode", choices=['train-lstm', 'predict-lstm', 'backtest-lstm'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'train-lstm':
        train_lstm_model()
    elif args.mode == 'predict-lstm':
        predict_with_lstm()
    elif args.mode == 'backtest-lstm':
        backtest_lstm_model()