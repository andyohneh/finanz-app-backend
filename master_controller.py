# backend/master_controller.py (Finale Version mit Google Cloud Anbindung)
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

# NEU: Google Cloud Bibliotheken
from google.cloud import storage

# Machine Learning Bibliotheken
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, predictions

# ==============================================================================
# 1. ZENTRALE KONFIGURATION
# ==============================================================================
load_dotenv()
# NEU: Google Cloud Konfiguration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
INITIAL_CAPITAL = 100

STRATEGIES = {
    'daily': {'features': ['RSI', 'ATR', 'MACD_diff', 'Stoch'], 'feature_func': lambda df: df.assign(RSI=ta.momentum.rsi(df['close'], window=14), ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14), MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9), Stoch=ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3))},
    'swing': {'features': ['BB_Width', 'RSI', 'WilliamsR', 'SMA_20'], 'feature_func': lambda df: df.assign(RSI=ta.momentum.rsi(df['close'], window=14), SMA_20=ta.trend.sma_indicator(df['close'], window=20), BB_Width=ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2), WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14))},
    'genius': {'features': ['ATR', 'ADX', 'CCI', 'WilliamsR'], 'feature_func': lambda df: df.assign(ADX=ta.trend.adx(df['high'], df['low'], df['close'], window=14), ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14), WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14), CCI=ta.trend.cci(df['high'], df['low'], df['close'], window=20))}
}

# ==============================================================================
# 2. HILFSFUNKTIONEN (CLOUD & DATEN)
# ==============================================================================

def upload_to_gcs(source_file_name, destination_blob_name):
    """Lädt eine lokale Datei in den Google Cloud Storage Bucket hoch."""
    if not GCS_BUCKET_NAME: print("GCS_BUCKET_NAME nicht konfiguriert."); return
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"✅ Datei {source_file_name} erfolgreich nach gs://{GCS_BUCKET_NAME}/{destination_blob_name} hochgeladen.")
    except Exception as e:
        print(f"❌ Fehler beim GCS-Upload: {e}")

def download_from_gcs(source_blob_name, destination_file_name):
    """Lädt eine Datei aus dem GCS Bucket herunter."""
    if not GCS_BUCKET_NAME: print("GCS_BUCKET_NAME nicht konfiguriert."); return
    try:
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"✅ Datei gs://{GCS_BUCKET_NAME}/{source_blob_name} erfolgreich heruntergeladen.")
    except Exception as e:
        print(f"❌ Fehler beim GCS-Download von {source_blob_name}: {e}")
        raise # Fehler weitergeben, damit der Prozess abbricht

def create_target(df, period=5):
    df['future_return'] = df['close'].pct_change(period).shift(-period)
    conditions = [(df['future_return'] > 0.02), (df['future_return'] < -0.02)]
    choices = [1, 0]; df['target'] = np.select(conditions, choices, default=2)
    return df

# ==============================================================================
# 3. KERNFUNKTIONEN (TRAIN, BACKTEST, PREDICT)
# ==============================================================================

def train_all_models():
    print("=== STARTE MODELL-TRAINING (LOKAL) & UPLOAD ZU GCS ===")
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
                    features = config['features']; X = df[features]; y = df['target']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    scaler = StandardScaler().fit(X_train); X_train_scaled = scaler.transform(X_train); X_test_scaled = scaler.transform(X_test)
                    
                    model = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbosity=-1).fit(X_train_scaled, y_train, feature_name=features)
                    
                    y_pred = model.predict(X_test_scaled)
                    accuracy = (y_pred == y_test).mean()
                    print(f"Modell-Genauigkeit: {accuracy:.2f}")

                    base_path_local = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    base_path_gcs = f"models/model_{name}_{symbol.replace('/', '')}" # Pfad im Bucket

                    joblib.dump(model, f"{base_path_local}_model.pkl")
                    joblib.dump(scaler, f"{base_path_local}_scaler.pkl")
                    with open(f"{base_path_local}_features.json", 'w') as f: json.dump(features, f)
                    
                    # Lade die fertigen Modelle in die Cloud
                    upload_to_gcs(f"{base_path_local}_model.pkl", f"{base_path_gcs}_model.pkl")
                    upload_to_gcs(f"{base_path_local}_scaler.pkl", f"{base_path_gcs}_scaler.pkl")
                    upload_to_gcs(f"{base_path_local}_features.json", f"{base_path_gcs}_features.json")
                    
                except Exception as e: print(f"Ein FEHLER ist aufgetreten: {e}")
    print("\n=== MODELL-TRAINING & UPLOAD ABGESCHLOSSEN ===")


def predict_all_signals():
    print("=== STARTE SIGNAL-GENERATOR (VON GCS) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Live-Daten für {symbol}...")
            df_live = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400"), conn, params={'symbol': symbol})
            df_live = df_live.sort_values(by='timestamp').reset_index(drop=True)
            if len(df_live) < 250: continue

            for name, config in STRATEGIES.items():
                print(f"--- Generiere Signal: {name.upper()} für {symbol} ---")
                try:
                    # Lade die neuesten Modelle aus der Cloud herunter
                    base_path_gcs = f"models/model_{name}_{symbol.replace('/', '')}"
                    base_path_local = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    
                    download_from_gcs(f"{base_path_gcs}_model.pkl", f"{base_path_local}_model.pkl")
                    download_from_gcs(f"{base_path_gcs}_scaler.pkl", f"{base_path_local}_scaler.pkl")
                    download_from_gcs(f"{base_path_gcs}_features.json", f"{base_path_local}_features.json")

                    model = joblib.load(f"{base_path_local}_model.pkl")
                    scaler = joblib.load(f"{base_path_local}_scaler.pkl")
                    with open(f"{base_path_local}_features.json", 'r') as f: features = json.load(f)
                    
                    df_features = config['feature_func'](df_live.copy()).dropna()
                    X_predict = df_features[features].tail(1)
                    X_scaled = scaler.transform(X_predict.values)
                    prediction = model.predict(X_predict)
                    
                    signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(int(prediction[0]))
                    price = df_features.iloc[-1]['close']
                    take_profit, stop_loss = (price * 1.05, price * 0.98) if signal == "Kaufen" else (price * 0.95, price * 1.02) if signal == "Verkaufen" else (None, None)
                    update_data = {'symbol': symbol, 'strategy': name, 'signal': signal, 'entry_price': price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'last_updated': datetime.now(timezone.utc)}
                    
                    stmt = insert(predictions).values(update_data)
                    stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_=update_data)
                    conn.execute(stmt)
                    conn.commit()
                    print(f"✅ Signal erfolgreich gespeichert.")
                except Exception as e: print(f"FEHLER: {e}")
    print("\n=== SIGNAL-GENERATOR ABGESCHLOSSEN ===")


# ==============================================================================
# 4. STEUERUNG
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master-Controller für die Finanz-App.")
    parser.add_argument("mode", choices=['train', 'predict'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_all_models()
    elif args.mode == 'predict':
        predict_all_signals()