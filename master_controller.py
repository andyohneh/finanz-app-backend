# backend/master_controller.py (Die finale, ultimative Version)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
import requests
import argparse
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Machine Learning Bibliotheken
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Dropout

# Datenbank-Anbindung & Cloud
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from google.cloud import storage
from google.oauth2 import service_account
from database import engine, predictions, daily_sentiment

import finnhub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==============================================================================
# 1. ZENTRALE KONFIGURATION
# ==============================================================================
load_dotenv()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_SERVICE_ACCOUNT_JSON_STR = os.getenv("GCS_SERVICE_ACCOUNT_JSON")

SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
SEQUENCE_LENGTH = 60
INITIAL_CAPITAL = 100

STRATEGIES = {
    'daily_lstm': {
        'features': ['close', 'SMA_50', 'RSI', 'sentiment_score'],
        'feature_func': lambda df: df.assign(
            SMA_50=ta.trend.sma_indicator(df['close'], window=50),
            RSI=ta.momentum.rsi(df['close'], window=14))
    },
    'genius_lstm': {
        'features': ['close', 'RSI', 'MACD_diff', 'WilliamsR', 'ATR', 'sentiment_score'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14))
    }
}

# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================
def get_gcs_client():
    if not GCS_SERVICE_ACCOUNT_JSON_STR: return storage.Client()
    credentials_info = json.loads(GCS_SERVICE_ACCOUNT_JSON_STR)
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    return storage.Client(credentials=credentials)

def upload_to_gcs(source_file_name, destination_blob_name):
    if not GCS_BUCKET_NAME: print("GCS_BUCKET_NAME nicht konfiguriert."); return
    try:
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"✅ Datei {os.path.basename(source_file_name)} hochgeladen.")
    except Exception as e: print(f"❌ Fehler beim GCS-Upload: {e}")

def download_from_gcs(source_blob_name, destination_file_name):
    if not GCS_BUCKET_NAME: print("GCS_BUCKET_NAME nicht konfiguriert."); return
    try:
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    except Exception as e:
        print(f"❌ Modell {source_blob_name} nicht in der Cloud gefunden. Breche ab.")
        raise

def load_data_with_sentiment(symbol, conn):
    query = text("SELECT h.timestamp, h.open, h.high, h.low, h.close, h.volume, COALESCE(s.sentiment_score, 0.0) as sentiment_score FROM historical_data_daily h LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = s.date WHERE h.symbol = :symbol ORDER BY h.timestamp ASC")
    df = pd.read_sql_query(query, conn, params={'symbol': symbol})
    df['sentiment_score'] = df['sentiment_score'].ffill().bfill().fillna(0.0)
    return df

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
def fetch_data():
    if not TWELVEDATA_API_KEY: print("FEHLER: TWELVEDATA_API_KEY nicht gefunden."); return
    print("=== STARTE DATEN-IMPORT (TWELVEDATA) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            try:
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                data = response.json()
                if data.get('status') == 'ok' and 'values' in data:
                    records = [{'timestamp': datetime.strptime(v['datetime'], '%Y-%m-%d'), 'symbol': symbol, 'open': float(v['open']), 'high': float(v['high']), 'low': float(v['low']), 'close': float(v['close']), 'volume': int(v.get('volume', 0))} for v in data['values']]
                    if not records: continue
                    trans = conn.begin()
                    for record in records:
                        stmt = text("INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume) VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume) ON CONFLICT (timestamp, symbol) DO NOTHING")
                        conn.execute(stmt, record)
                    trans.commit()
                    print(f"✅ {len(records)} Datensätze für {symbol} importiert.")
                else: print(f"Fehlerhafte API-Antwort: {data.get('message')}")
            except Exception as e: print(f"Ein FEHLER ist bei {symbol} aufgetreten: {e}")
    print("\n=== DATEN-IMPORT ABGESCHLOSSEN ===")

def fetch_sentiment():
    if not FINNHUB_API_KEY: print("FEHLER: FINNHUB_API_KEY nicht gefunden."); return
    print("=== STARTE SENTIMENT-ANALYSE (FINNHUB) ===")
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    analyzer = SentimentIntensityAnalyzer()
    with engine.connect() as conn:
        for asset in SYMBOLS:
            try:
                date_obj = (datetime.now() - timedelta(1)).date()
                print(f"Sammle Nachrichten für {asset}...")
                category = 'crypto' if 'BTC' in asset else 'general'
                news = finnhub_client.general_news(category, min_id=0)[:50]
                avg_score = 0.0
                if news:
                    scores = [analyzer.polarity_scores(article['headline'])['compound'] for article in news]
                    avg_score = sum(scores) / len(scores)
                print(f"-> Sentiment für {asset}: {avg_score:.4f}")
                stmt = insert(daily_sentiment).values(asset=asset, date=date_obj, sentiment_score=avg_score)
                stmt = stmt.on_conflict_do_update(index_elements=['asset', 'date'], set_={'sentiment_score': stmt.excluded.sentiment_score})
                conn.execute(stmt)
                conn.commit()
            except Exception as e: print(f"FEHLER bei Sentiment für {asset}: {e}")
    print("\n=== SENTIMENT-ANALYSE ABGESCHLOSSEN ===")

def train_all_models():
    print("=== STARTE LSTM MODELL-TRAINING FÜR ALLE STRATEGIEN ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten & Sentiment für {symbol}...")
            df_raw = load_data_with_sentiment(symbol, conn)
            if len(df_raw) < SEQUENCE_LENGTH + 50: continue
            for name, config in STRATEGIES.items():
                print(f"--- Trainiere '{name}' für {symbol} ---")
                try:
                    df_features = config['feature_func'](df_raw.copy())
                    X, y, scaler = prepare_data_for_lstm(df_features, config['features'])
                    if X is None: print("-> Nicht genügend Daten nach Bereinigung."); continue
                    
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
                    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
                    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=3)

                    model = Sequential([Input(shape=(X.shape[1], X.shape[2])), LSTM(units=50, return_sequences=True), Dropout(0.2), LSTM(units=50, return_sequences=False), Dropout(0.2), Dense(units=25), Dense(units=3, activation='softmax')])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    
                    print("Starte intelligentes Training...")
                    model.fit(X_train, y_train_cat, epochs=100, batch_size=32, validation_data=(X_val, y_val_cat), callbacks=[early_stopping], verbose=0)
                    
                    base_path_local = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model.save(f"{base_path_local}.keras")
                    joblib.dump(scaler, f"{base_path_local}_scaler.pkl")
                    
                    upload_to_gcs(f"{base_path_local}.keras", f"{os.path.basename(base_path_local)}.keras")
                    upload_to_gcs(f"{base_path_local}_scaler.pkl", f"{os.path.basename(base_path_local)}_scaler.pkl")
                except Exception as e: print(f"FEHLER: {e}")
    print("\n=== MODELL-TRAINING & UPLOAD ABGESCHLOSSEN ===")

def backtest_all_models():
    print("=== STARTE BACKTESTING ===")
    all_results = {'daily_lstm': [], 'genius_lstm': []}
    equity_curves = {'daily_lstm': {}, 'genius_lstm': {}}
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für Backtest von {symbol}...")
            df_full = load_data_with_sentiment(symbol, conn)
            if df_full.empty: continue
            for name, config in STRATEGIES.items():
                print(f"-- Starte Backtest für {name.upper()}...")
                try:
                    base_gcs_name = f"model_{name}_{symbol.replace('/', '')}"
                    base_local_path = f"{MODELS_DIR}/{base_gcs_name}"
                    download_from_gcs(f"{base_gcs_name}.keras", f"{base_local_path}.keras")
                    download_from_gcs(f"{base_gcs_name}_scaler.pkl", f"{base_local_path}_scaler.pkl")

                    model = load_model(f"{base_local_path}.keras")
                    scaler = joblib.load(f"{base_local_path}_scaler.pkl")
                    
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
    print("=== STARTE SIGNAL-GENERATOR ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            try:
                print(f"\n--- Verarbeite {symbol} ---")
                df_live = load_data_with_sentiment(symbol, conn).tail(120).copy()
                if len(df_live) < SEQUENCE_LENGTH: continue

                for name, config in STRATEGIES.items():
                    print(f"-> Generiere '{name}' Signal...")
                    base_gcs_name = f"model_{name}_{symbol.replace('/', '')}"
                    base_local_path = f"{MODELS_DIR}/{base_gcs_name}"
                    download_from_gcs(f"{base_gcs_name}.keras", f"{base_local_path}.keras")
                    download_from_gcs(f"{base_gcs_name}_scaler.pkl", f"{base_local_path}_scaler.pkl")
                    
                    model = load_model(f"{base_local_path}.keras")
                    scaler = joblib.load(f"{base_local_path}_scaler.pkl")

                    df_features = config['feature_func'](df_live).dropna()
                    last_sequence = df_features[config['features']].tail(SEQUENCE_LENGTH)
                    if len(last_sequence) < SEQUENCE_LENGTH: continue
                    
                    last_sequence_scaled = scaler.transform(last_sequence)
                    X_predict = np.array([last_sequence_scaled])
                    prediction_proba = model.predict(X_predict)[0]
                    confidence = round(np.max(prediction_proba) * 100, 2)
                    signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(np.argmax(prediction_proba))
                    price = df_features.iloc[-1]['close']
                    atr = df_features.iloc[-1].get('ATR', price * 0.02) # Fallback für ATR
                    
                    take_profit, stop_loss = (price + (2.5 * atr), price - (1.5 * atr)) if signal == "Kaufen" else (price - (2.5 * atr), price + (1.5 * atr)) if signal == "Verkaufen" else (None, None)

                    update_data = {'symbol': symbol, 'strategy': name, 'signal': signal, 'confidence': confidence, 'entry_price': price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'last_updated': datetime.now(timezone.utc)}
                    stmt = insert(predictions).values(update_data)
                    stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_=update_data)
                    conn.execute(stmt)
                    conn.commit()
                    print(f"✅ Signal für '{name}' gespeichert: {signal} ({confidence}%)")
            except Exception as e: print(f"FEHLER bei {symbol}: {e}")
    print("\n=== SIGNAL-GENERATOR ABGESCHLOSSEN ===")

# ==============================================================================
# 4. STEUERUNG
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master-Controller für die Finanz-App.")
    parser.add_argument("mode", choices=['fetch-data', 'fetch-sentiment', 'train', 'backtest', 'predict'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'fetch-data':
        fetch_data()
    elif args.mode == 'fetch-sentiment':
        fetch_sentiment()
    elif args.mode == 'train':
        train_all_models()
    elif args.mode == 'backtest':
        backtest_all_models()
    elif args.mode == 'predict':
        predict_all_signals()