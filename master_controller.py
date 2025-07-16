# backend/master_controller.py (Die finale, vollständige Champions-League-Version)
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

# Datenbank-Anbindung
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, predictions, daily_sentiment

# Sentiment Analyse
import finnhub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==============================================================================
# 1. ZENTRALE KONFIGURATION
# ==============================================================================
load_dotenv()
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
INITIAL_CAPITAL = 100
SEQUENCE_LENGTH = 60
RISK_PER_TRADE = 0.02 # 2% des Kapitals pro Trade

MINIMUM_TRADE_SIZES = {
    "BTC/USD": 0.001,
    "XAU/USD": 0.003
}

STRATEGIES = {
    'daily_lstm': {
        'features': ['close', 'SMA_50', 'RSI', 'sentiment_score'],
        'feature_func': lambda df: df.assign(
            SMA_50=ta.trend.sma_indicator(df['close'], window=50),
            RSI=ta.momentum.rsi(df['close'], window=14)
        )
    },
    'genius_lstm': {
        'features': ['close', 'RSI', 'MACD_diff', 'WilliamsR', 'ATR', 'sentiment_score'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        )
    }
}

# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================
def load_data_with_sentiment(symbol, conn):
    query = text("""
        SELECT h.timestamp, h.open, h.high, h.low, h.close, h.volume, COALESCE(s.sentiment_score, 0.0) as sentiment_score
        FROM historical_data_daily h LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = s.date
        WHERE h.symbol = :symbol ORDER BY h.timestamp ASC
    """)
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
                response = requests.get(url, timeout=20); response.raise_for_status(); data = response.json()
                if data.get('status') == 'ok' and 'values' in data:
                    records = [{'timestamp': datetime.strptime(v['datetime'], '%Y-%m-%d'), 'symbol': symbol, 'open': float(v['open']), 'high': float(v['high']), 'low': float(v['low']), 'close': float(v['close']), 'volume': int(v.get('volume', 0))} for v in data['values']]
                    if not records: continue
                    trans = conn.begin()
                    for record in records:
                        stmt = text("INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume) VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume) ON CONFLICT (timestamp, symbol) DO NOTHING")
                        conn.execute(stmt, record)
                    trans.commit(); print(f"✅ {len(records)} Datensätze für {symbol} importiert.")
                else: print(f"Fehlerhafte API-Antwort: {data.get('message')}")
            except Exception as e: print(f"Ein FEHLER bei {symbol}: {e}")
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
                avg_score = sum([analyzer.polarity_scores(article['headline'])['compound'] for article in news]) / len(news) if news else 0.0
                print(f"-> Sentiment für {asset}: {avg_score:.4f}")
                stmt = insert(daily_sentiment).values(asset=asset, date=date_obj, sentiment_score=avg_score)
                stmt = stmt.on_conflict_do_update(index_elements=['asset', 'date'], set_={'sentiment_score': stmt.excluded.sentiment_score})
                conn.execute(stmt); conn.commit()
            except Exception as e: print(f"FEHLER bei Sentiment für {asset}: {e}")
    print("\n=== SENTIMENT-ANALYSE ABGESCHLOSSEN ===")

def train_all_models():
    print("=== STARTE MODELL-TRAINING (LOKAL) ===")
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
                    if X is None: continue
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
                    y_train_cat, y_val_cat = tf.keras.utils.to_categorical(y_train, 3), tf.keras.utils.to_categorical(y_val, 3)
                    model = Sequential([Input(shape=(X.shape[1], X.shape[2])), LSTM(50, return_sequences=True), Dropout(0.2), LSTM(50), Dropout(0.2), Dense(25), Dense(3, activation='softmax')])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    model.fit(X_train, y_train_cat, epochs=100, batch_size=32, validation_data=(X_val, y_val_cat), callbacks=[early_stopping], verbose=1)
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model.save(f"{base_path}.keras"); joblib.dump(scaler, f"{base_path}_scaler.pkl")
                    print(f"✅ Modell für {name} erfolgreich gespeichert.")
                except Exception as e: print(f"FEHLER: {e}")
    print("\n=== LOKALES MODELL-TRAINING ABGESCHLOSSEN ===")

def backtest_all_models():
    print("=== STARTE BACKTESTING ===")
    all_results = {}
    equity_curves = {}
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für Backtest von {symbol}...")
            df_full = load_data_with_sentiment(symbol, conn)
            if df_full.empty: continue
            for name, config in STRATEGIES.items():
                if name not in all_results: all_results[name] = []
                if name not in equity_curves: equity_curves[name] = {}
                print(f"-- Starte Backtest für {name.upper()}...")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    if not os.path.exists(f"{base_path}.keras"): continue
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
    print("=== STARTE SIGNAL-GENERATOR (MIT BROKER-REGELN) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Verarbeite {symbol} ---")
            try:
                df_live = load_data_with_sentiment(symbol, conn).tail(120).copy()
                if len(df_live) < SEQUENCE_LENGTH: continue
                for name, config in STRATEGIES.items():
                    print(f"-> Generiere '{name}' Signal...")
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    if not os.path.exists(f"{base_path}.keras"): continue
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
                    atr_value = df_features.iloc[-1].get('ATR', price * 0.02)
                    take_profit, stop_loss, position_size = None, None, None
                    if signal in ["Kaufen", "Verkaufen"]:
                        stop_loss_distance = 1.5 * atr_value
                        risk_amount = INITIAL_CAPITAL * RISK_PER_TRADE
                        calculated_size = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
                        min_size = MINIMUM_TRADE_SIZES.get(symbol, 0.001)
                        position_size = max(calculated_size, min_size)
                        if signal == "Kaufen": take_profit, stop_loss = price + (2.5 * atr_value), price - stop_loss_distance
                        elif signal == "Verkaufen": take_profit, stop_loss = price - (2.5 * atr_value), price + stop_loss_distance
                    update_data = {'symbol': symbol, 'strategy': name, 'signal': signal, 'confidence': confidence, 'entry_price': price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'position_size': position_size, 'last_updated': datetime.now(timezone.utc)}
                    stmt = insert(predictions).values(update_data)
                    stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_=update_data)
                    conn.execute(stmt)
                    conn.commit()
                    print(f"✅ Signal gespeichert: {signal} | Empf. Größe: {position_size:.4f} Einheiten")
            except Exception as e: print(f"Ein FEHLER bei {symbol}: {e}")
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