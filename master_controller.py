# backend/master_controller.py (Die finale, vollständige und unzerstörbare Version)
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
from collections import Counter

# Machine Learning Bibliotheken
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Dropout
from lightgbm import LGBMClassifier

# Datenbank-Anbindung & weitere
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, predictions, daily_sentiment
import finnhub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

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
RISK_PER_TRADE = 0.02

MINIMUM_TRADE_SIZES = {
    "BTC/USD": 0.001,
    "XAU/USD": 0.003
}

STRATEGIES = {
    'daily_lstm': { # Dein neuer, sauberer Name
        'models': ['lstm', 'lgbm', 'rf'],
        'features_lstm': ['close', 'SMA_50', 'RSI', 'sentiment_score', 'vix'],
        'features_tree': ['SMA_50', 'RSI', 'sentiment_score', 'vix'],
        'feature_func': lambda df: df.assign(
            SMA_50=ta.trend.sma_indicator(df['close'], window=50),
            RSI=ta.momentum.rsi(df['close'], window=14))
    },
    'genius_lstm': { # Dein neuer, sauberer Name
        'models': ['lstm', 'lgbm', 'rf'],
        'features_lstm': ['close', 'RSI', 'MACD_diff', 'WilliamsR', 'ATR', 'sentiment_score', 'vix'],
        'features_tree': ['RSI', 'MACD_diff', 'WilliamsR', 'ATR', 'sentiment_score', 'vix'],
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
def load_data_with_sentiment(symbol, conn):
    query = text("SELECT h.timestamp, h.open, h.high, h.low, h.close, h.volume, h.vix, COALESCE(s.sentiment_score, 0.0) as sentiment_score FROM historical_data_daily h LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = s.date WHERE h.symbol = :symbol ORDER BY h.timestamp ASC")
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

def create_target_for_tree_models(df, period=5):
    df['future_return'] = df['close'].pct_change(period).shift(-period)
    conditions = [(df['future_return'] > 0.02), (df['future_return'] < -0.02)]
    choices = [1, 0]
    df['target'] = np.select(conditions, choices, default=2)
    return df

# ==============================================================================
# 3. KERNFUNKTIONEN
# ==============================================================================
def fetch_data():
    if not TWELVEDATA_API_KEY: print("FEHLER: TWELVEDATA_API_KEY nicht gefunden."); return
    print("=== STARTE DATEN-IMPORT (TWELVEDATA + VIX) ===")
    print("\n--- Lade VIX (Angst-Index) Daten von yfinance ---")
    try:
        vix_df = yf.download('^VIX', period="max", interval="1d")
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
        vix_df.reset_index(inplace=True)
        vix_df.columns = [col.lower() for col in vix_df.columns]
        vix_df = vix_df[['date', 'close']]
        vix_df.rename(columns={'date': 'timestamp', 'close': 'vix'}, inplace=True)
        vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp']).dt.tz_localize(None)
        print(f"✅ {len(vix_df)} VIX-Datenpunkte geladen.")
    except Exception as e: print(f"FEHLER beim Laden der VIX-Daten: {e}"); return

    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Lade Daten für {symbol} von Twelvedata ---")
            try:
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=30); response.raise_for_status(); data = response.json()
                if data.get('status') == 'ok' and 'values' in data:
                    df_asset = pd.DataFrame(data['values']); df_asset['timestamp'] = pd.to_datetime(df_asset['datetime'])
                    df_merged = pd.merge(df_asset, vix_df, on='timestamp', how='left')
                    df_merged['vix'] = df_merged['vix'].ffill().bfill()
                    df_merged.dropna(subset=['vix'], inplace=True)
                    records = [{'timestamp': r['timestamp'], 'symbol': symbol, 'open': float(r['open']), 'high': float(r['high']), 'low': float(r['low']), 'close': float(r['close']), 'volume': int(r.get('volume', 0)), 'vix': float(r['vix'])} for _, r in df_merged.iterrows()]
                    if not records: continue
                    print(f"Füge {len(records)} Datensätze für {symbol} inkl. VIX in die DB ein...")
                    trans = conn.begin()
                    try:
                        for record in records:
                            stmt = text("INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume, vix) VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume, :vix) ON CONFLICT (timestamp, symbol) DO UPDATE SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume, vix = EXCLUDED.vix")
                            conn.execute(stmt, record)
                        trans.commit(); print(f"✅ Daten für {symbol} erfolgreich importiert.")
                    except Exception as e_insert: trans.rollback(); print(f"FEHLER beim Einfügen: {e_insert}")
                else: print(f"Fehlerhafte API-Antwort von Twelvedata: {data.get('message')}")
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
                avg_score = sum([analyzer.polarity_scores(article['headline'])['compound'] for article in news]) / len(news) if news else 0.0
                print(f"-> Sentiment für {asset}: {avg_score:.4f}")
                stmt = insert(daily_sentiment).values(asset=asset, date=date_obj, sentiment_score=avg_score)
                stmt = stmt.on_conflict_do_update(index_elements=['asset', 'date'], set_={'sentiment_score': stmt.excluded.sentiment_score})
                conn.execute(stmt); conn.commit()
            except Exception as e: print(f"FEHLER bei Sentiment für {asset}: {e}")
    print("\n=== SENTIMENT-ANALYSE ABGESCHLOSSEN ===")

def train_all_models():
    print("=== STARTE ENSEMBLE MODELL-TRAINING (3 TITANEN) ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für {symbol}...")
            df_raw = load_data_with_sentiment(symbol, conn)
            if len(df_raw) < 250: continue

            for name, config in STRATEGIES.items():
                print(f"--- Trainiere Team '{name.upper()}' für {symbol} ---")
                df_features = config['feature_func'](df_raw.copy())

                for model_type in config['models']:
                    try:
                        print(f"-> Trainiere {model_type.upper()}-Modell...")
                        base_path = f"{MODELS_DIR}/model_{name}_{model_type}_{symbol.replace('/', '')}"

                        if model_type == 'lstm':
                            X, y, scaler = prepare_data_for_lstm(df_features, config['features_lstm'])
                            if X is None: print("-> Nicht genug Daten für LSTM."); continue
                            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                            y_train_cat, y_val_cat = tf.keras.utils.to_categorical(y_train, 3), tf.keras.utils.to_categorical(y_val, 3)
                            model = Sequential([Input(shape=(X.shape[1], X.shape[2])), LSTM(50), Dropout(0.2), Dense(25), Dense(3, activation='softmax')])
                            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            model.fit(X_train, y_train_cat, epochs=100, validation_data=(X_val, y_val_cat), callbacks=[early_stopping], verbose=1)
                            model.save(f"{base_path}.keras")
                            joblib.dump(scaler, f"{base_path}_scaler.pkl")

                        elif model_type in ['lgbm', 'rf']:
                            df_tree = create_target_for_tree_models(df_features.copy())
                            df_tree.dropna(inplace=True)
                            X = df_tree[config['features_tree']]; y = df_tree['target']
                            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                            scaler = StandardScaler().fit(X_train)
                            X_train_scaled = scaler.transform(X_train)
                            
                            if model_type == 'lgbm':
                                model = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train_scaled, y_train)
                            else: # rf
                                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train_scaled, y_train)
                            
                            joblib.dump(model, f"{base_path}.pkl")
                            joblib.dump(scaler, f"{base_path}_scaler.pkl")
                        
                        print(f"✅ {model_type.upper()}-Modell für '{name}' erfolgreich gespeichert.")
                    except Exception as e:
                        print(f"FEHLER beim Training des {model_type.upper()}-Modells: {e}")
    print("\n=== ENSEMBLE MODELL-TRAINING ABGESCHLOSSEN ===")


def backtest_all_models():
    print("=== STARTE ENSEMBLE BACKTESTING (3 TITANEN) ===")
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
                    df_features = config['feature_func'](df_full.copy()).dropna()
                    team_predictions = pd.DataFrame(index=df_features.index)

                    for model_type in config['models']:
                        base_path = f"{MODELS_DIR}/model_{name}_{model_type}_{symbol.replace('/', '')}"
                        if not os.path.exists(f"{base_path}{'.keras' if model_type == 'lstm' else '.pkl'}"): continue

                        if model_type == 'lstm':
                            model = load_model(f"{base_path}.keras")
                            scaler = joblib.load(f"{base_path}_scaler.pkl")
                            all_scaled_data = scaler.transform(df_features[config['features_lstm']])
                            X_backtest = []
                            for i in range(SEQUENCE_LENGTH, len(all_scaled_data)):
                                X_backtest.append(all_scaled_data[i-SEQUENCE_LENGTH:i])
                            if not X_backtest: continue
                            predicted_classes = np.argmax(model.predict(np.array(X_backtest), verbose=0), axis=1)
                            team_predictions[f'signal_{model_type}'] = pd.Series(predicted_classes, index=df_features.index[SEQUENCE_LENGTH:])
                        
                        elif model_type in ['lgbm', 'rf']:
                            model = joblib.load(f"{base_path}.pkl")
                            scaler = joblib.load(f"{base_path}_scaler.pkl")
                            X_backtest = df_features[config['features_tree']]
                            X_scaled = scaler.transform(X_backtest.values) # .values für saubere Logs
                            predicted_classes = model.predict(X_scaled)
                            team_predictions[f'signal_{model_type}'] = predicted_classes

                    def get_final_signal(row):
                        votes = [int(v) for v in row.dropna()]
                        if not votes: return 2
                        vote_counts = Counter(votes)
                        most_common_vote, count = vote_counts.most_common(1)[0]
                        return most_common_vote if count >= 2 else 2

                    df_trade = df_features.copy()
                    df_trade['signal'] = team_predictions.apply(get_final_signal, axis=1)
                    
                    df_trade['daily_return'] = df_trade['close'].pct_change()
                    df_trade['strategy_return'] = np.where(df_trade['signal'].shift(1) == 1, df_trade['daily_return'], 0)
                    df_trade['strategy_return'] = np.where(df_trade['signal'].shift(1) == 0, -df_trade['daily_return'], df_trade['strategy_return'])
                    
                    total_return_pct = df_trade['strategy_return'].sum() * 100
                    df_trade['equity_curve'] = INITIAL_CAPITAL * (1 + df_trade['strategy_return'].cumsum())
                    equity_curves[name][symbol] = {'dates': df_trade['timestamp'].dt.strftime('%Y-%m-%d').tolist(),'values': df_trade['equity_curve'].fillna(INITIAL_CAPITAL).round(2).tolist()}
                    
                    trades = df_trade[df_trade['signal'] != 2]
                    win_rate = (len(trades[trades['strategy_return'] > 0]) / len(trades) * 100) if not trades.empty else 0
                    all_results[name].append({'Symbol': symbol, 'Gesamtrendite_%': round(total_return_pct, 2), 'Gewinnrate_%': round(win_rate, 2), 'Anzahl_Trades': len(trades)})
                    print(f"Ergebnis: {total_return_pct:.2f}% Rendite")
                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")

    with open('backtest_results.json', 'w') as f: json.dump(all_results, f, indent=4)
    with open('equity_curves.json', 'w') as f: json.dump(equity_curves, f, indent=4)
    print("\n✅ Backtest abgeschlossen.")

def predict_all_signals():
    print("=== STARTE ENSEMBLE SIGNAL-GENERATOR (3 TITANEN) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Verarbeite {symbol} ---")
            df_live = load_data_with_sentiment(symbol, conn).tail(200).copy()
            if len(df_live) < SEQUENCE_LENGTH: continue

            for name, config in STRATEGIES.items():
                print(f"-> Generiere '{name}' Signal mit Team-Abstimmung...")
                try:
                    df_features = config['feature_func'](df_live.copy()).dropna()
                    predictions_list = []
                    
                    for model_type in config['models']:
                        base_path = f"{MODELS_DIR}/model_{name}_{model_type}_{symbol.replace('/', '')}"
                        if not os.path.exists(f"{base_path}{'.keras' if model_type == 'lstm' else '.pkl'}"): continue
                        
                        if model_type == 'lstm':
                            model = load_model(f"{base_path}.keras")
                            scaler = joblib.load(f"{base_path}_scaler.pkl")
                            last_sequence = df_features[config['features_lstm']].tail(SEQUENCE_LENGTH)
                            if len(last_sequence) < SEQUENCE_LENGTH: continue
                            last_sequence_scaled = scaler.transform(last_sequence)
                            prediction = np.argmax(model.predict(np.array([last_sequence_scaled]), verbose=0)[0])
                            predictions_list.append(prediction)

                        elif model_type in ['lgbm', 'rf']:
                            model = joblib.load(f"{base_path}.pkl")
                            scaler = joblib.load(f"{base_path}_scaler.pkl")
                            X_predict = df_features[config['features_tree']].tail(1)
                            X_scaled = scaler.transform(X_predict.values)
                            prediction = model.predict(X_scaled)[0]
                            predictions_list.append(int(prediction))
                    
                    if not predictions_list: print("-> Keine Modelle konnten eine Vorhersage treffen."); continue

                    vote_counts = Counter(predictions_list)
                    final_signal_code, count = vote_counts.most_common(1)[0]
                    
                    if count < 2: final_signal_code = 2

                    signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(final_signal_code)
                    price = float(df_features.iloc[-1]['close'])
                    
                    update_data = {'symbol': symbol, 'strategy': name, 'signal': signal, 'entry_price': price, 'last_updated': datetime.now(timezone.utc)}
                    stmt = insert(predictions).values(update_data)
                    stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'strategy'], set_={k:v for k,v in update_data.items() if k != 'id'})
                    conn.execute(stmt); conn.commit()
                    
                    print(f"✅ Team-Signal für '{name}' gespeichert: {signal} (Stimmen: {dict(vote_counts)})")
                except Exception as e:
                    print(f"FEHLER bei der Vorhersage für {name}: {e}")
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