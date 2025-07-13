# backend/master_controller.py (Finale Version mit Sentiment im Backtester)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

# Machine Learning Bibliotheken
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
SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
SEQUENCE_LENGTH = 60

LSTM_STRATEGY = {
    'name': 'genius_lstm',
    'features': ['close', 'RSI', 'MACD_diff', 'WilliamsR', 'ATR', 'sentiment_score'],
    'feature_func': lambda df: df.assign(
        RSI=ta.momentum.rsi(df['close'], window=14),
        MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9),
        WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14),
        ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    )
}

# ==============================================================================
# 2. FUNKTIONEN
# ==============================================================================

def load_data_with_sentiment(symbol, conn):
    """
    Lädt historische Preisdaten UND die dazugehörigen Sentiment-Scores
    und füllt fehlende Werte intelligent auf.
    """
    query = text("""
        SELECT
            h.timestamp, h.open, h.high, h.low, h.close, h.volume,
            COALESCE(s.sentiment_score, 0.0) as sentiment_score
        FROM historical_data_daily h
        LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = s.date
        WHERE h.symbol = :symbol
        ORDER BY h.timestamp ASC
    """)
    df = pd.read_sql_query(query, conn, params={'symbol': symbol})
    df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill').fillna(0.0)
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

def train_lstm_model():
    print("=== STARTE LSTM MODELL-TRAINING ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für {symbol}...")
            df_raw = load_data_with_sentiment(symbol, conn)
            if len(df_raw) < SEQUENCE_LENGTH + 50: continue
            print(f"--- Trainiere LSTM für {symbol} ---")
            try:
                df_features = LSTM_STRATEGY['feature_func'](df_raw.copy())
                X, y, scaler = prepare_data_for_lstm(df_features, LSTM_STRATEGY['features'])
                if X is None: print("Nicht genügend Daten nach Bereinigung."); continue
                y_categorical = tf.keras.utils.to_categorical(y, num_classes=3)
                model = Sequential([
                    Input(shape=(X.shape[1], X.shape[2])),
                    LSTM(units=50, return_sequences=True), Dropout(0.2),
                    LSTM(units=50, return_sequences=False), Dropout(0.2),
                    Dense(units=25), Dense(units=3, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(X, y_categorical, epochs=25, batch_size=32, verbose=0)
                print(f"Training für {symbol} abgeschlossen.")
                base_path = f"{MODELS_DIR}/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                model.save(f"{base_path}.keras")
                joblib.dump(scaler, f"{base_path}_scaler.pkl")
                print(f"✅ LSTM-Modell für {symbol} gespeichert.")
            except Exception as e: print(f"FEHLER: {e}")
    print("\n=== MODELL-TRAINING ABGESCHLOSSEN ===")


def backtest_lstm_model():
    print("=== STARTE LSTM BACKTEST (DETAILLIERTE ANALYSE) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für Backtest von {symbol}...")
            # HIER IST DIE FINALE KORREKTUR: Wir nutzen die richtige Ladefunktion
            df_full = load_data_with_sentiment(symbol, conn)
            if df_full.empty: continue
            
            print(f"--- Backteste LSTM für {symbol} ---")
            try:
                base_path = f"{MODELS_DIR}/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                model = load_model(f"{base_path}.keras")
                scaler = joblib.load(f"{base_path}_scaler.pkl")
                
                df_features = LSTM_STRATEGY['feature_func'](df_full.copy()).dropna()
                
                all_scaled_data = scaler.transform(df_features[LSTM_STRATEGY['features']])
                X_backtest = []
                for i in range(SEQUENCE_LENGTH, len(all_scaled_data)):
                    X_backtest.append(all_scaled_data[i-SEQUENCE_LENGTH:i])
                
                if not X_backtest:
                    print("Keine Daten für Backtest-Vorhersage vorhanden.")
                    continue
                    
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
    print("=== STARTE LSTM SIGNAL-GENERATOR (MIT ALLEM!) ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Live-Daten & Sentiment für {symbol}...")
            df_live = load_data_with_sentiment(symbol, conn)
            if len(df_live) < SEQUENCE_LENGTH: continue
            print(f"--- Generiere LSTM-Signal für {symbol} ---")
            try:
                base_path = f"{MODELS_DIR}/model_{LSTM_STRATEGY['name']}_{symbol.replace('/', '')}"
                model = load_model(f"{base_path}.keras")
                scaler = joblib.load(f"{base_path}_scaler.pkl")
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
                take_profit, stop_loss = None, None
                if signal == "Kaufen":
                    take_profit = price + (2.5 * atr_value)
                    stop_loss = price - (1.5 * atr_value)
                elif signal == "Verkaufen":
                    take_profit = price - (2.5 * atr_value)
                    stop_loss = price + (1.5 * atr_value)
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
    # Wir fügen den neuen Backtest-Modus hinzu
    parser.add_argument("mode", choices=['train-lstm', 'predict-lstm', 'backtest-lstm'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'train-lstm':
        train_lstm_model()
    elif args.mode == 'predict-lstm':
        predict_with_lstm()
    elif args.mode == 'backtest-lstm':
        backtest_lstm_model()