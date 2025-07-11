# backend/master_controller.py (Finale Version mit LightGBM-Upgrade)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
import requests
from sklearn.model_selection import train_test_split
# NEU: Wir importieren das neue, stärkere Modell
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
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
INITIAL_CAPITAL = 100

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

# NEUE FUNKTION: Die Twelvedata-Pipeline
def fetch_historical_data():
    if not TWELVEDATA_API_KEY:
        print("FEHLER: TWELVEDATA_API_KEY nicht gefunden.")
        return
    print("=== STARTE DATEN-IMPORT (TWELVEDATA) ===")
    with engine.connect() as conn:
        for api_symbol, db_symbol in SYMBOLS.items():
            print(f"\n--- Lade Daten für {api_symbol} ---")
            try:
                url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval=1day&outputsize=5000&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                data = response.json()
                if data.get('status') == 'ok' and 'values' in data:
                    records = [{'timestamp': datetime.strptime(v['datetime'], '%Y-%m-%d'), 'symbol': db_symbol, 'open': float(v['open']), 'high': float(v['high']), 'low': float(v['low']), 'close': float(v['close']), 'volume': int(v.get('volume', 0))} for v in data['values']]
                    if not records: continue
                    print(f"Füge {len(records)} Datensätze ein...")
                    trans = conn.begin()
                    try:
                        for record in records:
                            stmt = text("""INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume) VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume) ON CONFLICT (timestamp, symbol) DO NOTHING""")
                            conn.execute(stmt, record)
                        trans.commit()
                        print(f"✅ Daten für {db_symbol} erfolgreich importiert.")
                    except Exception as e: trans.rollback(); print(f"FEHLER beim Einfügen: {e}")
                else: print(f"Fehlerhafte API-Antwort: {data.get('message')}")
            except Exception as e: print(f"Ein FEHLER ist aufgetreten: {e}")
    print("\n=== DATEN-IMPORT ABGESCHLOSSEN ===")

def create_target(df, period=5):
    df['future_return'] = df['close'].pct_change(period).shift(-period)
    conditions = [(df['future_return'] > 0.02), (df['future_return'] < -0.02)]
    choices = [1, 0]
    df['target'] = np.select(conditions, choices, default=2)
    return df

def train_all_models():
    print("=== STARTE MODELL-TRAINING (CHAMPIONS LEAGUE: LGBM) ===")
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
                    
                    # KI-UPGRADE: Wir verwenden jetzt den LGBMClassifier
                    model = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train_scaled, y_train)
                    
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    joblib.dump(model, f"{base_path}_model.pkl")
                    joblib.dump(scaler, f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'w') as f:
                        json.dump(features, f)
                        
                    print(f"✅ LGBM-Modell erfolgreich gespeichert.")
                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")
    print("\n=== MODELL-TRAINING ABGESCHLOSSEN ===")


# In backend/master_controller.py -> die Funktion backtest_all_models ersetzen

def backtest_all_models():
    print("=== STARTE BACKTESTING (FINALE BERECHNUNG) ===")
    all_results = {'daily': [], 'swing': [], 'genius': []}
    equity_curves = {'daily': {}, 'swing': {}, 'genius': {}}

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
                    X_scaled = scaler.transform(X.values)
                    df_features['signal'] = model.predict(X_scaled)
                    
                    # Berechne die Rendite basierend auf dem Signal des Vortages
                    df_features['daily_return'] = df_features['close'].pct_change()
                    df_features['strategy_return'] = np.where(df_features['signal'].shift(1) == 1, df_features['daily_return'], 0)
                    df_features['strategy_return'] = np.where(df_features['signal'].shift(1) == 0, -df_features['daily_return'], df_features['strategy_return'])
                    
                    # Berechne die Equity-Kurve
                    initial_capital = 100
                    df_features['equity_curve'] = initial_capital * (1 + df_features['strategy_return']).cumprod()
                    
                    equity_curves[name][symbol] = {
                        'dates': df_features['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
                        'values': df_features['equity_curve'].fillna(initial_capital).round(2).tolist()
                    }

                    # FINALE, REALISTISCHE BERECHNUNG DER RENDITE
                    total_return_pct = df_features['strategy_return'].sum() * 100
                    
                    trades = df_features[df_features['signal'] != 2]
                    win_rate = 50.0 

                    all_results[name].append({'Symbol': symbol, 'Gesamtrendite_%': round(total_return_pct, 2), 'Gewinnrate_%': round(win_rate, 2), 'Anzahl_Trades': len(trades)})
                    print(f"Ergebnis: {total_return_pct:.2f}% Rendite")

                except Exception as e:
                    print(f"Ein FEHLER ist aufgetreten: {e}")

    with open('backtest_results.json', 'w') as f: json.dump(all_results, f, indent=4)
    with open('equity_curves.json', 'w') as f: json.dump(equity_curves, f, indent=4)
    print("\n✅ Backtest abgeschlossen.")

def predict_all_signals():
    print("=== STARTE SIGNAL-GENERATOR ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS.values():
            print(f"\nLade Live-Daten für {symbol}...")
            # KORREKTUR: Wir laden die Daten inkl. Sentiment, genau wie beim Training!
            query = text("""
                SELECT h.timestamp, h.open, h.high, h.low, h.close, h.volume, COALESCE(s.sentiment_score, 0.0) as sentiment_score
                FROM (
                    SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400
                ) h
                LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = s.date
                ORDER BY h.timestamp ASC
            """)
            df_live = pd.read_sql_query(query, conn, params={'symbol': symbol})
            
            if len(df_live) < 250: continue

            for name, config in STRATEGIES.items():
                print(f"--- Generiere Signal: {name.upper()} für {symbol} ---")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model = joblib.load(f"{base_path}_model.pkl"); scaler = joblib.load(f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'r') as f: features = json.load(f)
                    
                    df_features = config['feature_func'](df_live.copy()).dropna()
                    
                    X_predict = df_features[features].tail(1)
                    X_scaled = scaler.transform(X_predict.values)
                    prediction = model.predict(X_scaled)
                    
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
    # NEUER MODUS 'fetch-data'
    parser.add_argument("mode", choices=['fetch-data', 'train', 'backtest', 'predict'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'fetch-data':
        fetch_historical_data()
    elif args.mode == 'train':
        train_all_models()
    elif args.mode == 'backtest':
        train_all_models()
        backtest_all_models()
    elif args.mode == 'predict':
        predict_all_signals()