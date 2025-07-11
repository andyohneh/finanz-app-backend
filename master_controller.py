# backend/master_controller.py (Finale Version ohne Warnungen)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
import requests
from sklearn.model_selection import train_test_split
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
SYMBOLS = ["BTC/USD", "XAU/USD"]
MODELS_DIR = "models"
INITIAL_CAPITAL = 100

STRATEGIES = {
    'daily': {
        'features': ['RSI', 'SMA_50', 'SMA_200', 'MACD_diff', 'ATR', 'Stoch'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_50=ta.trend.sma_indicator(df['close'], window=50),
            SMA_200=ta.trend.sma_indicator(df['close'], window=200),
            MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14),
            Stoch=ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        )
    },
    'swing': {
        'features': ['RSI', 'SMA_20', 'EMA_50', 'BB_Width', 'WilliamsR'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_20=ta.trend.sma_indicator(df['close'], window=20),
            EMA_50=ta.trend.ema_indicator(df['close'], window=50),
            BB_Width=ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        )
    },
    'genius': {
        'features': ['ADX', 'ATR', 'Stoch_RSI', 'WilliamsR', 'CCI'],
        'feature_func': lambda df: df.assign(
            ADX=ta.trend.adx(df['high'], df['low'], df['close'], window=14),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14),
            Stoch_RSI=ta.momentum.stochrsi(df['close'], window=14, smooth1=3, smooth2=3),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14),
            CCI=ta.trend.cci(df['high'], df['low'], df['close'], window=20)
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
    print("=== STARTE MODELL-TRAINING (PERFEKTIONIERT) ===")
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
                    features = config['features']
                    X = df[features]; y = df['target']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    scaler = StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Wir behalten die Feature-Namen beim Training bei
                    model = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbosity=-1).fit(X_train_scaled, y_train, feature_name=features)
                    
                    y_pred = model.predict(X_test_scaled)
                    accuracy = (y_pred == y_test).mean()
                    print(f"Modell-Genauigkeit: {accuracy:.2f}")

                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    joblib.dump(model, f"{base_path}_model.pkl"); joblib.dump(scaler, f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'w') as f: json.dump(features, f)
                    print(f"✅ Perfektioniertes LGBM-Modell gespeichert.")
                except Exception as e: print(f"FEHLER: {e}")
    print("\n=== MODELL-TRAINING ABGESCHLOSSEN ===")


def backtest_all_models():
    print("=== STARTE BACKTESTING (PERFEKTIONIERT) ===")
    all_results = {'daily': [], 'swing': [], 'genius': []}
    equity_curves = {'daily': {}, 'swing': {}, 'genius': {}}
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Daten für Backtest von {symbol}...")
            df_symbol = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp"), conn, params={'symbol': symbol})
            if df_symbol.empty: continue
            for name, config in STRATEGIES.items():
                print(f"-- Starte Backtest für {name.upper()}...")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model = joblib.load(f"{base_path}_model.pkl"); scaler = joblib.load(f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'r') as f: features = json.load(f)
                    
                    df_features = config['feature_func'](df_symbol.copy()).dropna()
                    X = df_features[features]
                    # KORREKTUR: Wir übergeben die reinen Werte, um die Warnung zu vermeiden
                    X_scaled = scaler.transform(X.values)
                    df_features['signal'] = model.predict(X_scaled)
                    
                    df_features['daily_return'] = df_features['close'].pct_change()
                    df_features['strategy_return'] = np.where(df_features['signal'].shift(1) == 1, df_features['daily_return'], np.where(df_features['signal'].shift(1) == 0, -df_features['daily_return'], 0))
                    
                    df_features['equity_curve'] = INITIAL_CAPITAL * (1 + df_features['strategy_return']).cumprod()
                    equity_curves[name][symbol] = {'dates': df_features['timestamp'].dt.strftime('%Y-%m-%d').tolist(),'values': df_features['equity_curve'].fillna(INITIAL_CAPITAL).round(2).tolist()}
                    
                    total_return_pct = df_features['strategy_return'].sum() * 100
                    trades = df_features[df_features['signal'] != 2]
                    win_rate = (len(trades[trades['strategy_return'] > 0]) / len(trades) * 100) if not trades.empty else 0
                    all_results[name].append({'Symbol': symbol, 'Gesamtrendite_%': round(total_return_pct, 2), 'Gewinnrate_%': round(win_rate, 2), 'Anzahl_Trades': len(trades)})
                    print(f"Ergebnis: {total_return_pct:.2f}% Rendite, {win_rate:.2f}% Gewinnrate")
                except Exception as e: print(f"FEHLER: {e}")
    with open('backtest_results.json', 'w') as f: json.dump(all_results, f, indent=4)
    with open('equity_curves.json', 'w') as f: json.dump(equity_curves, f, indent=4)
    print("\n✅ Backtest abgeschlossen.")

def predict_all_signals():
    #... (Diese Funktion bleibt unverändert)
    print("=== STARTE SIGNAL-GENERATOR ===")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\nLade Live-Daten für {symbol}...")
            df_live = pd.read_sql_query(text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400"), conn, params={'symbol': symbol})
            df_live = df_live.sort_values(by='timestamp').reset_index(drop=True)
            if len(df_live) < 250: continue
            for name, config in STRATEGIES.items():
                print(f"--- Generiere Signal: {name.upper()} für {symbol} ---")
                try:
                    base_path = f"{MODELS_DIR}/model_{name}_{symbol.replace('/', '')}"
                    model = joblib.load(f"{base_path}_model.pkl"); scaler = joblib.load(f"{base_path}_scaler.pkl")
                    with open(f"{base_path}_features.json", 'r') as f: features = json.load(f)
                    df_features = config['feature_func'](df_live.copy()).dropna()
                    X_predict = df_features[features].tail(1)
                    X_scaled = scaler.transform(X_predict.values) # .values zur Sicherheit
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
    parser.add_argument("mode", choices=['train', 'backtest', 'predict'], help="Der auszuführende Modus.")
    args = parser.parse_args()

    if args.mode == 'train':
        train_all_models()
    elif args.mode == 'backtest':
        train_all_models()
        backtest_all_models()
    elif args.mode == 'predict':
        predict_all_signals()