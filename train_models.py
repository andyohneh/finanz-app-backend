# backend/train_models.py
import pandas as pd
import numpy as np
import joblib
import os
import ta
from ta.utils import dropna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sqlalchemy import text

from database import engine

# --- KONFIGURATION ---
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

def create_target(df, period=5):
    df['future_return'] = df['close'].pct_change(period).shift(-period)
    conditions = [
        (df['future_return'] > 0.02),
        (df['future_return'] < -0.02),
    ]
    choices = [1, 0] # 1 für Kaufen, 0 für Verkaufen
    df['target'] = pd.np.select(conditions, choices, default=2) # 2 für Halten
    return df

def train_and_save_models():
    """
    Lädt Daten, trainiert Modelle für jede Strategie und jedes Symbol und speichert sie.
    """
    # Sicherstellen, dass der 'models'-Ordner existiert
    os.makedirs('models', exist_ok=True)
    
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n{'='*20}\nLade Daten für {symbol}...\n{'='*20}")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp")
            df_raw = pd.read_sql_query(query, conn, params={'symbol': symbol})

            if df_raw.empty or len(df_raw) < 250:
                print(f"Nicht genügend Daten für {symbol}, überspringe Training.")
                continue

            for strategy_name, config in STRATEGIES.items():
                print(f"\n--- Trainiere Modell für Strategie: '{strategy_name.upper()}' ---")
                
                try:
                    # 1. Features & Target erstellen
                    df = config['feature_func'](df_raw.copy())
                    df = create_target(df)
                    df.dropna(inplace=True)

                    features = config['features']
                    X = df[features]
                    y = df['target']
                    
                    if len(X) < 100:
                        print("Nicht genügend Daten nach Feature Engineering.")
                        continue

                    # 2. Daten aufteilen und skalieren
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # 3. Modell trainieren
                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                    model.fit(X_train_scaled, y_train)

                    # 4. Genauigkeit prüfen
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"Modell-Genauigkeit (Accuracy): {accuracy:.2f}")

                    # 5. Modell UND Scaler zusammen speichern
                    symbol_filename = symbol.replace('/', '')
                    model_path = f"models/model_{strategy_name}_{symbol_filename}.pkl"
                    
                    joblib.dump({'model': model, 'scaler': scaler}, model_path)
                    print(f"✅ Modell erfolgreich gespeichert unter: {model_path}")

                except Exception as e:
                    print(f"Ein FEHLER ist beim Training aufgetreten: {e}")

if __name__ == "__main__":
    train_and_save_models()