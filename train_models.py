# backend/train_models.py (Finale Version mit Sentiment-Integration)
import pandas as pd
import numpy as np
import joblib
import os
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from database import engine

# --- KONFIGURATION ---
SYMBOLS = ["BTC/USD", "XAU/USD"]
STRATEGIES = {
    'daily': {
        'features': ['RSI', 'SMA_50', 'SMA_200', 'MACD_diff', 'sentiment_score'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_50=ta.trend.sma_indicator(df['close'], window=50),
            SMA_200=ta.trend.sma_indicator(df['close'], window=200),
            MACD_diff=ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
        )
    },
    'swing': {
        'features': ['RSI', 'SMA_20', 'EMA_50', 'BB_Width', 'sentiment_score'],
        'feature_func': lambda df: df.assign(
            RSI=ta.momentum.rsi(df['close'], window=14),
            SMA_20=ta.trend.sma_indicator(df['close'], window=20),
            EMA_50=ta.trend.ema_indicator(df['close'], window=50),
            BB_Width=ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
        )
    },
    'genius': {
        'features': ['ADX', 'ATR', 'Stoch_RSI', 'WilliamsR', 'sentiment_score'],
        'feature_func': lambda df: df.assign(
            ADX=ta.trend.adx(df['high'], df['low'], df['close'], window=14),
            ATR=ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14),
            Stoch_RSI=ta.momentum.stochrsi(df['close'], window=14, smooth1=3, smooth2=3),
            WilliamsR=ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        )
    }
}

def load_data_with_sentiment(symbol: str) -> pd.DataFrame:
    """
    Lädt historische Preisdaten UND die dazugehörigen Sentiment-Scores
    aus der Datenbank mit einem LEFT JOIN.
    """
    print(f"Lade Preis- und Sentiment-Daten für {symbol}...")
    with engine.connect() as conn:
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
        return df

def create_target(df, period=5):
    """Definiert das Ziel, das die KI vorhersagen soll."""
    df['future_return'] = df['close'].pct_change(period).shift(-period)
    conditions = [
        (df['future_return'] > 0.02),
        (df['future_return'] < -0.02),
    ]
    choices = [1, 0] # 1=Kaufen, 0=Verkaufen
    df['target'] = np.select(conditions, choices, default=2) # 2=Halten
    return df

def train_and_save_models():
    """Steuert den gesamten Trainingsprozess."""
    os.makedirs('models', exist_ok=True)
    for symbol in SYMBOLS:
        print(f"\n{'='*20}\nVerarbeite Trainingsdaten für {symbol}\n{'='*20}")
        df_raw = load_data_with_sentiment(symbol)
        
        if df_raw.empty or len(df_raw) < 250:
            print(f"Nicht genügend Daten für {symbol}, überspringe Training.")
            continue

        for name, config in STRATEGIES.items():
            print(f"\n--- Trainiere Modell: {name.upper()} für {symbol} ---")
            try:
                df = config['feature_func'](df_raw.copy())
                df = create_target(df)
                df.dropna(inplace=True)

                features = config['features']
                X = df[features]
                y = df['target']

                if len(X) < 100:
                    print("Nicht genügend Daten nach Feature Engineering.")
                    continue

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                scaler = StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X_train_scaled, y_train)
                
                model_path = f"models/model_{name}_{symbol.replace('/', '')}.pkl"
                joblib.dump({'model': model, 'scaler': scaler, 'features': features}, model_path)
                print(f"✅ Modell erfolgreich gespeichert: {model_path}")

            except Exception as e:
                print(f"Ein FEHLER ist beim Training aufgetreten: {e}")

if __name__ == "__main__":
    train_and_save_models()