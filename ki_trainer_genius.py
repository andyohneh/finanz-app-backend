# ki_trainer_genius.py (FINALE Version ohne transformers)
import pandas as pd
import numpy as np
import ta
import joblib
import os
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from dotenv import load_dotenv

# --- KONFIGURATION & DATENBANK ---
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

MODEL_PATH_BTC = 'models/model_genius_BTCUSD.pkl'
MODEL_PATH_XAU = 'models/model_genius_XAUUSD.pkl'

# --- DATENLADEN & FEATURE ENGINEERING (MIT SENTIMENT AUS DB) ---

def load_data_from_db(symbol: str) -> pd.DataFrame:
    """Lädt historische Preisdaten UND die dazugehörigen Sentiment-Scores aus der Datenbank."""
    print(f"Lade Preis- und Sentiment-Daten für {symbol} aus der Datenbank...")
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT h.timestamp, h.open, h.high, h.low, h.close, h.volume, s.sentiment_score
                FROM historical_data_daily h
                LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = DATE(s.date)
                WHERE h.symbol = :symbol ORDER BY h.timestamp ASC
            """)
            df = pd.read_sql(query, conn, params={'symbol': symbol})
            df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
            print(f"Erfolgreich {len(df)} Datenpunkte mit Sentiment-Scores geladen.")
            return df
    except Exception as e:
        print(f"Ein Fehler beim Laden der kombinierten Daten ist aufgetreten: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt technische Indikatoren für die Genius-Strategie hinzu."""
    print("Füge technische Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=150)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd_diff'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df.dropna(inplace=True)
    return df

def define_target_and_features(df: pd.DataFrame, future_candles: int = 7, profit_factor: float = 2.5, loss_factor: float = 1.2):
    """Definiert das Ziel und die Features, inklusive Sentiment."""
    df['future_high'] = df['high'].rolling(window=future_candles, min_periods=1).max().shift(-future_candles)
    df['future_low'] = df['low'].rolling(window=future_candles, min_periods=1).min().shift(-future_candles)
    
    atr = df['atr'].copy()
    take_profit_price = df['close'] + (atr * profit_factor)
    stop_loss_price = df['close'] - (atr * loss_factor)

    conditions = [ (df['future_high'] >= take_profit_price), (df['future_low'] <= stop_loss_price) ]
    choices = [1, -1]
    df['target'] = np.select(conditions, choices, default=0)

    df.dropna(subset=['target', 'future_high', 'future_low'], inplace=True)
    
    features = [ 'sma_fast', 'sma_slow', 'rsi', 'macd_diff', 'atr', 'sentiment_score' ]
    
    X = df[features]
    y = df['target']
    return X, y

# --- MODELLTRAINING & SPEICHERN ---

def train_and_save_model(symbol: str, model_path: str):
    """Steuert den gesamten Trainingsprozess für die Genius-Strategie."""
    data = load_data_from_db(symbol)
    if data.empty: return

    featured_data = add_features(data.copy())
    X, y = define_target_and_features(featured_data.copy())

    if X.empty or y.empty:
        print(f"Nicht genügend Daten für {symbol}. Training wird übersprungen.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Starte das Training des Genius-Modells für {symbol}...")
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=10)
    model.fit(X_train, y_train)
    print("Modell-Training abgeschlossen.")

    print("\n--- Evaluationsergebnis ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)']))
    print("-------------------------\n")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modell für {symbol} erfolgreich unter '{model_path}' gespeichert.")

# --- HAUPTPROGRAMM ---

if __name__ == '__main__':
    print("=== Starte KI-Training für Genius-Strategie ===")
    train_and_save_model('BTC/USD', MODEL_PATH_BTC)
    train_and_save_model('XAU/USD', MODEL_PATH_XAU)
    print("\n=== KI-Training für Genius-Strategie abgeschlossen. ===")