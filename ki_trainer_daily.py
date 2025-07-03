import pandas as pd
import numpy as np
import ta
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
from dotenv import load_dotenv

# --- KONFIGURATION & DATENBANK ---
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("Keine DATABASE_URL in der .env-Datei gefunden!")
engine = create_engine(DATABASE_URL)

MODEL_PATH_BTC = 'models/model_daily_BTCUSD.pkl'
MODEL_PATH_XAU = 'models/model_daily_XAUUSD.pkl'

# --- DATENLADEN & FEATURE ENGINEERING ---

def load_data_from_db(symbol: str) -> pd.DataFrame:
    """
    Lädt historische Preisdaten UND die dazugehörigen Sentiment-Scores 
    aus der Datenbank.
    """
    print(f"Lade Preis- und Sentiment-Daten für {symbol} aus der Datenbank...")
    try:
        with engine.connect() as conn:
            # Wir nutzen ein LEFT JOIN, um Preisdaten mit Sentiment-Scores zu verbinden.
            # DATE(h.timestamp) stellt sicher, dass wir nur nach dem Datum abgleichen.
            query = text("""
                SELECT
                    h.timestamp, h.open, h.high, h.low, h.close, h.volume,
                    s.sentiment_score
                FROM historical_data_daily h
                LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = DATE(s.date)
                WHERE h.symbol = :symbol
                ORDER BY h.timestamp ASC
            """)
            df = pd.read_sql(query, conn, params={'symbol': symbol})
            
            # Fehlende Sentiment-Werte (für Tage ohne Nachrichten) mit neutralem Wert 0.0 füllen.
            df['sentiment_score'].fillna(0.0, inplace=True)
            
            print(f"Erfolgreich {len(df)} Datenpunkte mit Sentiment-Scores geladen.")
            return df
    except Exception as e:
        print(f"Ein Fehler beim Laden der kombinierten Daten ist aufgetreten: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt die 10 Kern-Indikatoren als Features hinzu."""
    print(f"Füge für {len(df)} Tages-Datenpunkte technische Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd_diff'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['bb_width'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2) - ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['roc'] = ta.momentum.roc(df['close'], window=12)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    
    # Der Sentiment-Score ist bereits geladen, wir müssen hier nichts hinzufügen.
    df.dropna(inplace=True)
    return df

def define_target_and_features(df: pd.DataFrame, future_candles: int = 5, profit_factor: float = 2.0, loss_factor: float = 1.0):
    """Definiert das Ziel (y) und die Features (X), inklusive Sentiment-Score."""
    df['future_high'] = df['high'].rolling(window=future_candles, min_periods=1).max().shift(-future_candles)
    df['future_low'] = df['low'].rolling(window=future_candles, min_periods=1).min().shift(-future_candles)
    
    atr = df['atr'].copy()
    take_profit_price = df['close'] + (atr * profit_factor)
    stop_loss_price = df['close'] - (atr * loss_factor)

    conditions = [
        (df['future_high'] >= take_profit_price),
        (df['future_low'] <= stop_loss_price)
    ]
    choices = [1, -1] # 1 für Kaufen (TP erreicht), -1 für Verkaufen (SL erreicht)
    df['target'] = np.select(conditions, choices, default=0) # 0 für Halten

    df.dropna(subset=['target', 'future_high', 'future_low'], inplace=True)
    
    # Wir fügen 'sentiment_score' zur Liste der Features hinzu!
    features = [
        'sma_fast', 'sma_slow', 'rsi', 'macd_diff', 'bb_width', 
        'stoch_k', 'roc', 'atr', 'adx', 'cci', 
        'sentiment_score'
    ]
    
    X = df[features]
    y = df['target']
    
    return X, y

# --- MODELLTRAINING & SPEICHERN ---

def train_and_save_model(symbol: str, model_path: str):
    """
    Hauptfunktion, die den gesamten Prozess für ein Asset steuert:
    Daten laden, Features hinzufügen, Modell trainieren und speichern.
    """
    # 1. Daten laden
    data = load_data_from_db(symbol)
    if data.empty:
        print(f"Keine Daten für {symbol} geladen. Training wird übersprungen.")
        return

    # 2. Features und Zielvariable erstellen
    featured_data = add_features(data.copy())
    X, y = define_target_and_features(featured_data.copy())

    if X.empty or y.empty:
        print(f"Nicht genügend Daten für {symbol}, um Features und Ziel zu erstellen. Training wird übersprungen.")
        return

    # 3. Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Trainingsdaten für {symbol}: {X_train.shape[0]} Zeilen")
    print(f"Testdaten für {symbol}: {X_test.shape[0]} Zeilen")
    
    # 4. Modell trainieren
    print(f"Starte das Training des RandomForest-Modells für {symbol}...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Modell-Training abgeschlossen.")

    # 5. Modell evaluieren
    print("\n--- Evaluationsergebnis ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)']))
    print("-------------------------\n")

    # 6. Modell speichern
    # Sicherstellen, dass der Ordner 'models' existiert
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modell für {symbol} erfolgreich unter '{model_path}' gespeichert.")

# --- HAUPTPROGRAMM ---

if __name__ == '__main__':
    print("=== Starte tägliches KI-Training für BTC/USD und XAU/USD ===")
    
    # Training für Bitcoin
    print("\n--- Training für BTC/USD ---")
    # GEÄNDERT: von 'BTCUSD' zu 'BTC/USD'
    train_and_save_model('BTC/USD', MODEL_PATH_BTC)
    
    # Training für Gold
    print("\n--- Training für XAU/USD ---")
    # GEÄNDERT: von 'XAUUSD' zu 'XAU/USD'
    train_and_save_model('XAU/USD', MODEL_PATH_XAU)
    
    print("\n=== Tägliches KI-Training abgeschlossen. ===")