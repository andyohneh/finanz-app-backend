# ki_trainer.py (Version mit korrekter Speicherung für den Predictor)
import pandas as pd
import numpy as np
import ta
import joblib
import os
import subprocess
from sqlalchemy import text
from database import engine
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- NEUE KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']

# --- WICHTIG: Die folgenden Funktionen müssen mit predictor.py synchron sein! ---

def load_data_from_db(symbol: str, limit: int = 5000):
    """Lädt die neuesten Daten für ein Symbol aus der Datenbank."""
    print(f"Lade die letzten {limit} Datenpunkte für {symbol} aus der Datenbank...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol, 'limit': limit})
            if not df.empty:
                # Daten sind absteigend sortiert, für die Feature-Berechnung müssen sie aufsteigend sein
                df = df.iloc[::-1].reset_index(drop=True)
            print(f"Erfolgreich {len(df)} Datenpunkte geladen.")
            return df
    except Exception as e:
        print(f"Ein Fehler beim Laden der Daten ist aufgetreten: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt die technischen Indikatoren als Features hinzu."""
    print(f"Füge für {len(df)} Datenpunkte Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df.dropna(inplace=True)
    return df

# --- Die Triple-Barrier-Methode für das Labeling ---
def get_daily_volatility(close, lookback=100):
    """Berechnet die tägliche Volatilität als Standardabweichung der log-Returns."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=lookback).std()

def get_labels_triple_barrier(prices, tp_mult, sl_mult, max_period):
    """
    Erstellt Labels basierend auf der Triple-Barrier-Methode.
    1 = Kaufen (Take Profit getroffen)
   -1 = Verkaufen (Stop Loss getroffen)
    0 = Halten (Zeitlimit erreicht)
    """
    labels = pd.Series(np.nan, index=prices.index)
    volatility = get_daily_volatility(prices['close']) * 2 # Faktor 2 als Puffer
    
    for i in range(len(prices) - max_period):
        entry_price = prices['close'].iloc[i]
        
        # Dynamische Barrieren basierend auf Volatilität
        tp_barrier = entry_price * (1 + volatility.iloc[i] * tp_mult)
        sl_barrier = entry_price * (1 - volatility.iloc[i] * sl_mult)

        # Überprüfe die zukünftigen Preise
        for j in range(1, max_period + 1):
            future_price = prices['close'].iloc[i + j]
            
            if future_price >= tp_barrier:
                labels.iloc[i] = 1 # Kaufen
                break
            elif future_price <= sl_barrier:
                labels.iloc[i] = -1 # Verkaufen
                break
        
        # Wenn nach max_period keine Barriere getroffen wurde
        if pd.isna(labels.iloc[i]):
            labels.iloc[i] = 0 # Halten

    return labels

def add_labels(df, tp_mult=2, sl_mult=1.5, max_period=60):
    """Fügt die Labels zum DataFrame hinzu."""
    print("Starte Triple-Barrier-Labeling...")
    labels = get_labels_triple_barrier(df[['close']], tp_mult, sl_mult, max_period)
    df['label'] = labels
    # Wir löschen die letzten Zeilen, für die kein Label generiert werden konnte
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"Labeling abgeschlossen. Verteilung der Labels:\n{df['label'].value_counts(normalize=True)}")
    return df

# --- Modelltraining und Speichern ---
def train_and_save_model(df, symbol):
    """Trainiert das Modell und speichert es im korrekten Format und Verzeichnis."""
    feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
    X = df[feature_columns]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Modell-Performance auf Test-Daten:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # --- KORREKTUR: Verzeichnis und Dateiname an predictor.py anpassen ---
    # 1. Sicherstellen, dass das 'models' Verzeichnis existiert
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 2. Dateinamen anpassen (z.B. 'BTC/USD' -> 'BTCUSD')
    symbol_filename = symbol.replace('/', '')
    
    # 3. Korrekten Pfad und Dateinamen erstellen
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}.pkl')
    
    joblib.dump(model, model_path)
    print(f"Modell erfolgreich gespeichert unter: {model_path}")

def commit_and_push_models():
    """Fügt Modelle zum Git-Staging hinzu, committet und pusht sie."""
    # (Optional, falls du die Modelle auf GitHub sichern willst)
    print("Versuche, Modelle zu GitHub hochzuladen...")
    try:
        # --- KORREKTUR: Korrekten Pfad für 'git add' verwenden ---
        subprocess.run(['git', 'add', f'{MODEL_DIR}/*.pkl'], check=True)
        
        # Prüfen, ob es überhaupt Änderungen gibt
        status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status_result.stdout:
            print("Keine neuen Modell-Änderungen zum Committen gefunden.")
            return
            
        commit_message = f"KI-Modelle automatisch aktualisiert am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        
        # Git-Push (Stelle sicher, dass die Authentifizierung konfiguriert ist)
        subprocess.run(['git', 'push'], check=True)
        print("Modelle erfolgreich auf GitHub hochgeladen!")
    except Exception as e:
        print(f"Ein Fehler beim Git-Push ist aufgetreten: {e}")

def run_training_pipeline():
    """Führt den gesamten Trainingsprozess für alle Symbole aus."""
    for symbol in SYMBOLS:
        print(f"\n--- Starte Verarbeitung für {symbol} ---")
        df = load_data_from_db(symbol)
        if not df.empty and len(df) > 250: # Mindestanzahl für Features und Labeling
            df_features = add_features(df)
            df_labeled = add_labels(df_features, tp_mult=2, sl_mult=1.5, max_period=60)
            if not df_labeled.empty:
                train_and_save_model(df_labeled, symbol)
            else:
                print("Nicht genügend Daten nach dem Labeling übrig.")
        else:
            print(f"Nicht genügend Daten für {symbol} geladen, um das Training zu starten.")
    
    # Optional: Modelle nach dem Training automatisch hochladen
    # commit_and_push_models()

if __name__ == '__main__':
    run_training_pipeline()