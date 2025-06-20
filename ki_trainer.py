# ki_trainer.py (PLATIN-STANDARD-VERSION mit GridSearchCV)
import pandas as pd
import numpy as np
import ta
import joblib
import os
import subprocess
from sqlalchemy import text
from database import engine
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV # NEU: GridSearchCV importieren
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']

def load_data_from_db(symbol: str, limit: int = 4000):
    # Diese Funktion bleibt unverändert
    print(f"Lade die letzten {limit} Datenpunkte für {symbol} aus der Datenbank...")
    # ... (Code wie bisher) ...
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol, 'limit': limit})
            if not df.empty:
                df = df.iloc[::-1].reset_index(drop=True)
            print(f"Erfolgreich {len(df)} Datenpunkte geladen.")
            return df
    except Exception as e:
        print(f"Ein Fehler beim Laden der Daten ist aufgetreten: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Diese Funktion bleibt unverändert (unser bestes Feature-Set)
    print(f"Füge für {len(df)} Datenpunkte Features hinzu...")
    # ... (Code wie bisher) ...
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

def get_labels_triple_barrier(prices, tp_mult=2, sl_mult=1.5, max_period=60):
    # Die Triple-Barrier-Methode bleibt unverändert
    # ... (Code wie bisher) ...
    labels = pd.Series(np.nan, index=prices.index)
    log_returns = np.log(prices['close'] / prices['close'].shift(1))
    volatility = log_returns.rolling(window=100).std() * 2 
    
    for i in range(len(prices) - max_period):
        entry_price = prices['close'].iloc[i]
        tp_barrier = entry_price * (1 + volatility.iloc[i] * tp_mult)
        sl_barrier = entry_price * (1 - volatility.iloc[i] * sl_mult)
        for j in range(1, max_period + 1):
            future_price = prices['close'].iloc[i + j]
            if future_price >= tp_barrier:
                labels.iloc[i] = 1 # Kaufen
                break
            elif future_price <= sl_barrier:
                labels.iloc[i] = -1 # Verkaufen
                break
        if pd.isna(labels.iloc[i]):
            labels.iloc[i] = 0 # Halten
    return labels

def add_labels(df, tp_mult=2, sl_mult=1.5, max_period=60):
    # Diese Funktion bleibt unverändert
    print("Starte Triple-Barrier-Labeling...")
    # ... (Code wie bisher) ...
    labels = get_labels_triple_barrier(df[['close']], tp_mult, sl_mult, max_period)
    df['label'] = labels
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"Labeling abgeschlossen. Verteilung der Labels:\n{df['label'].value_counts(normalize=True)}")
    return df

def train_and_save_model(df, symbol):
    """
    Trainiert das Modell mit Hyperparameter-Tuning und speichert das BESTE Modell.
    """
    feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
    X = df[feature_columns]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- START DER PLATIN-LOGIK: Hyperparameter-Tuning ---
    
    # 1. Definiere das "Gitter" der zu testenden Parameter (unser "Rezeptbuch")
    param_grid = {
        'n_estimators': [100, 200],         # Anzahl der Bäume im Wald
        'max_depth': [10, 20, None],        # Maximale Tiefe eines Baumes
        'min_samples_leaf': [1, 2, 4],      # Mindestanzahl an Samples in einem Blatt
        'max_features': ['sqrt', 'log2']    # Anzahl der Features, die bei jedem Split berücksichtigt werden
    }

    # 2. Initialisiere das Basis-Modell
    model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # 3. Initialisiere die "Grid Search"
    # cv=3: Führt für jede Kombination eine 3-fache Kreuzvalidierung durch
    # n_jobs=-1: Nutzt alle verfügbaren CPU-Kerne, um den Prozess zu beschleunigen
    # verbose=2: Zeigt uns detailliert an, was gerade getestet wird
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    print("\n--- Starte Hyperparameter-Tuning (Grid Search)... Das kann dauern! ---")
    grid_search.fit(X_train, y_train)
    print("--- Hyperparameter-Tuning abgeschlossen! ---")

    # 4. Hole dir das beste Modell und die besten Parameter
    print(f"Beste gefundene Parameter: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # --- ENDE DER PLATIN-LOGIK ---

    print("\nModell-Performance des BESTEN Modells auf Test-Daten:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Speichere das optimierte Modell
    os.makedirs(MODEL_DIR, exist_ok=True)
    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}.pkl')
    joblib.dump(best_model, model_path)
    print(f"Bestes Modell erfolgreich gespeichert unter: {model_path}")

def run_training_pipeline():
    # Diese Funktion bleibt unverändert
    for symbol in SYMBOLS:
        print(f"\n--- Starte Verarbeitung für {symbol} ---")
        df = load_data_from_db(symbol)
        if not df.empty and len(df) > 250:
            df_features = add_features(df)
            df_labeled = add_labels(df_features)
            if not df_labeled.empty:
                train_and_save_model(df_labeled, symbol)
            else:
                print("Nicht genügend Daten nach dem Labeling übrig.")
        else:
            print(f"Nicht genügend Daten für {symbol} geladen, um das Training zu starten.")

if __name__ == '__main__':
    run_training_pipeline()