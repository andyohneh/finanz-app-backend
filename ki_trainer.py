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

def load_data_from_db(symbol: str, limit: int = 4000):
    # Diese Funktion bleibt unverändert
    print(f"Lade die letzten {limit} Datenpunkte für {symbol} aus der Datenbank...")
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

def add_features(df):
    # Diese Funktion bleibt unverändert (unser bestes Feature-Set)
    print(f"Füge für {len(df)} Datenpunkte Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df['bb_high_band'] = indicator_bb.bollinger_hband()
    df['bb_low_band'] = indicator_bb.bollinger_lband()
    df['bb_pband'] = indicator_bb.bollinger_pband()
    df['bb_wband'] = indicator_bb.bollinger_wband()
    
    df.dropna(inplace=True)
    return df

def add_labels(df, future_periods=240, percent_change_threshold=0.003):
    # Diese Funktion bleibt unverändert
    print(f"Füge Ziel-Labels hinzu (Horizont: {future_periods} min, Schwelle: {percent_change_threshold*100:.2f}%)...")
    future_close = df['close'].shift(-future_periods)
    df['price_change'] = (future_close - df['close']) / df['close']
    
    df['signal'] = 0
    df.loc[df['price_change'] > percent_change_threshold, 'signal'] = 1
    df.loc[df['price_change'] < -percent_change_threshold, 'signal'] = -1
    
    return df.dropna(subset=['price_change', 'signal'])

# ### KOMPLETT ÜBERARBEITETE FUNKTION ###
def train_and_save_model(df, symbol):
    """
    Findet die besten Hyperparameter mittels GridSearchCV, trainiert das finale Modell
    und speichert es.
    """
    print(f"Starte Hyperparameter-Tuning für {symbol}...")
    
    features = [
        'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr',
        'bb_high_band', 'bb_low_band', 'bb_pband', 'bb_wband'
    ]
    target = 'signal'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Definiere das "Gitter" der zu testenden Parameter
    param_grid = {
        'n_estimators': [100, 200],         # Anzahl der Bäume im Wald
        'max_depth': [10, 20, None],         # Maximale Tiefe eines Baumes
        'min_samples_leaf': [2, 4],          # Mindest-Datenpunkte in einem End-Blatt
        'max_features': ['sqrt', 'log2']     # Anzahl der Features pro Baum-Split
    }
    
    # 2. Erstelle das GridSearchCV-Objekt
    # n_jobs=-1 nutzt alle CPU-Kerne für schnelleres Training
    # verbose=2 zeigt uns den Fortschritt an
    # scoring='f1_macro' sagt ihm, dass der f1-score unser wichtigstes Gütekriterium ist
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
                               param_grid=param_grid,
                               cv=3, # 3-fache Kreuzvalidierung (weniger als 5 für schnellere Ausführung)
                               n_jobs=-1,
                               verbose=2,
                               scoring='f1_macro')
    
    # 3. Starte die Suche - DAS IST DER TEIL, DER LANGE DAUERT!
    print("Starte die Gittersuche (GridSearch)... Das kann einige Minuten dauern.")
    grid_search.fit(X_train, y_train)
    
    # 4. Hole das beste gefundene Modell und die besten Parameter
    print("\nGittersuche abgeschlossen!")
    print(f"Beste gefundene Parameter: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # 5. Gib den Leistungsbericht mit dem BESTEN Modell aus
    print(f"\nLeistungsbericht für {symbol} mit optimierten Parametern:")
    print(classification_report(y_test, best_model.predict(X_test), target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)']))
    
    # 6. Speichere das optimierte Modell
    safe_symbol_name = symbol.replace('/', '')
    model_filename = f'models/{safe_symbol_name}_model.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\nBestes Modell als '{model_filename}' gespeichert.")

# Der Rest der Datei (push_models, main, etc.) bleibt unverändert
def push_models_to_github():
    git_user = os.getenv('GIT_USER')
    git_pat = os.getenv('GIT_PAT')
    repo_name = os.getenv('GIT_REPO')
    if not all([git_user, git_pat, repo_name]):
        print("Git-Umgebungsvariablen nicht gesetzt. Überspringe Push.")
        return
    try:
        print("Versuche, neue Modelle auf GitHub hochzuladen...")
        subprocess.run(['git', 'config', '--global', 'user.email', 'action@github.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'GitHub Action'], check=True)
        subprocess.run(['git', 'add', 'models/'], check=True)
        status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status_result.stdout:
            print("Keine neuen Modell-Änderungen zum Committen gefunden.")
            return
        commit_message = f"KI-Modelle automatisch aktualisiert am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        remote_url = f"httpshttps://{git_user}:{git_pat}@github.com/{repo_name}.git"
        subprocess.run(['git', 'push', remote_url], check=True)
        print("Modelle erfolgreich auf GitHub hochgeladen!")
    except Exception as e:
        print(f"Ein Fehler beim Git-Push ist aufgetreten: {e}")

def process_symbol(symbol: str):
    print(f"\n--- Starte Verarbeitung für {symbol} ---")
    df = load_data_from_db(symbol)
    if not df.empty and len(df) > 250:
        df_features = add_features(df)
        df_final = add_labels(df_features, future_periods=240, percent_change_threshold=0.003)
        if not df_final.empty:
            train_and_save_model(df_final, symbol)
        else:
            print("Nicht genügend Daten nach dem Labeling übrig.")
    else:
        print(f"Nicht genügend Daten für {symbol}, um das Training zu starten.")

def main():
    SYMBOLS = ['BTC/USD', 'XAU/USD']
    for symbol in SYMBOLS:
        process_symbol(symbol)
    push_models_to_github()
    print("\n--- Nächtliches Training beendet. ---")

if __name__ == "__main__":
    main()