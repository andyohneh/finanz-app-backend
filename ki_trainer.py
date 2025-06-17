import pandas as pd
import numpy as np
import ta
import joblib
import os
import subprocess
from sqlalchemy import text
from database import engine
from datetime import datetime

def load_data_from_db(symbol: str, limit: int = 4000):
    """Lädt eine große Menge an Daten für das Training."""
    print(f"Lade die letzten {limit} Datenpunkte für {symbol} aus der Datenbank...")
    try:
        with engine.connect() as conn:
            query = text(f"SELECT * FROM historical_data WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT {limit}")
            df = pd.read_sql_query(query, conn)
            df = df.iloc[::-1].reset_index(drop=True)
            print(f"Erfolgreich {len(df)} Datenpunkte geladen.")
            return df
    except Exception as e:
        print(f"Ein Fehler beim Laden der Daten ist aufgetreten: {e}")
        return pd.DataFrame()

def add_features(df):
    """Fügt technische Indikatoren als neue Spalten zum DataFrame hinzu."""
    print(f"Füge für {len(df)} Datenpunkte Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df.dropna(inplace=True)
    return df

def add_labels(df, future_periods, percent_change_threshold):
    """Erstellt die Ziel-Labels basierend auf den übergebenen Strategie-Parametern."""
    print(f"Füge Ziel-Labels hinzu (Horizont: {future_periods} min, Schwelle: {percent_change_threshold:.2%})...")
    df['future_price'] = df['close'].shift(-future_periods)
    df['price_change_pct'] = (df['future_price'] - df['close']) / df['close']
    conditions = [
        df['price_change_pct'] > percent_change_threshold,
        df['price_change_pct'] < -percent_change_threshold,
    ]
    choices = [1, -1]
    df['target'] = np.select(conditions, choices, default=0)
    df.dropna(inplace=True)
    df.drop(columns=['future_price', 'price_change_pct'], inplace=True)
    df['target'] = df['target'].astype(int)
    return df

def train_and_save_model(df, symbol):
    """Trainiert ein Modell und speichert es als .joblib-Datei."""
    print(f"Starte Training für {symbol}...")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    features = ['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr']
    X = df[features]
    y = df['target']
    
    # Überprüfen, ob alle drei Klassen (1, 0, -1) vorhanden sind. Mindestens 2 müssen es sein.
    if len(y.unique()) < 2:
        print(f"FEHLER: Nicht genügend Signal-Varianten in den Daten für {symbol}, um ein Modell zu trainieren. Training wird übersprungen.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print(f"Training für {symbol} abgeschlossen.")

    # Evaluierung auf den Test-Daten
    y_pred = model.predict(X_test)
    print(f"\nLeistungsbericht für {symbol} auf Test-Daten:")
    print(classification_report(y_test, y_pred, target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)'], zero_division=0))

    model_filename = f"{symbol.lower()}_model.joblib"
    joblib.dump(model, model_filename)
    print(f"\nModell als '{model_filename}' gespeichert.")

def push_models_to_github():
    """Committet und pusht die Modelldateien nach GitHub."""
    print("Versuche, neue Modelle auf GitHub hochzuladen...")
    try:
        git_user = os.getenv('GIT_USER_NAME')
        git_email = os.getenv('GIT_USER_EMAIL')
        git_pat = os.getenv('GITHUB_PAT')
        repo_name = 'andyohneh/finanz-app-backend' # <-- BITTE ANPASSEN, FALLS NÖTIG

        if not all([git_user, git_email, git_pat]):
            print("Git-Umgebungsvariablen nicht gesetzt. Überspringe Push.")
            return

        subprocess.run(['git', 'config', '--global', 'user.name', git_user], check=True)
        subprocess.run(['git', 'config', '--global', 'user.email', git_email], check=True)
        subprocess.run(['git', 'add', '*.joblib'], check=True)

        status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if 'M btcusd_model.joblib' not in status_result.stdout and 'M xauusd_model.joblib' not in status_result.stdout:
            print("Keine neuen Modell-Änderungen zum Hochladen gefunden.")
            return

        commit_message = f"Auto-Update: KI-Modelle neu trainiert (Swing-Trading-Strategie) am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)

        remote_url = f"https://{git_user}:{git_pat}@github.com/{repo_name}.git"
        subprocess.run(['git', 'push', remote_url], check=True)

        print("Modelle erfolgreich auf GitHub hochgeladen!")
    except Exception as e:
        print(f"Ein Fehler beim Git-Push ist aufgetreten: {e}")

def process_symbol(symbol: str):
    print(f"\n--- Starte Verarbeitung für {symbol} ---")
    df = load_data_from_db(symbol)
    if not df.empty and len(df) > 250: # Brauchen genug Daten für 240min Lookahead
        df_features = add_features(df)
        
        # --- HIER IST DIE STRATEGIE-ÄNDERUNG ---
        # Wir schauen 240 Minuten (4 Std) in die Zukunft und verlangen eine Bewegung von 0.3%
        df_final = add_labels(df_features, future_periods=240, percent_change_threshold=0.003)
        # ----------------------------------------

        if not df_final.empty:
            train_and_save_model(df_final, symbol)
        else:
            print("Nicht genügend Daten nach dem Labeling übrig.")
    else:
        print(f"Nicht genügend Daten für {symbol}, um das Training zu starten.")

# --- Haupt-Logik ---
if __name__ == "__main__":
    symbols_to_process = ['BTCUSD', 'XAUUSD']
    for symbol in symbols_to_process:
        process_symbol(symbol)
    
    push_models_to_github()
    print("\n--- Nächtliches Training beendet. ---")