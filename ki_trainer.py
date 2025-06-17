import pandas as pd
import numpy as np
import ta
import joblib
import os
import subprocess
from sqlalchemy import text
from database import engine
from datetime import datetime # <-- Die fehlende Zeile

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

def add_labels(df, future_periods=15, percent_change_threshold=0.0005):
    """Erstellt die Ziel-Labels (Kaufen/Verkaufen/Halten)."""
    print(f"Füge für {len(df)} Datenpunkte Labels hinzu...")
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

    features = ['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    model.fit(X_train, y_train)
    print(f"Training für {symbol} abgeschlossen.")
    
    model_filename = f"{symbol.lower()}_model.joblib"
    joblib.dump(model, model_filename)
    print(f"Modell als '{model_filename}' gespeichert.")

def push_models_to_github():
    """Committet und pusht die Modelldateien nach GitHub."""
    print("Versuche, neue Modelle auf GitHub hochzuladen...")
    try:
        git_user = os.getenv('GIT_USER_NAME')
        git_email = os.getenv('GIT_USER_EMAIL')
        git_pat = os.getenv('GITHUB_PAT')
        
        # WICHTIG: Ersetze dies mit deinem exakten GitHub-Benutzernamen und Repository-Namen
        repo_name = 'andyohneh/finanz-app-backend' 

        if not all([git_user, git_email, git_pat]):
            print("Git-Umgebungsvariablen (GIT_USER_NAME, GIT_USER_EMAIL, GITHUB_PAT) sind nicht gesetzt. Überspringe Push.")
            return

        # Git im aktuellen Verzeichnis für den Roboter konfigurieren
        subprocess.run(['git', 'config', '--global', 'user.name', git_user], check=True)
        subprocess.run(['git', 'config', '--global', 'user.email', git_email], check=True)

        # Die Modelldateien zum Commit hinzufügen
        subprocess.run(['git', 'add', '*.joblib'], check=True)

        # Überprüfen, ob es überhaupt Änderungen gibt, die committed werden können
        status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status_result.stdout:
            print("Keine neuen Modell-Änderungen zum Hochladen gefunden.")
            return

        # Commit erstellen
        commit_message = f"Auto-Update: KI-Modelle neu trainiert am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)

        # Push durchführen mit dem Personal Access Token zur Authentifizierung
        remote_url = f"https://{git_user}:{git_pat}@github.com/{repo_name}.git"
        subprocess.run(['git', 'push', remote_url], check=True)

        print("Modelle erfolgreich auf GitHub hochgeladen!")

    except subprocess.CalledProcessError as e:
        print(f"Ein Git-Fehler ist aufgetreten: {e}. Output: {e.stdout}. Stderr: {e.stderr}")
    except Exception as e:
        print(f"Ein allgemeiner Fehler beim Git-Push ist aufgetreten: {e}")

# --- Haupt-Logik ---
if __name__ == "__main__":
    symbols_to_process = ['BTCUSD', 'XAUUSD']
    for symbol in symbols_to_process:
        print(f"\n--- Starte Verarbeitung für {symbol} ---")
        df = load_data_from_db(symbol)
        if not df.empty and len(df) > 100:
            df_features = add_features(df)
            df_final = add_labels(df_features)
            train_and_save_model(df_final, symbol)
        else:
            print(f"Nicht genügend Daten für {symbol}, um das Training zu starten.")
    
    # Nach dem Training aller Modelle, den Push ausführen
    push_models_to_github()
    
    print("\n--- Nächtliches Training beendet. ---")
