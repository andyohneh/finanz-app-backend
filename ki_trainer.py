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
    """Fügt technische Indikatoren als neue Spalten zum DataFrame hinzu."""
    print(f"Füge für {len(df)} Datenpunkte Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # ### NEUER CODEBLOCK: BOLLINGER BÄNDER ###
    # Initialisiere den Bollinger Band Indikator
    indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    # Füge die Bänder und nützliche Werte als Features hinzu
    df['bb_high_band'] = indicator_bb.bollinger_hband()  # Oberes Band
    df['bb_low_band'] = indicator_bb.bollinger_lband()   # Unteres Band
    df['bb_pband'] = indicator_bb.bollinger_pband()      # %B (Position des Kurses innerhalb der Bänder)
    df['bb_wband'] = indicator_bb.bollinger_wband()      # Bandbreite (Maß für Volatilität)
    # ### ENDE NEUER CODEBLOCK ###
    
    df.dropna(inplace=True)
    return df

def add_labels(df, future_periods=240, percent_change_threshold=0.003):
    print(f"Füge Ziel-Labels hinzu (Horizont: {future_periods} min, Schwelle: {percent_change_threshold*100:.2f}%)...")
    future_close = df['close'].shift(-future_periods)
    df['price_change'] = (future_close - df['close']) / df['close']
    
    df['signal'] = 0
    df.loc[df['price_change'] > percent_change_threshold, 'signal'] = 1
    df.loc[df['price_change'] < -percent_change_threshold, 'signal'] = -1
    
    return df.dropna(subset=['price_change', 'signal'])

def train_and_save_model(df, symbol):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    print(f"Starte Training für {symbol}...")
    
    # ### ERWEITERT: Die Liste der Features wurde um die Bollinger Bänder ergänzt ###
    features = [
        'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr',
        'bb_high_band', 'bb_low_band', 'bb_pband', 'bb_wband'
    ]
    target = 'signal'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print(f"Training für {symbol} abgeschlossen.\n")
    print(f"Leistungsbericht für {symbol} auf Test-Daten:")
    print(classification_report(y_test, model.predict(X_test), target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)']))
    
    safe_symbol_name = symbol.replace('/', '')
    model_filename = f'models/{safe_symbol_name}_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nModell als '{model_filename}' gespeichert.")

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
        remote_url = f"https://{git_user}:{git_pat}@github.com/{repo_name}.git"
        subprocess.run(['git', 'push', remote_url], check=True)
        print("Modelle erfolgreich auf GitHub hochgeladen!")
    except Exception as e:
        print(f"Ein Fehler beim Git-Push ist aufgetreten: {e}")

def main():
    SYMBOLS = ['BTC/USD', 'XAU/USD']
    
    for symbol in SYMBOLS:
        process_symbol(symbol)
        
    push_models_to_github()
    print("\n--- Nächtliches Training beendet. ---")

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

if __name__ == "__main__":
    main()