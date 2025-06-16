import pandas as pd
import numpy as np
import ta
from sqlalchemy import text
from database import engine
import joblib # NEUER IMPORT zum Speichern des Modells

# IMPORTE für Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data_from_db(symbol: str, limit: int = 2000):
    """
    Lädt die letzten 'limit' Datenpunkte für ein Symbol aus der Datenbank.
    """
    print(f"Lade die letzten {limit} Datenpunkte für {symbol} aus der Datenbank...")
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT * FROM historical_data 
                WHERE symbol = '{symbol}' 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """)
            df = pd.read_sql_query(query, conn)
            df = df.iloc[::-1].reset_index(drop=True)
            print(f"Erfolgreich {len(df)} Datenpunkte geladen.")
            return df
    except Exception as e:
        print(f"Ein Fehler beim Laden der Daten ist aufgetreten: {e}")
        return pd.DataFrame()

def add_features(df):
    """
    Fügt technische Indikatoren als neue Spalten zum DataFrame hinzu.
    """
    print("Füge technische Indikatoren (Features) hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    print("Features erfolgreich hinzugefügt.")
    return df

def add_labels(df, future_periods=15, percent_change_threshold=0.0005):
    """
    Erstellt die Ziel-Labels (Kaufen/Verkaufen/Halten) basierend auf zukünftigen Preisänderungen.
    """
    print("Füge Ziel-Labels (1, 0, -1) hinzu...")
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
    print("Labels erfolgreich hinzugefügt.")
    return df

def train_and_evaluate(df):
    """
    Trainiert ein RandomForest-Modell und bewertet seine Leistung.
    """
    print("Starte Training und Evaluierung des Modells...")
    
    # 1. Features (X) und Ziel (y) definieren
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal']
    X = df[features]
    y = df['target']
    
    # 2. Daten aufteilen in Trainings- (80%) und Test-Set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Daten aufgeteilt: {len(X_train)} Trainings-Punkte, {len(X_test)} Test-Punkte.")
    
    # 3. Modell initialisieren
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # 4. Modell trainieren
    print("Trainiere das Modell...")
    model.fit(X_train, y_train)
    print("Training abgeschlossen.")
    
    # 5. Vorhersagen auf den Test-Daten machen
    y_pred = model.predict(X_test)
    
    # 6. Ergebnisse bewerten und ausgeben
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModell-Genauigkeit (Accuracy): {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)']))
    
    # --- NEUER TEIL: MODELL SPEICHERN ---
    model_filename = f"{symbol.lower()}_model.joblib"
    print(f"Speichere das trainierte Modell als '{model_filename}'...")
    joblib.dump(model, model_filename)
    print("Modell erfolgreich gespeichert.")
    # ------------------------------------
    
    return model

def process_symbol(symbol: str):
    """
    Führt den gesamten Prozess für ein einzelnes Symbol aus.
    """
    print(f"\n--- Starte Verarbeitung für {symbol} ---")
    data = load_data_from_db(symbol)
    
    if not data.empty:
        data_with_features = add_features(data)
        final_data = add_labels(data_with_features)
        
        if not final_data.empty:
            train_and_evaluate(final_data)
        else:
            print(f"Nicht genügend Daten für {symbol}, um den Prozess abzuschließen.")
            
    print(f"--- Verarbeitung für {symbol} beendet ---")


# --- Haupt-Logik ---
if __name__ == "__main__":
    symbols_to_process = ['BTCUSD', 'XAUUSD']
    for symbol in symbols_to_process:
        process_symbol(symbol)