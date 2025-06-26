# ki_trainer_4h.py (Version 2.0 - trainiert das 'Genie'-Modell mit Tages-Trend als Kontext)
import pandas as pd
import numpy as np
import ta
import joblib
import os
from sqlalchemy import text
from database import engine, historical_data_4h, historical_data_daily # Beide Tabellen importieren
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']

def load_data(symbol: str):
    """Lädt SOWOHL die 4h-Daten als auch die Tages-Daten aus der Datenbank."""
    print(f"Lade 4h- und Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query_4h = text("SELECT * FROM historical_data_4h WHERE symbol = :symbol ORDER BY timestamp ASC")
            df_4h = pd.read_sql_query(query_4h, conn, params={'symbol': symbol})
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp']) # Sicherstellen, dass es ein Datetime-Objekt ist
            
            query_daily = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df_daily = pd.read_sql_query(query_daily, conn, params={'symbol': symbol})
            df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp']) # Sicherstellen, dass es ein Datetime-Objekt ist

            print(f"Erfolgreich {len(df_4h)} 4h-Punkte und {len(df_daily)} Tages-Punkte geladen.")
            return df_4h, df_daily
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return pd.DataFrame(), pd.DataFrame()

def add_contextual_features(df_4h: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    """Fügt der 4h-KI den übergeordneten Tages-Trend als Feature hinzu."""
    print("Füge Kontext-Features hinzu...")
    
    # 1. Berechne den Tages-Trend
    df_daily['sma_trend_daily'] = ta.trend.sma_indicator(df_daily['close'], window=50)
    df_daily['is_daily_uptrend'] = (df_daily['close'] > df_daily['sma_trend_daily']).astype(int)
    
    # Bereite die Tages-Daten für den Merge vor (nur Datum, kein Zeitstempel)
    df_daily['date'] = df_daily['timestamp'].dt.date
    daily_context = df_daily[['date', 'is_daily_uptrend']].drop_duplicates(subset='date', keep='last')
    
    # 2. Füge das Tages-Feature zu den 4h-Daten hinzu
    df_4h['date'] = df_4h['timestamp'].dt.date
    df_merged = pd.merge(df_4h, daily_context, on='date', how='left')
    
    # Fülle fehlende Werte (für Wochenenden etc.) mit dem letzten bekannten Wert
    df_merged['is_daily_uptrend'].fillna(method='ffill', inplace=True)
    df_merged.drop(columns=['date'], inplace=True)
    
    print("Kontext-Features erfolgreich hinzugefügt.")
    return df_merged

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt die normalen technischen 4h-Indikatoren hinzu."""
    print("Füge technische 4h-Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9); df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2); df['bb_high'] = bollinger.bollinger_hband(); df['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3); df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
    # WICHTIG: Die dropna() Funktion muss am Ende kommen, nachdem alle Features berechnet wurden.
    df.dropna(inplace=True)
    return df

def get_labels_triple_barrier(prices, tp_mult, sl_mult, max_period):
    """Erstellt Labels basierend auf der Triple-Barrier-Methode."""
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
                labels.iloc[i] = 1; break
            elif future_price <= sl_barrier:
                labels.iloc[i] = -1; break
        if pd.isna(labels.iloc[i]):
            labels.iloc[i] = 0
    return labels

def add_labels(df: pd.DataFrame, tp_mult=2.5, sl_mult=1.5, max_period=30) -> pd.DataFrame:
    """Fügt die Ziel-Labels zum DataFrame hinzu."""
    print(f"Starte Triple-Barrier-Labeling (max_period = {max_period} * 4 Stunden)...")
    labels = get_labels_triple_barrier(df[['close']], tp_mult, sl_mult, max_period)
    df['label'] = labels
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"Labeling abgeschlossen. Verteilung:\n{df['label'].value_counts(normalize=True)}")
    return df

def train_and_save_genius_model(df: pd.DataFrame, symbol: str):
    """Trainiert das neue 'Genie'-Modell und speichert es."""
    # WICHTIG: Die neue Spalte 'is_daily_uptrend' ist jetzt ein Feature!
    feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 'is_daily_uptrend']
    X = df[feature_columns]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    param_grid = { 'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_leaf': [1, 2, 4]}
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    print("\n--- Starte Hyperparameter-Tuning für 'Genie'-Modell... ---")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nBeste gefundene Parameter: {grid_search.best_params_}")

    print("\nPerformance des 'Genie'-Modells auf Test-Daten:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_4h_genius.pkl')
    joblib.dump(best_model, model_path)
    print(f"Bestes 'Genie'-Modell erfolgreich gespeichert unter: {model_path}")

if __name__ == '__main__':
    for symbol in SYMBOLS:
        print(f"\n--- Starte 'Genie'-Training für {symbol} ---")
        df_4h, df_daily = load_data(symbol)
        
        if not df_4h.empty and not df_daily.empty:
            df_with_context = add_contextual_features(df_4h, df_daily)
            df_with_features = add_technical_features(df_with_context)
            df_labeled = add_labels(df_with_features)
            
            if not df_labeled.empty:
                train_and_save_genius_model(df_labeled, symbol)