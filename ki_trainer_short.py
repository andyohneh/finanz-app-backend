# ki_trainer_short.py (Spezialisiert auf Short-Signale auf Tages-Daten)
import pandas as pd
import numpy as np
import ta
import joblib
import os
from sqlalchemy import text
from database import engine, historical_data_daily
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']

def load_data_from_db(symbol: str):
    print(f"Lade alle Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            return df
    except Exception as e:
        return pd.DataFrame(), print(f"Fehler: {e}")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (Die Funktion bleibt identisch zu ki_trainer_daily.py) ...
    print(f"Füge Features für Short-Modell hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20); df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9); df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2); df['bb_high'] = bollinger.bollinger_hband(); df['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3); df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
    df.dropna(inplace=True)
    return df

def get_short_labels_triple_barrier(prices, tp_mult, sl_mult, max_period):
    """
    NEUE LABELING-LOGIK FÜR SHORT-TRADES
    -1 = Guter Short-Trade (Preis fällt, Take Profit erreicht)
     1 = Schlechter Short-Trade (Preis steigt, Stop Loss erreicht)
     0 = Halten
    """
    labels = pd.Series(np.nan, index=prices.index)
    log_returns = np.log(prices['close'] / prices['close'].shift(1))
    volatility = log_returns.rolling(window=100).std() * 2
    
    for i in range(len(prices) - max_period):
        entry_price = prices['close'].iloc[i]
        
        # WICHTIG: Die Logik ist umgekehrt! Take Profit ist UNTEN, Stop Loss ist OBEN.
        take_profit_barrier = entry_price * (1 - volatility.iloc[i] * tp_mult)
        stop_loss_barrier = entry_price * (1 + volatility.iloc[i] * sl_mult)

        for j in range(1, max_period + 1):
            future_price = prices['close'].iloc[i + j]
            if future_price <= take_profit_barrier: # Preis fällt zum Ziel
                labels.iloc[i] = -1 # Guter Short-Trade
                break
            elif future_price >= stop_loss_barrier: # Preis steigt zum Stop
                labels.iloc[i] = 1 # Schlechter Short-Trade
                break
        
        if pd.isna(labels.iloc[i]):
            labels.iloc[i] = 0 # Halten

    return labels

def add_labels(df, tp_mult=2.0, sl_mult=1.5, max_period=60):
    print(f"Starte Short-Labeling (max_period = {max_period} Tage)...")
    labels = get_short_labels_triple_barrier(df[['close']], tp_mult, sl_mult, max_period)
    df['label'] = labels
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"Labeling abgeschlossen. Verteilung:\n{df['label'].value_counts(normalize=True)}")
    return df

def train_and_save_model(df, symbol):
    # ... (Die Trainingslogik mit GridSearchCV bleibt identisch) ...
    # Nur der Speichername des Modells wird angepasst.
    feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
    X = df[feature_columns]; y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    param_grid = { 'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [2, 4] }
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    print("\n--- Starte Hyperparameter-Tuning für Short-Modell... ---")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"\nBeste Parameter für Short-Modell: {grid_search.best_params_}")
    print("\nPerformance des besten SHORT-Modells:")
    y_pred = best_model.predict(X_test); print(classification_report(y_test, y_pred))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_short.pkl')
    joblib.dump(best_model, model_path)
    print(f"Bestes Short-Modell erfolgreich gespeichert unter: {model_path}")

def run_training_pipeline():
    for symbol in SYMBOLS:
        print(f"\n--- Starte Short-Trading-Verarbeitung für {symbol} ---")
        df = load_data_from_db(symbol);
        if not df.empty and len(df) > 250:
            df_features = add_features(df); df_labeled = add_labels(df_features)
            if not df_labeled.empty:
                train_and_save_model(df_labeled, symbol)

if __name__ == '__main__':
    run_training_pipeline()