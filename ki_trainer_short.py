# ki_trainer_short_legendary.py (Trainiert die beste Short-KI)
import pandas as pd
import numpy as np
import ta
import joblib
import os
from sqlalchemy import text
from database import engine, historical_data_daily
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint

# --- KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']

def load_data_from_db(symbol: str):
    """Lädt ALLE TAGES-Daten für ein Symbol aus der Datenbank."""
    print(f"Lade alle Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            return df
    except Exception as e:
        return pd.DataFrame()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt die 10 Kern-Indikatoren als Features hinzu."""
    print(f"Füge für {len(df)} Tages-Datenpunkte Features hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband(); df['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
    df.dropna(inplace=True)
    return df

def get_short_labels_triple_barrier(prices, tp_mult, sl_mult, max_period):
    """Erstellt Labels für Short-Trades."""
    labels = pd.Series(np.nan, index=prices.index)
    log_returns = np.log(prices['close'] / prices['close'].shift(1))
    volatility = log_returns.rolling(window=100).std() * 2
    for i in range(len(prices) - max_period):
        entry_price = prices['close'].iloc[i]
        take_profit_barrier = entry_price * (1 - volatility.iloc[i] * tp_mult)
        stop_loss_barrier = entry_price * (1 + volatility.iloc[i] * sl_mult)
        for j in range(1, max_period + 1):
            future_price = prices['close'].iloc[i + j]
            if future_price <= take_profit_barrier:
                labels.iloc[i] = -1; break
            elif future_price >= stop_loss_barrier:
                labels.iloc[i] = 1; break
        if pd.isna(labels.iloc[i]):
            labels.iloc[i] = 0
    return labels

def add_labels(df: pd.DataFrame, tp_mult=2.0, sl_mult=1.5, max_period=60) -> pd.DataFrame:
    """Fügt die Ziel-Labels zum DataFrame hinzu."""
    print(f"Starte Short-Labeling (max_period = {max_period} Tage)...")
    labels = get_short_labels_triple_barrier(df[['close']], tp_mult, sl_mult, max_period)
    df['label'] = labels
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"Labeling abgeschlossen. Verteilung:\n{df['label'].value_counts(normalize=True)}")
    return df

def train_legendary_short_model(df, symbol):
    """Führt einen Wettkampf durch, um das beste SHORT-Modell zu finden."""
    feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
    X = df[feature_columns]
    y = df['label'] + 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_params = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {'n_estimators': randint(100, 500), 'max_depth': [10, 20, None], 'min_samples_leaf': randint(1, 5)}
        },
        'XGBClassifier': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', verbose=-1),
            'params': {'n_estimators': randint(100, 500), 'max_depth': randint(3, 10), 'learning_rate': [0.01, 0.1, 0.2]}
        },
        'LGBMClassifier': {
            'model': LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
            'params': {'n_estimators': randint(100, 500), 'learning_rate': [0.01, 0.1, 0.2], 'num_leaves': randint(20, 50)}
        }
    }

    best_score = 0; best_model = None; champion_name = ""

    print("\n--- STARTE LEGENDÄREN KI-WETTKAMPF (SHORT) ---")
    for model_name, mp in model_params.items():
        print(f"\n===== Teste Champion: {model_name} =====")
        random_search = RandomizedSearchCV(mp['model'], param_distributions=mp['params'], n_iter=25, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
        random_search.fit(X_train, y_train)
        
        if random_search.best_score_ > best_score:
            best_score = random_search.best_score_
            best_model = random_search.best_estimator_
            champion_name = model_name
            print(f"--- NEUER CHAMPION: {model_name} mit Score {random_search.best_score_:.4f} ---")

    print(f"\n\n--- DER SIEGER IST: {champion_name} mit einem Score von {best_score:.4f} ---")
    
    print("\nPerformance des legendären SHORT-Modells auf den Test-Daten:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)']))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_short.pkl')
    joblib.dump(best_model, model_path)
    print(f"Legendäres Short-Modell erfolgreich gespeichert unter: {model_path}")

def run_training_pipeline():
    for symbol in SYMBOLS:
        print(f"\n--- Starte Short-Verarbeitung für {symbol} ---")
        df = load_data_from_db(symbol)
        if not df.empty and len(df) > 250:
            df_features = add_features(df)
            df_labeled = add_labels(df_features)
            if not df_labeled.empty:
                train_legendary_short_model(df_labeled, symbol)

if __name__ == '__main__':
    run_training_pipeline()