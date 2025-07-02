# ki_trainer_genius.py (Platin-Standard: Trainiert mit technischer UND Sentiment-Analyse)
import pandas as pd
import numpy as np
import ta
import joblib
import os
import requests
from sqlalchemy import text
from dotenv import load_dotenv
from database import engine, historical_data_daily
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint
from transformers import pipeline

# --- KONFIGURATION & MODELL-LADEN ---
load_dotenv()
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

print("Lade Sentiment-Analyse-Modell (kann beim ersten Mal dauern)...")
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
print("Modell geladen.")

def fetch_news_sentiment(keywords: str) -> float:
    """Holt Nachrichten und gibt den durchschnittlichen Sentiment-Score zurück."""
    print(f"Suche Nachrichten-Stimmung für: {keywords}...")
    url = (f"https://newsapi.org/v2/everything?q=({keywords})&language=de&pageSize=20&sortBy=relevancy&apiKey={NEWSAPI_KEY}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', []) if article.get('title')]
        if not headlines: 
            print("Keine Schlagzeilen gefunden.")
            return 0.0
        
        sentiments = sentiment_pipeline(headlines)
        total_score = sum(s['score'] if s['label'] == 'positive' else -s['score'] for s in sentiments)
        average_score = round(total_score / len(sentiments), 4) if headlines else 0
        print(f"Stimmung für '{keywords}' analysiert. Score: {average_score}")
        return average_score
    except Exception as e:
        print(f"Fehler beim Abrufen der Nachrichten-Stimmung: {e}")
        return 0.0

def load_data_from_db(symbol: str):
    """Lädt ALLE TAGES-Daten für ein Symbol aus der Datenbank."""
    print(f"Lade Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            return df
    except Exception as e:
        print(f"Fehler beim Laden der DB-Daten: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame, sentiment_score: float) -> pd.DataFrame:
    """Fügt technische Indikatoren UND den Sentiment-Score als Feature hinzu."""
    print("Füge technische und Sentiment-Features hinzu...")
    df_copy = df.copy()
    df_copy['sma_fast'] = ta.trend.sma_indicator(df_copy['close'], window=20)
    df_copy['sma_slow'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd = ta.trend.MACD(df_copy['close'], window_slow=26, window_fast=12, window_sign=9); df_copy['macd'] = macd.macd(); df_copy['macd_signal'] = macd.macd_signal()
    df_copy['atr'] = ta.volatility.AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df_copy['close'], window=20, window_dev=2); df_copy['bb_high'] = bollinger.bollinger_hband(); df_copy['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14, smooth_window=3); df_copy['stoch_k'] = stoch.stoch(); df_copy['stoch_d'] = stoch.stoch_signal()
    
    df_copy['sentiment'] = sentiment_score
    
    df_copy.dropna(inplace=True)
    return df_copy

def get_labels_triple_barrier(prices, tp_mult, sl_mult, max_period):
    labels = pd.Series(np.nan, index=prices.index)
    log_returns = np.log(prices['close'] / prices['close'].shift(1))
    volatility = log_returns.rolling(window=100).std() * 2
    for i in range(len(prices) - max_period):
        entry_price = prices['close'].iloc[i]
        tp_barrier = entry_price * (1 + volatility.iloc[i] * tp_mult)
        sl_barrier = entry_price * (1 - volatility.iloc[i] * sl_mult)
        for j in range(1, max_period + 1):
            future_price = prices['close'].iloc[i + j]
            if future_price >= tp_barrier: labels.iloc[i] = 1; break
            elif future_price <= sl_barrier: labels.iloc[i] = -1; break
        if pd.isna(labels.iloc[i]): labels.iloc[i] = 0
    return labels

def add_labels(df: pd.DataFrame, tp_mult=2.0, sl_mult=1.5, max_period=60) -> pd.DataFrame:
    print(f"Starte Triple-Barrier-Labeling (max_period = {max_period} Tage)...")
    labels = get_labels_triple_barrier(df[['close']], tp_mult, sl_mult, max_period)
    df['label'] = labels
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"Labeling abgeschlossen. Verteilung:\n{df['label'].value_counts(normalize=True)}")
    return df

def train_genius_model(df, symbol):
    """Trainiert das neue 'Genie'-Modell mit dem Sentiment-Feature."""
    feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 'sentiment']
    X = df[feature_columns]
    y = df['label'] + 1 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model_params = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {'n_estimators': randint(100, 500), 'max_depth': [10, 20, 30, None], 'min_samples_leaf': randint(1, 5)}
        },
        'XGBClassifier': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'params': {'n_estimators': randint(100, 500), 'max_depth': randint(3, 10), 'learning_rate': [0.01, 0.1, 0.2]}
        },
        'LGBMClassifier': {
            'model': LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
            'params': {'n_estimators': randint(100, 500), 'learning_rate': [0.01, 0.1, 0.2], 'num_leaves': randint(20, 50)}
        }
    }

    best_score = 0; best_model = None; champion_name = ""
    print("\n--- STARTE 'GENIE' KI-WETTKAMPF ---")
    for model_name, mp in model_params.items():
        print(f"\n===== Teste Champion: {model_name} =====")
        random_search = RandomizedSearchCV(mp['model'], param_distributions=mp['params'], n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
        random_search.fit(X_train, y_train)
        
        if random_search.best_score_ > best_score:
            best_score = random_search.best_score_
            best_model = random_search.best_estimator_
            champion_name = model_name
            print(f"--- NEUER CHAMPION: {model_name} mit Score {random_search.best_score_:.4f} ---")

    print(f"\n\n--- DER SIEGER IST: {champion_name} mit einem Score von {best_score:.4f} ---")
    
    print("\nPerformance des 'Genie'-Modells auf den Test-Daten:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Verkaufen (-1)', 'Halten (0)', 'Kaufen (1)']))
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_genius.pkl')
    joblib.dump(best_model, model_path)
    print(f"'Genie'-Modell erfolgreich gespeichert unter: {model_path}")

if __name__ == '__main__':
    btc_sentiment = fetch_news_sentiment("Bitcoin OR BTC")
    xau_sentiment = fetch_news_sentiment("Gold OR Goldpreis OR XAU")
    
    sentiment_scores = {
        'BTC/USD': btc_sentiment,
        'XAU/USD': xau_sentiment
    }

    for symbol, sentiment in sentiment_scores.items():
        print(f"\n--- Starte 'Genie'-Training für {symbol} mit Sentiment {sentiment} ---")
        df = load_data_from_db(symbol)
        if not df.empty and len(df) > 250:
            df_with_features = add_features(df, sentiment)
            df_labeled = add_labels(df_with_features)
            
            if not df_labeled.empty:
                train_genius_model(df_labeled, symbol)