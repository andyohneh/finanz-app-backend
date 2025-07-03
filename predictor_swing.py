# predictor_swing.py (NEU - für die Live-Vorhersage der Swing-Strategie)
import pandas as pd
import ta
import joblib
import finnhub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv

# --- KONFIGURATION & INITIALISIERUNG ---
load_dotenv()
# WICHTIG: Wir laden jetzt die Swing-Modelle
MODEL_PATH_BTC = 'models/model_swing_BTCUSD.pkl'
MODEL_PATH_XAU = 'models/model_swing_XAUUSD.pkl'
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
sentiment_analyzer = SentimentIntensityAnalyzer()

# --- LIVE-SENTIMENT-ANALYSE ---

def get_live_sentiment(symbol: str) -> float:
    """Holt die neuesten Nachrichten und berechnet einen Live-Sentiment-Score."""
    print(f"Analysiere Live-Nachrichtenstimmung für {symbol}...")
    try:
        category = 'crypto' if symbol == 'BTC/USD' else 'forex'
        news = finnhub_client.general_news(category, min_id=0)[:20]
        if not news:
            print("Keine aktuellen Nachrichten gefunden, verwende neutralen Score.")
            return 0.0
        scores = [sentiment_analyzer.polarity_scores(article['headline'])['compound'] for article in news]
        avg_score = sum(scores) / len(scores)
        print(f"Live-Sentiment-Score: {avg_score:.4f}")
        return avg_score
    except Exception as e:
        print(f"Fehler bei der Live-Sentiment-Analyse: {e}. Verwende neutralen Score.")
        return 0.0

# --- FEATURE ENGINEERING & VORHERSAGE ---

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt die technischen Indikatoren der Swing-Strategie hinzu."""
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=60)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd_diff'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    return df

def get_prediction(symbol: str, data: pd.DataFrame, model_path: str):
    """Trifft eine Swing-Vorhersage basierend auf den neuesten Daten und dem Live-Sentiment."""
    if not os.path.exists(model_path):
        return {"error": f"Modell {model_path} nicht gefunden."}

    model = joblib.load(model_path)
    featured_data = add_features(data.copy())
    live_sentiment = get_live_sentiment(symbol)
    
    last_row = featured_data.iloc[[-1]].copy()
    last_row['sentiment_score'] = live_sentiment
    
    # WICHTIG: Die Features müssen exakt denen aus dem ki_trainer_swing.py entsprechen
    features = ['sma_fast', 'sma_slow', 'rsi', 'macd_diff', 'atr', 'sentiment_score']
    
    last_row.dropna(subset=features, inplace=True)
    if last_row.empty:
        return {"error": "Nicht genügend Daten für eine Vorhersage."}

    X_live = last_row[features]
    
    prediction = model.predict(X_live)[0]
    signal_map = {1: 'Kaufen', -1: 'Verkaufen', 0: 'Halten'}
    
    atr = X_live['atr'].iloc[0]
    last_close = data['close'].iloc[-1]
    # Angepasste TP/SL-Ziele für die Swing-Strategie
    take_profit = last_close + (2.5 * atr)
    stop_loss = last_close - (1.5 * atr)
    
    return {
        "signal": signal_map.get(prediction, "Unbekannt"),
        "entry_price": last_close,
        "take_profit": take_profit,
        "stop_loss": stop_loss
    }