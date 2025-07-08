# backend/predictor_genius.py
import pandas as pd
import joblib
import ta
from sqlalchemy import text
from database import engine

def load_live_data_with_sentiment(symbol: str) -> pd.DataFrame:
    # Diese Funktion ist identisch und könnte in ein gemeinsames Modul
    with engine.connect() as conn:
        query_prices = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 400")
        df = pd.read_sql_query(query_prices, conn, params={'symbol': symbol})
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        query_sentiment = text("SELECT sentiment_score FROM daily_sentiment WHERE asset = :asset ORDER BY date DESC LIMIT 1")
        sentiment_score = conn.execute(query_sentiment, {"asset": symbol}).scalar_one_or_none()
        df['sentiment_score'] = sentiment_score if sentiment_score is not None else 0.0
        return df

def get_prediction(df_raw, model_path):
    try:
        model_data = joblib.load(model_path)
        model, scaler, features = model_data['model'], model_data['scaler'], model_data['features']

        df_features = df_raw.copy()
        df_features['ADX'] = ta.trend.adx(df_features['high'], df_features['low'], df_features['close'], window=14)
        df_features['ATR'] = ta.volatility.average_true_range(df_features['high'], df_features['low'], df_features['close'], window=14)
        df_features['Stoch_RSI'] = ta.momentum.stochrsi(df_features['close'], window=14, smooth1=3, smooth2=3)
        df_features['WilliamsR'] = ta.momentum.williams_r(df_features['high'], df_features['low'], df_features['close'], lbp=14)
        df_features.dropna(inplace=True)
        
        X_predict = df_features[features].tail(1)
        X_scaled = scaler.transform(X_predict)
        prediction = model.predict(X_scaled)
        
        signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(int(prediction[0]))
        price = df_features.iloc[-1]['close']
        atr = df_features.iloc[-1]['ATR']
        return {'signal': signal, 'entry_price': price, 'take_profit': price + (2*atr), 'stop_loss': price - atr}
    except Exception as e:
        return {'error': str(e)}