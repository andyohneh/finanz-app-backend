# backend/predictor_daily.py
import pandas as pd
import joblib
import ta
from ta.utils import dropna

def get_prediction(df, model_path):
    try:
        df_features = df.copy()

        # Features exakt wie im Trainer berechnen
        df_features['RSI'] = ta.momentum.rsi(df_features['close'], window=14)
        df_features['SMA_50'] = ta.trend.sma_indicator(df_features['close'], window=50)
        df_features['SMA_200'] = ta.trend.sma_indicator(df_features['close'], window=200)
        df_features['MACD_diff'] = ta.trend.macd_diff(df_features['close'], window_slow=26, window_fast=12, window_sign=9)
        df_features.dropna(inplace=True)

        if df_features.empty:
            return {'error': 'Nicht genügend Daten nach Feature Engineering.'}

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Feature-Liste exakt wie im Trainer
        features = ['RSI', 'SMA_50', 'SMA_200', 'MACD_diff']
        X_predict = df_features[features].tail(1)
        X_scaled = scaler.transform(X_predict)
        
        prediction = model.predict(X_scaled)
        signal_code = int(prediction[0])
        
        signal_map = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}
        signal = signal_map.get(signal_code, "Unbekannt")

        entry_price = df_features.iloc[-1]['close']
        take_profit = entry_price * 1.05 if signal == "Kaufen" else entry_price * 0.95 if signal == "Verkaufen" else None
        stop_loss = entry_price * 0.98 if signal == "Kaufen" else entry_price * 1.02 if signal == "Verkaufen" else None

        return {'signal': signal, 'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss}
    except Exception as e:
        return {'error': f'Fehler in predictor_daily: {e}'}