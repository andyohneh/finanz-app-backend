# backend/predictor_daily.py
import pandas as pd
import joblib
import ta

def get_prediction(df, model_path):
    try:
        df_features = df.copy()
        model_data = joblib.load(model_path)
        model, scaler, features = model_data['model'], model_data['scaler'], model_data['features']

        # Features exakt wie im Trainer berechnen
        df_features['RSI'] = ta.momentum.rsi(df_features['close'], window=14)
        df_features['SMA_50'] = ta.trend.sma_indicator(df_features['close'], window=50)
        df_features['SMA_200'] = ta.trend.sma_indicator(df_features['close'], window=200)
        df_features['MACD_diff'] = ta.trend.macd_diff(df_features['close'], window_slow=26, window_fast=12, window_sign=9)
        df_features.dropna(inplace=True)

        # Vorhersage mit der korrekten Feature-Reihenfolge
        X_predict = df_features[features].tail(1)
        X_scaled = scaler.transform(X_predict)
        prediction = model.predict(X_scaled)
        
        signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(int(prediction[0]))
        price = df_features.iloc[-1]['close']
        return {'signal': signal, 'entry_price': price, 'take_profit': price * 1.05, 'stop_loss': price * 0.98}
    except Exception as e: return {'error': str(e)}