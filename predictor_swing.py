# backend/predictor_swing.py
import pandas as pd
import joblib
import ta
from ta.utils import dropna

def get_prediction(df, model_path):
    try:
        df_features = df.copy()

        # Features from the 'swing' training
        df_features['RSI'] = ta.momentum.rsi(df_features['close'], window=14)
        df_features['SMA_20'] = ta.trend.sma_indicator(df_features['close'], window=20)
        df_features['EMA_50'] = ta.trend.ema_indicator(df_features['close'], window=50)
        df_features['BB_Width'] = ta.volatility.bollinger_wband(df_features['close'], window=20, window_dev=2)
        df_features.dropna(inplace=True)

        if df_features.empty:
            return {'error': 'Not enough data after feature engineering.'}

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        features = ['RSI', 'SMA_20', 'EMA_50', 'BB_Width']
        X_predict = df_features[features].tail(1)
        X_scaled = scaler.transform(X_predict)
        
        prediction = model.predict(X_scaled)
        signal_code = int(prediction[0])
        
        signal_map = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}
        signal = signal_map.get(signal_code, "Unbekannt")

        entry_price = df.iloc[-1]['close']
        take_profit = entry_price * 1.10 if signal == "Kaufen" else entry_price * 0.90 if signal == "Verkaufen" else None
        stop_loss = entry_price * 0.95 if signal == "Kaufen" else entry_price * 1.05 if signal == "Verkaufen" else None

        return {'signal': signal, 'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss}
    except Exception as e:
        return {'error': f'Error in predictor_swing: {e}'}