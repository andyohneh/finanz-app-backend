# backend/predictor_swing.py
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
        df_features['SMA_20'] = ta.trend.sma_indicator(df_features['close'], window=20)
        df_features['EMA_50'] = ta.trend.ema_indicator(df_features['close'], window=50)
        df_features['BB_Width'] = ta.volatility.bollinger_wband(df_features['close'], window=20, window_dev=2)
        df_features.dropna(inplace=True)
        
        X_predict = df_features[features].tail(1)
        X_scaled = scaler.transform(X_predict)
        prediction = model.predict(X_scaled)
        
        signal = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}.get(int(prediction[0]))
        price = df_features.iloc[-1]['close']
        return {'signal': signal, 'entry_price': price, 'take_profit': price * 1.1, 'stop_loss': price * 0.95}
    except Exception as e: return {'error': str(e)}