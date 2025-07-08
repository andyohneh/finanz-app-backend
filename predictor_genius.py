# backend/predictor_genius.py
import pandas as pd
import joblib
import ta

def get_prediction(df, model_path):
    try:
        df_features = df.copy()
        model_data = joblib.load(model_path)
        model, scaler, features = model_data['model'], model_data['scaler'], model_data['features']

        # Features exakt wie im Trainer berechnen
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
    except Exception as e: return {'error': str(e)}