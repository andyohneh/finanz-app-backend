# backend/predictor_genius.py
import pandas as pd
import joblib
import ta
from ta.utils import dropna

def get_prediction(df, model_path):
    try:
        df_features = df.copy()

        # Features exakt wie im Trainer berechnen
        df_features['ADX'] = ta.trend.adx(df_features['high'], df_features['low'], df_features['close'], window=14)
        df_features['ATR'] = ta.volatility.average_true_range(df_features['high'], df_features['low'], df_features['close'], window=14)
        df_features['Stoch_RSI'] = ta.momentum.stochrsi(df_features['close'], window=14, smooth1=3, smooth2=3)
        df_features['WilliamsR'] = ta.momentum.williams_r(df_features['high'], df_features['low'], df_features['close'], lbp=14)
        df_features.dropna(inplace=True)

        if df_features.empty:
            return {'error': 'Nicht genügend Daten nach Feature Engineering.'}

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Feature-Liste exakt wie im Trainer
        features = ['ADX', 'ATR', 'Stoch_RSI', 'WilliamsR']
        X_predict = df_features[features].tail(1)
        X_scaled = scaler.transform(X_predict)
        
        prediction = model.predict(X_scaled)
        signal_code = int(prediction[0])
        
        signal_map = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}
        signal = signal_map.get(signal_code, "Unbekannt")

        entry_price = df_features.iloc[-1]['close']
        atr_value = df_features.iloc[-1]['ATR']
        take_profit = entry_price + (2 * atr_value) if signal == "Kaufen" else entry_price - (2 * atr_value) if signal == "Verkaufen" else None
        stop_loss = entry_price - (1 * atr_value) if signal == "Kaufen" else entry_price + (1 * atr_value) if signal == "Verkaufen" else None
        
        return {'signal': signal, 'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss}
    except Exception as e:
        return {'error': f'Fehler in predictor_genius: {e}'}