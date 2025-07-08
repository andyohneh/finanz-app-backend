# backend/predictor_genius.py
import pandas as pd
import joblib
import ta
from ta.utils import dropna

def get_prediction(df, model_path):
    try:
        df = df.copy()
        df = dropna(df)

        # Feature Engineering - EXAKT wie beim Training des Genius-Modells
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['Stoch_RSI'] = ta.momentum.stochrsi(df['close'], window=14, smooth1=3, smooth2=3)
        df['WilliamsR'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        df.dropna(inplace=True)

        if df.empty: return {'error': 'Nicht genügend Daten.'}

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']

        features = ['ADX', 'ATR', 'Stoch_RSI', 'WilliamsR']

        if not all(feature in df.columns for feature in features):
            return {'error': f'Features fehlen: {features}'}
            
        X_predict = df[features].tail(1)
        X_scaled = scaler.transform(X_predict)

        prediction = model.predict(X_scaled)
        signal_code = prediction[0]
        
        signal_map = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}
        signal = signal_map.get(signal_code, "Unbekannt")

        entry_price = df.iloc[-1]['close']
        atr_value = df.iloc[-1]['ATR']
        # Hier kannst du deine spezifische TP/SL-Logik für Genius einfügen
        take_profit = entry_price + (2 * atr_value) if signal == "Kaufen" else entry_price - (2 * atr_value) if signal == "Verkaufen" else None
        stop_loss = entry_price - (1 * atr_value) if signal == "Kaufen" else entry_price + (1 * atr_value) if signal == "Verkaufen" else None
        
        return {'signal': signal, 'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss}
    except Exception as e:
        return {'error': f'Fehler in predictor_genius: {e}'}