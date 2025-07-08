# backend/predictor_swing.py
import pandas as pd
import joblib
import ta
from ta.utils import dropna

def get_prediction(df, model_path):
    try:
        df = df.copy()
        df = dropna(df)

        # Feature Engineering - EXAKT wie beim Training des Swing-Modells
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['BB_Width'] = ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
        df.dropna(inplace=True)

        if df.empty: return {'error': 'Nicht genügend Daten.'}

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        features = ['RSI', 'SMA_20', 'EMA_50', 'BB_Width']

        if not all(feature in df.columns for feature in features):
            return {'error': f'Features fehlen: {features}'}

        X_predict = df[features].tail(1)
        X_scaled = scaler.transform(X_predict)
        
        prediction = model.predict(X_scaled)
        signal_code = prediction[0]
        
        signal_map = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}
        signal = signal_map.get(signal_code, "Unbekannt")

        entry_price = df.iloc[-1]['close']
        # Hier kannst du deine spezifische TP/SL-Logik für Swing einfügen
        take_profit = entry_price * 1.10 if signal == "Kaufen" else entry_price * 0.90 if signal == "Verkaufen" else None
        stop_loss = entry_price * 0.95 if signal == "Kaufen" else entry_price * 1.05 if signal == "Verkaufen" else None

        return {'signal': signal, 'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss}
    except Exception as e:
        return {'error': f'Fehler in predictor_swing: {e}'}