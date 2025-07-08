# backend/predictor_swing.py (Finale, korrigierte Version)
import pandas as pd
import joblib
import ta
from ta.utils import dropna

def get_prediction(df, model_path):
    try:
        df = df.copy()
        df = dropna(df)

        # 1. Feature Engineering
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['BB_width'] = ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
        df.dropna(inplace=True)

        if df.empty:
            return {'error': 'Nicht genügend Daten nach Feature Engineering.'}

        # 2. Modell laden
        model = joblib.load(model_path)
        
        # 3. Features für die Vorhersage vorbereiten
        features = ['open', 'high', 'low', 'close', 'volume', 'SMA_50', 'EMA_20', 'RSI', 'BB_width']
        
        if not all(feature in df.columns for feature in features):
            return {'error': 'Einige Features fehlen im DataFrame.'}
            
        X_predict = df[features].tail(1)

        if X_predict.empty:
            return {'error': 'Keine Daten für die Vorhersage vorhanden.'}

        # 4. Vorhersage treffen
        prediction = model.predict(X_predict)
        
        # KORREKTUR: Das Ergebnis ist ein Array, wir brauchen das erste Element.
        signal_code = prediction[0]
        
        signal_map = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}
        signal = signal_map.get(signal_code, "Unbekannt")

        # 5. Trade-Parameter berechnen
        last_row = df.iloc[-1]
        entry_price = last_row['close']
        
        if signal == "Kaufen":
            take_profit = entry_price * 1.10
            stop_loss = entry_price * 0.95
        elif signal == "Verkaufen":
            take_profit = entry_price * 0.90
            stop_loss = entry_price * 1.05
        else: # Halten
            take_profit = None
            stop_loss = None

        return {
            'signal': signal,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }
    except Exception as e:
        return {'error': f'Fehler in predictor_swing: {e}'}