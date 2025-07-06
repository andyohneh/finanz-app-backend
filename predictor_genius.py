# backend/predictor_genius.py
import pandas as pd
import ta
import joblib
from sklearn.preprocessing import StandardScaler

def calculate_trade_levels(signal, current_price):
    entry_price = current_price
    if signal == "Kaufen":
        entry_price = current_price * 0.997
    elif signal == "Verkaufen":
        entry_price = current_price * 1.003

    sl_percentage = 0.01
    tp_percentage = 0.05

    if signal == "Kaufen":
        stop_loss = entry_price * (1 - sl_percentage)
        take_profit = entry_price * (1 + tp_percentage)
    elif signal == "Verkaufen":
        stop_loss = entry_price * (1 + sl_percentage)
        take_profit = entry_price * (1 - tp_percentage)
    else:
        stop_loss, take_profit = 0, 0
        
    return round(entry_price, 4), round(take_profit, 4), round(stop_loss, 4)

def get_prediction(df, model_path):
    try:
        # Genius verwendet mehr Indikatoren
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['Stoch_RSI'] = ta.momentum.stochrsi(df['close'], window=14, smooth1=3, smooth2=3)
        df['Williams_%R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df.dropna(inplace=True)

        if df.empty:
            return {'error': 'DataFrame nach Indikatoren leer.'}

        model_data = joblib.load(model_path)
        model, scaler = model_data['model'], model_data['scaler']
        
        features = df[['RSI', 'Stoch_RSI', 'Williams_%R', 'ATR']]
        last_features = features.tail(1)
        scaled_features = scaler.transform(last_features)
        
        prediction_numeric = model.predict(scaled_features)[0]
        
        signal_map = {0: "Verkaufen", 1: "Kaufen", 2: "Halten"}
        signal_text = signal_map.get(prediction_numeric, "Unbekannt")
        
        current_price = df['close'].iloc[-1]
        entry, tp, sl = calculate_trade_levels(signal_text, current_price)
        
        return {"signal": signal_text, "entry_price": entry, "take_profit": tp, "stop_loss": sl}
        
    except FileNotFoundError:
        return {'error': f"Modell-Datei nicht gefunden: {model_path}"}
    except Exception as e:
        return {'error': f"Fehler in predictor_genius: {str(e)}"}