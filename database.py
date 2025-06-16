import pandas as pd
import numpy as np
import ta
import joblib
import requests
import os
from sqlalchemy.dialects.postgresql import insert
from database import engine, predictions
from datetime import datetime

TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

def get_live_data_for_features(symbol, interval='1min', outputsize=100):
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol.replace('USD', '/USD')}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'ok' and 'values' in data:
            df = pd.DataFrame(data['values'])
            df = df.rename(columns={'datetime': 'timestamp'})
            df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"Fehler beim Abrufen der Live-Daten für {symbol}: {e}")
    return None

def add_features(df):
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

def update_predictions():
    SYMBOLS = ['BTCUSD', 'XAUUSD']
    MODELS = {}
    for symbol in SYMBOLS:
        try:
            MODELS[symbol] = joblib.load(f"{symbol.lower()}_model.joblib")
        except FileNotFoundError:
            print(f"Modell für {symbol} nicht gefunden.")
            continue
    
    for symbol in SYMBOLS:
        if symbol not in MODELS: continue
        
        live_df = get_live_data_for_features(symbol)
        if live_df is None or live_df.empty: continue

        current_price = live_df.iloc[-1]['close']
        df_with_features = add_features(live_df)
        if df_with_features.empty: continue

        latest_features = df_with_features.tail(1)[['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal']]
        
        prediction = MODELS[symbol].predict(latest_features)
        signal_map = {1: "Kaufen", -1: "Verkaufen", 0: "Halten"}
        signal_text = signal_map.get(prediction[0], "Unbekannt")
        
        stmt = insert(predictions).values(symbol=symbol, signal=signal_text, price=current_price)
        stmt = stmt.on_conflict_do_update(index_elements=['symbol'], set_=dict(signal=signal_text, price=current_price, last_updated=datetime.utcnow()))
        
        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()
            print(f"Vorhersage für {symbol} ({signal_text} @ {current_price}) in DB gespeichert.")

if __name__ == "__main__":
    print("Starte Prediction Worker...")
    update_predictions()
    print("Prediction Worker beendet.")