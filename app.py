# app.py (Finale Debug-Version)
import os
import joblib
import pandas as pd
import ta
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, template_folder='templates')
CORS(app)

TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
MODELS = {}
SYMBOLS = ['BTCUSD', 'XAUUSD']

def add_features(df):
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df.dropna(inplace=True)
    return df

def get_live_data_for_features(symbol, interval='1min', outputsize=100):
    try:
        api_symbol = symbol.replace('USD', '/USD')
        url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}"
        
        # --- NEUE DEBUG-ZEILE ---
        print(f"DEBUG: Rufe URL für {symbol} auf: {url}")
        # -------------------------

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'ok' and 'values' in data:
            df = pd.DataFrame(data['values'])
            df = df.rename(columns={'datetime': 'timestamp'})
            if 'volume' not in df.columns:
                df['volume'] = 0.0
            df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"Fehler beim Abrufen der Live-Daten für {symbol}: {e}")
    return None

def load_models():
    print("Lade KI-Modelle...")
    for symbol in SYMBOLS:
        try:
            MODELS[symbol] = joblib.load(f"{symbol.lower()}_model.joblib")
            print(f"Modell '{symbol.lower()}_model.joblib' erfolgreich geladen.")
        except Exception as e:
            print(f"FEHLER beim Laden des Modells für {symbol}: {e}")
    print("Modell-Ladevorgang beendet.")

def get_prediction(symbol):
    if symbol not in MODELS:
        return "Modell nicht verfügbar", 0.0, None, None
    live_df = get_live_data_for_features(symbol)
    if live_df is None or live_df.empty:
        return "Datenfehler", 0.0, None, None
    df_with_features = add_features(live_df)
    if df_with_features.empty:
        return "Nicht genügend Daten für Indikatoren", 0.0, None, None
    latest_data = df_with_features.iloc[-1]
    current_price, current_atr = latest_data['close'], latest_data['atr']
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal']
    latest_features = df_with_features.tail(1)[features]
    prediction = MODELS[symbol].predict(latest_features)
    signal_map = {1: "Kaufen", -1: "Verkaufen", 0: "Halten"}
    signal_text = signal_map.get(prediction[0], "Unbekannt")
    tp_price, sl_price = None, None
    if signal_text == "Kaufen":
        sl_price = current_price - (2 * current_atr)
        tp_price = current_price + (4 * current_atr)
    elif signal_text == "Verkaufen":
        sl_price = current_price + (2 * current_atr)
        tp_price = current_price - (4 * current_atr)
    return signal_text, current_price, tp_price, sl_price

@app.route('/data')
def get_data():
    assets_data = []
    for symbol in SYMBOLS:
        signal, price, take_profit, stop_loss = get_prediction(symbol)
        color, icon = "grey", "minus-circle"
        if signal == "Kaufen": color, icon = "green", "arrow-up"
        elif signal == "Verkaufen": color, icon = "red", "arrow-down"
        assets_data.append({
            "asset": symbol,
            "currentPrice": f"{price:.2f}" if price else "N/A",
            "entry": f"{price:.2f}" if signal in ["Kaufen", "Verkaufen"] else "N/A",
            "takeProfit": f"{take_profit:.2f}" if take_profit else "N/A",
            "stopLoss": f"{stop_loss:.2f}" if stop_loss else "N/A",
            "signal": signal, "color": color, "icon": icon
        })
    return jsonify({"assets": assets_data})

@app.route('/')
def index():
    return render_template('index.html')

load_models()