# app.py (Version 2.0 mit TP/SL)
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
    """Fügt technische Indikatoren inkl. ATR als neue Spalten zum DataFrame hinzu."""
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # NEU: ATR (Average True Range) für die Volatilität hinzufügen
    atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()
    
    df.dropna(inplace=True)
    return df

def get_live_data_for_features(symbol, interval='1min', outputsize=100):
    """Holt genügend Live-Daten von der API, um die Features berechnen zu können."""
    try:
        api_symbol = symbol.replace('USD', '/USD')
        url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}"
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
    """Lädt die trainierten Modelle beim Start der App in den Speicher."""
    print("Lade KI-Modelle...")
    for symbol in SYMBOLS:
        model_filename = f"{symbol.lower()}_model.joblib"
        try:
            MODELS[symbol] = joblib.load(model_filename)
            print(f"Modell '{model_filename}' erfolgreich geladen.")
        except FileNotFoundError:
            print(f"WARNUNG: Modelldatei '{model_filename}' nicht gefunden.")
    print("Alle verfügbaren Modelle geladen.")

def get_prediction(symbol):
    """Macht eine Vorhersage und berechnet TP/SL."""
    if symbol not in MODELS:
        return "Modell nicht verfügbar", 0.0, None, None

    live_df = get_live_data_for_features(symbol)
    if live_df is None or live_df.empty:
        return "Datenfehler", 0.0, None, None

    df_with_features = add_features(live_df)
    if df_with_features.empty:
        return "Nicht genügend Daten für Indikatoren", 0.0, None, None

    latest_data = df_with_features.iloc[-1]
    current_price = latest_data['close']
    current_atr = latest_data['atr']
    
    features_for_prediction = ['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal']
    latest_features = df_with_features.tail(1)[features_for_prediction]

    prediction = MODELS[symbol].predict(latest_features)
    signal_map = {1: "Kaufen", -1: "Verkaufen", 0: "Halten"}
    signal_text = signal_map.get(prediction[0], "Unbekannt")
    
    # NEU: TP/SL Berechnung
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
    """Der API-Endpunkt, den das Frontend aufruft."""
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
    """Diese Route liefert die index.html Seite aus."""
    return render_template('index.html')

load_models()