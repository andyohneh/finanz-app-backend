import os
import joblib
import pandas as pd
import ta
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from database import engine

load_dotenv()
app = Flask(__name__, template_folder='templates')
CORS(app)

TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
MODELS = {}
SYMBOLS = ['BTCUSD', 'XAUUSD']

def add_features(df):
    """
    Fügt technische Indikatoren hinzu.
    WICHTIG: Diese Funktion muss exakt identisch sein mit der in ki_trainer.py!
    """
    print("Füge Features zu Live-Daten hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # ### NEU: Die Bollinger Bänder, die dem Trainer bekannt sind ###
    indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df['bb_high_band'] = indicator_bb.bollinger_hband()
    df['bb_low_band'] = indicator_bb.bollinger_lband()
    df['bb_pband'] = indicator_bb.bollinger_pband()
    df['bb_wband'] = indicator_bb.bollinger_wband()
    # ### ENDE NEUER CODEBLOCK ###

    df.dropna(inplace=True)
    return df

def get_live_data_for_features(symbol, interval='1min', outputsize=100):
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    url = f"https://api.twelvedata.com/time_series?symbol={db_symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'values' not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data['values'])
        df = df.iloc[::-1].reset_index(drop=True)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except requests.exceptions.RequestException:
        return pd.DataFrame()

def load_models():
    print("Lade KI-Modelle...")
    for symbol in SYMBOLS:
        model_path = f'models/{symbol}_model.joblib'
        if os.path.exists(model_path):
            try:
                MODELS[symbol] = joblib.load(model_path)
                print(f"-> Modell für {symbol} erfolgreich geladen.")
            except Exception as e:
                print(f"Fehler beim Laden des Modells für {symbol}: {e}")
        else:
            print(f"WARNUNG: Keine Modelldatei für {symbol} unter {model_path} gefunden.")
    print("Modelle geladen.")

@app.before_request
def before_first_request_func():
    if not MODELS:
        load_models()

@app.route('/')
def index():
    return render_template('index.html')

def get_prediction(symbol):
    df = get_live_data_for_features(symbol)
    if df.empty or symbol not in MODELS:
        return "Datenfehler", 0, 0, 0
    df_with_features = add_features(df)
    if df_with_features.empty:
        return "Berechnungsfehler", 0, 0, 0
    
    latest_features_row = df_with_features.tail(1)
    current_price = latest_features_row['close'].iloc[0]
    current_atr = latest_features_row['atr'].iloc[0]
    
    features = MODELS[symbol].feature_names_in_
    latest_features = latest_features_row[features]
    
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
    return jsonify(assets_data)

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    query = text("""
        SELECT timestamp, close FROM historical_data 
        WHERE symbol = :symbol_param
        ORDER BY timestamp DESC 
        LIMIT 200
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
        result.reverse()
        labels = [row[0].strftime('%H:%M') for row in result]
        data_points = [row[1] for row in result]
        return jsonify({"labels": labels, "data": data_points})
    except Exception as e:
        print(f"Fehler beim Holen der historischen Daten für {symbol}: {e}")
        return jsonify({"error": "Konnte historische Daten nicht laden."}), 500

if __name__ == '__main__':
    # Diese Zeilen werden von Gunicorn ignoriert, sind aber für lokales Testen nützlich
    app.run(debug=True, port=5001)