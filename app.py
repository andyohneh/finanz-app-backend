# app.py (Finale Version)
import os
import joblib
import pandas as pd
import ta
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# .env Datei laden für die API-Schlüssel (wird für lokale Tests verwendet)
load_dotenv()

# Flask App initialisieren und den 'templates' Ordner angeben
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Globale Variablen und Konfiguration ---
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
MODELS = {} # Ein Dictionary, um unsere geladenen Modelle zu speichern
SYMBOLS = ['BTCUSD', 'XAUUSD'] # Die Symbole, die unsere App unterstützt

# --- Hilfsfunktionen für die KI-Vorhersage ---

def add_features(df):
    """Fügt technische Indikatoren als neue Spalten zum DataFrame hinzu."""
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

def get_live_data_for_features(symbol, interval='1min', outputsize=75):
    """Holt genügend Live-Daten von der API, um die Features berechnen zu können."""
    try:
        api_symbol = symbol.replace('USD', '/USD')
        url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'ok' and 'values' in data:
            df = pd.DataFrame(data['values'])
            df = df.rename(columns={'datetime': 'timestamp'})
            
            # --- HIER IST DIE KORREKTUR ---
            # Wir prüfen, ob 'volume' fehlt, und fügen es als 0 hinzu wenn nötig.
            if 'volume' not in df.columns:
                df['volume'] = 0.0
            # --------------------------------

            # Jetzt können wir die Typen sicher konvertieren
            df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"Fehler beim Abrufen der Live-Daten für {symbol}: {e}")
    return None

def load_models():
    """Lädt die trainierten Modelle für alle Symbole beim Start der App in den Speicher."""
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
    """Holt Live-Daten, berechnet Features und macht eine Vorhersage mit dem geladenen Modell."""
    if symbol not in MODELS:
        return "Modell nicht verfügbar", 0.0

    live_df = get_live_data_for_features(symbol)
    if live_df is None or live_df.empty:
        return "Datenfehler", 0.0

    current_price = live_df.iloc[-1]['close']
    df_with_features = add_features(live_df)
    
    if df_with_features.empty:
        return "Nicht genügend Daten für Indikatoren", current_price

    features_for_prediction = ['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal']
    latest_features = df_with_features.tail(1)[features_for_prediction]

    model = MODELS[symbol]
    prediction = model.predict(latest_features)
    
    signal_map = {1: "Kaufen", -1: "Verkaufen", 0: "Halten"}
    signal_text = signal_map.get(prediction[0], "Unbekannt")
    
    return signal_text, current_price

# --- Flask Routen (API Endpunkte) ---

@app.route('/data')
def get_data():
    """Der API-Endpunkt, den das Frontend aufruft."""
    assets_data = []
    for symbol in SYMBOLS:
        signal, price = get_prediction(symbol)
        
        color = "grey"
        icon = "minus-circle"
        if signal == "Kaufen":
            color = "green"
            icon = "arrow-up"
        elif signal == "Verkaufen":
            color = "red"
            icon = "arrow-down"
            
        assets_data.append({
            "asset": symbol,
            "currentPrice": f"{price:.2f}" if price else "N/A",
            "entry": f"{price:.2f}" if signal not in ["Halten", "Datenfehler", "Modell nicht verfügbar", "Nicht genügend Daten für Indikatoren"] else "N/A",
            "takeProfit": "N/A",
            "stopLoss": "N/A",
            "signal": signal,
            "color": color,
            "icon": icon
        })
    return jsonify({"assets": assets_data})

@app.route('/')
def index():
    """Diese Route liefert die index.html Seite aus."""
    return render_template('index.html')

# --- App Start ---

# Dieser Block wird nur ausgeführt, wenn das Skript direkt gestartet wird (z.B. lokal)
if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)

# Lade die Modelle auch, wenn Gunicorn die App startet
load_models()