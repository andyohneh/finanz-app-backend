import os
import joblib
import pandas as pd
import numpy as np
import ta
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# .env Datei laden für die API-Schlüssel
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Globale Variablen und Konfiguration ---
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
MODELS = {} # Ein Dictionary, um unsere geladenen Modelle zu speichern
SYMBOLS = ['BTCUSD', 'XAUUSD'] # Die Symbole, die unsere App unterstützt

# --- Hilfsfunktionen für die KI-Vorhersage ---

def add_features(df):
    """
    Fügt die exakt gleichen technischen Indikatoren wie im Training hinzu.
    Diese Funktion ist eine Kopie aus unserem ki_trainer.py.
    """
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

def get_live_data_for_features(symbol, interval='1min', outputsize=100):
    """
    Holt genügend Live-Daten von der API, um die Features berechnen zu können.
    """
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol.replace('USD', '/USD')}&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'ok' and 'values' in data:
            df = pd.DataFrame(data['values'])
            # Spalten umbenennen und in richtige Datentypen konvertieren
            df = df.rename(columns={'datetime': 'timestamp'})
            df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
            # Daten umdrehen, damit die neuesten Daten am Ende stehen
            return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"Fehler beim Abrufen der Live-Daten für {symbol}: {e}")
        return None

# --- Hauptfunktionen der App ---

def load_models():
    """
    Lädt die trainierten Modelle für alle Symbole beim Start der App in den Speicher.
    """
    print("Lade KI-Modelle...")
    for symbol in SYMBOLS:
        model_filename = f"{symbol.lower()}_model.joblib"
        try:
            MODELS[symbol] = joblib.load(model_filename)
            print(f"Modell '{model_filename}' erfolgreich geladen.")
        except FileNotFoundError:
            print(f"WARNUNG: Modelldatei '{model_filename}' nicht gefunden. Für dieses Symbol können keine Vorhersagen gemacht werden.")
    print("Alle verfügbaren Modelle geladen.")

def get_prediction(symbol):
    """
    Holt Live-Daten, berechnet Features und macht eine Vorhersage mit dem geladenen Modell.
    """
    if symbol not in MODELS:
        return "Modell nicht verfügbar", 0.0 # Signal und Preis

    # 1. Genügend Live-Daten für die Feature-Berechnung holen
    live_df = get_live_data_for_features(symbol)
    if live_df is None or live_df.empty:
        return "Datenfehler", 0.0

    current_price = live_df.iloc[-1]['close'] # Der aktuellste Preis

    # 2. Features hinzufügen (genau wie im Training)
    df_with_features = add_features(live_df)
    
    if df_with_features.empty:
        return "Nicht genügend Daten für Indikatoren", current_price

    # 3. Die letzte Zeile mit den aktuellsten Features für die Vorhersage auswählen
    latest_features = df_with_features.tail(1)[['open', 'high', 'low', 'close', 'volume', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal']]

    # 4. KI-Modell um eine Vorhersage bitten
    model = MODELS[symbol]
    prediction = model.predict(latest_features) # Gibt ein Array zurück, z.B. [1]
    
    # 5. Vorhersage in ein lesbares Signal umwandeln
    signal_map = {1: "Kaufen", -1: "Verkaufen", 0: "Halten"}
    signal_text = signal_map.get(prediction[0], "Unbekannt")
    
    return signal_text, current_price

# --- Flask Routen (API Endpunkte) ---
@app.route('/data')
def get_data():
    """
    Der API-Endpunkt, den das Frontend aufruft.
    """
    assets_data = []
    for symbol in SYMBOLS:
        print(f"Verarbeite Anfrage für {symbol}...")
        signal, price = get_prediction(symbol)
        
        # Farben und Icons basierend auf dem Signal setzen
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
            "entry": f"{price:.2f}" if signal != "Halten" else "N/A", # Entry-Preis ist der aktuelle Preis bei Signal
            "takeProfit": "N/A", # TP/SL sind nicht Teil des KI-Modells
            "stopLoss": "N/A",   # Könnte eine zukünftige Erweiterung sein
            "signal": signal,
            "color": color,
            "icon": icon
        })
        
    return jsonify({"assets": assets_data})


if __name__ == '__main__':
    # Lädt die Modelle einmal beim Start
    load_models()
    # Startet die Flask-App
    app.run(debug=True, port=5000)