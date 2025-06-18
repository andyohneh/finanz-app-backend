# app.py (Version mit kugelsicherem Laden der Modelle)
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
    Muss immer synchron mit ki_trainer.py sein!
    """
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high_indicator'] = bb.bollinger_hband_indicator()
    df['bb_low_indicator'] = bb.bollinger_lband_indicator()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=20)
    return df

def load_models():
    """Lädt die trainierten Modelle für alle Symbole mit einem robusten, absoluten Pfad."""
    global MODELS
    print("Lade KI-Modelle...")

    # --- KORREKTUR HIER: Robuste Pfad-Erstellung ---
    # Finde den absoluten Pfad des Ordners, in dem diese app.py Datei liegt.
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # Der 'models'-Ordner befindet sich im selben Verzeichnis.
    model_dir = os.path.join(base_dir, 'models')
    print(f"Suche Modelle im absoluten Pfad: {model_dir}") # Nützliche Debug-Ausgabe
    # --- ENDE KORREKTUR ---

    for symbol_api in SYMBOLS:
        db_symbol_filename = f"{symbol_api[:-3]}_{symbol_api[-3:]}"
        # Dieser Pfad ist jetzt absolut und unmissverständlich
        model_path = os.path.join(model_dir, f'model_{db_symbol_filename}.joblib')
        try:
            MODELS[symbol_api] = joblib.load(model_path)
            print(f"-> Modell für {symbol_api} erfolgreich geladen von '{model_path}'.")
        except FileNotFoundError:
            print(f"WARNUNG: Modelldatei nicht gefunden für {symbol_api} unter '{model_path}'.")
        except Exception as e:
            print(f"Fehler beim Laden des Modells für {symbol_api}: {e}")

def get_prediction(symbol: str):
    """Holt Live-Daten, berechnet Features und gibt eine Vorhersage zurück."""
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    url = f"https://api.twelvedata.com/time_series?symbol={db_symbol}&interval=1min&outputsize=100&apikey={TWELVEDATA_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok':
            print(f"API-Fehler für {symbol}: {data.get('message')}")
            return "API Fehler", None, None, None
            
        df = pd.DataFrame(data['values'])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        
        df_features = add_features(df)
        latest_features = df_features.iloc[[-1]]
        
        price = latest_features['close'].iloc[0]
        atr = latest_features['atr'].iloc[0]
        
        # 'datetime' ist kein Feature für das Modell
        model_features_cols = [col for col in df_features.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'atr']]
        X_live = latest_features[model_features_cols]
        
        model = MODELS.get(symbol)
        if model:
            signal = model.predict(X_live)[0]
            pt_sl_ratio = 2.0 if symbol == 'BTCUSD' else 1.5
            
            take_profit, stop_loss = None, None
            if signal == 'Kaufen':
                take_profit = price + (atr * pt_sl_ratio)
                stop_loss = price - atr
            elif signal == 'Verkaufen':
                take_profit = price - (atr * pt_sl_ratio)
                stop_loss = price + atr
                
            return signal, price, take_profit, stop_loss
        else:
            return "Kein Modell", None, None, None

    except Exception as e:
        print(f"Schwerwiegender Fehler bei der Vorhersage für {symbol}: {e}")
        return "Systemfehler", None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-financial-data')
def get_financial_data():
    assets_data = []
    for symbol in SYMBOLS:
        signal, price, take_profit, stop_loss = get_prediction(symbol)
        color, icon = "grey", "minus"
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
    query = text("SELECT timestamp, close FROM historical_data WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 200")
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
        result.reverse()
        labels = [row[0].strftime('%H:%M') for row in result]
        data_points = [row[1] for row in result]
        return jsonify({"labels": labels, "data": data_points})
    except Exception as e:
        print(f"Fehler beim Abrufen der Verlaufsdaten für {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5001)