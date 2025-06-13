from flask import Flask, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# --- API-SCHLÜSSEL SICHER VERWALTEN ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'Rk3PgYQk5cRd3MFZggibVSygaT3t1g9GvxVukLHDC6OZToinRhGDH5UQ29YtnCgw')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'cvlpf2G8qODAQ55iJQ6ZV8BfLzhCocGg5DR14ecDEBRGKKBvffBwii7o3ZhBC2Sk')
FMP_API_KEY = os.getenv('FMP_API_KEY', 'hbWngJ2fn18YpGJqH6R2lk5vHTx7pv1j')

# Basis-URLs der APIs
BINANCE_API_BASE_URL = "https://api.binance.com/api/v3"
FMP_API_BASE_URL = "https://financialmodelingprep.com/api/v3"

# --- HILFSFUNKTIONEN ZUM ABRUFEN DER DATEN ---

def get_bitcoin_price():
    try:
        response = requests.get(f"{BINANCE_API_BASE_URL}/ticker/price?symbol=BTCUSDT")
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen des Bitcoin-Preises: {e}")
        return None
    except KeyError:
        print("Bitcoin-Preis nicht im erwarteten Format gefunden.")
        return None

def get_gold_price():
    try:
        response = requests.get(f"{FMP_API_BASE_URL}/quote/GLD?apikey={FMP_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0 and 'price' in data[0]:
            return float(data[0]['price'])
        else:
            print("Gold-Preis (GLD) nicht im erwarteten Format gefunden oder API-Antwort leer.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen des Gold-Preises von FMP: {e}")
        return None
    except KeyError:
        print("Gold-Preis (GLD) nicht im erwarteten Format gefunden.")
        return None

def get_brent_oil_price():
    try:
        response = requests.get(f"{FMP_API_BASE_URL}/quote/BZ=F?apikey={FMP_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0 and 'price' in data[0]:
            return float(data[0]['price'])
        else:
            print("Brent Oil (BZ=F) Preis nicht im erwarteten Format gefunden oder API-Antwort leer.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen des Brent Oil Preises von FMP: {e}")
        return None
    except KeyError:
        print("Brent Oil (BZ=F) Preis nicht im erwarteten Format gefunden.")
        return None

# --- UNSERE "KI"-LOGIK (VEREINFACHT FÜR DEN START) ---
def calculate_trade_levels(current_price, asset_type):
    if current_price is None:
        return None, None, None

    entry_price = round(current_price, 2)

    if asset_type == "XAUUSD":
        take_profit = round(current_price * 1.008, 2)
        stop_loss = round(current_price * 0.995, 2)
    elif asset_type == "Bitcoin (BTC)":
        take_profit = round(current_price * 1.03, 2)
        stop_loss = round(current_price * 0.98, 2)
    elif asset_type == "Brent Oil (BBL)":
        take_profit = round(current_price * 1.015, 2)
        stop_loss = round(current_price * 0.99, 2)
    else:
        take_profit = round(current_price * 1.01, 2)
        stop_loss = round(current_price * 0.99, 2)

    return entry_price, take_profit, stop_loss

# --- FLASK-ROUTEN ---

@app.route('/')
def home():
    return "Hallo von deinem Backend-Server!"

@app.route('/api/finance_data')
def get_finance_data():
    response_data = []

    # --- Bitcoin Daten ---
    btc_price = get_bitcoin_price()
    btc_entry, btc_tp, btc_sl = calculate_trade_levels(btc_price, "Bitcoin (BTC)") # Calculate even if price is None to get N/A
    response_data.append({
        "asset": "Bitcoin (BTC)",
        "currentPrice": f"{btc_price:.2f}" if btc_price is not None else "N/A", # NEU: Aktueller Preis
        "entry": f"{btc_entry:.2f}" if btc_entry is not None else "N/A",
        "takeProfit": f"{btc_tp:.2f}" if btc_tp is not None else "N/A",
        "stopLoss": f"{btc_sl:.2f}" if btc_sl is not None else "N/A"
    })

    # --- XAUUSD (Gold) Daten ---
    gold_price = get_gold_price()
    gold_entry, gold_tp, gold_sl = calculate_trade_levels(gold_price, "XAUUSD") # Calculate even if price is None
    response_data.append({
        "asset": "XAUUSD",
        "currentPrice": f"{gold_price:.2f}" if gold_price is not None else "N/A", # NEU: Aktueller Preis
        "entry": f"{gold_entry:.2f}" if gold_entry is not None else "N/A",
        "takeProfit": f"{gold_tp:.2f}" if gold_tp is not None else "N/A",
        "stopLoss": f"{gold_sl:.2f}" if gold_sl is not None else "N/A"
    })

    # --- Brent Oil Daten ---
    brent_oil_price = get_brent_oil_price()
    brent_entry, brent_tp, brent_sl = calculate_trade_levels(brent_oil_price, "Brent Oil (BBL)") # Calculate even if price is None
    response_data.append({
        "asset": "Brent Oil (BBL)",
        "currentPrice": f"{brent_oil_price:.2f}" if brent_oil_price is not None else "N/A", # NEU: Aktueller Preis
        "entry": f"{brent_entry:.2f}" if brent_entry is not None else "N/A",
        "takeProfit": f"{brent_tp:.2f}" if brent_tp is not None else "N/A",
        "stopLoss": f"{brent_sl:.2f}" if brent_sl is not None else "N/A"
    })

    return jsonify(response_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)