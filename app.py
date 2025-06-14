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

# NEU: Twelve Data API Key
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', 'DEIN_TWELVEDATA_API_KEY') # ERSETZE DIES!

# Basis-URLs der APIs
BINANCE_API_BASE_URL = "https://api.binance.com/api/v3"
FMP_API_BASE_URL = "https://financialmodelingprep.com/api/v3"
# NEU: Twelve Data Basis-URL
TWELVEDATA_API_BASE_URL = "https://api.twelvedata.com"


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

# AKTUELLE ÄNDERUNG: Nutzt Twelve Data für XAUUSD
def get_gold_price():
    try:
        # Twelve Data Endpunkt für Währungspaare / Rohstoffe
        response = requests.get(f"{TWELVEDATA_API_BASE_URL}/price?symbol=XAU/USD&apikey={TWELVEDATA_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'price' in data: # Twelve Data gibt direkt ein Dictionary mit 'price' zurück
            return float(data['price'])
        else:
            print(f"XAU/USD Preis nicht im erwarteten Format gefunden von Twelve Data: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen des XAU/USD Preises von Twelve Data: {e}")
        return None
    except Exception as e: # Catch all other potential errors like JSON decoding
        print(f"Unbekannter Fehler beim Abrufen des XAU/USD Preises: {e}")
        return None

def get_brent_oil_price():
    # Wir bleiben bei FMP für Brent Oil, da das zu funktionieren scheint.
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
        # Für Gold verwenden wir jetzt typische 2 Nachkommastellen und engere Spreads
        take_profit = round(current_price * 1.005, 2) # z.B. 0.5% über Einstieg
        stop_loss = round(current_price * 0.998, 2)  # z.B. 0.2% unter Einstieg
    elif asset_type == "Bitcoin (BTC)":
        take_profit = round(current_price * 1.03, 2)
        stop_loss = round(current_price * 0.98, 2)
    elif asset_type == "Brent Oil (BBL)":
        take_profit = round(current_price * 1.015, 2)
        stop_loss = round(current_price * 0.99, 2)
    else:
        # Fallback für unbekannte Assets
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
    btc_entry, btc_tp, btc_sl = calculate_trade_levels(btc_price, "Bitcoin (BTC)")
    response_data.append({
        "asset": "Bitcoin (BTC)",
        "currentPrice": f"{btc_price:.2f}" if btc_price is not None else "N/A",
        "entry": f"{btc_entry:.2f}" if btc_entry is not None else "N/A",
        "takeProfit": f"{btc_tp:.2f}" if btc_tp is not None else "N/A",
        "stopLoss": f"{btc_sl:.2f}" if btc_sl is not None else "N/A"
    })

    # --- XAUUSD (Gold) Daten ---
    gold_price = get_gold_price() # Dies nutzt jetzt Twelve Data!
    gold_entry, gold_tp, gold_sl = calculate_trade_levels(gold_price, "XAUUSD")
    response_data.append({
        "asset": "XAUUSD",
        "currentPrice": f"{gold_price:.2f}" if gold_price is not None else "N/A",
        "entry": f"{gold_entry:.2f}" if gold_entry is not None else "N/A",
        "takeProfit": f"{gold_tp:.2f}" if gold_tp is not None else "N/A",
        "stopLoss": f"{gold_sl:.2f}" if gold_sl is not None else "N/A"
    })

    # --- Brent Oil Daten ---
    brent_oil_price = get_brent_oil_price()
    brent_entry, brent_tp, brent_sl = calculate_trade_levels(brent_oil_price, "Brent Oil (BBL)")
    response_data.append({
        "asset": "Brent Oil (BBL)",
        "currentPrice": f"{brent_oil_price:.2f}" if brent_oil_price is not None else "N/A",
        "entry": f"{brent_entry:.2f}" if brent_entry is not None else "N/A",
        "takeProfit": f"{brent_tp:.2f}" if brent_tp is not None else "N/A",
        "stopLoss": f"{brent_sl:.2f}" if brent_sl is not None else "N/A"
    })

    return jsonify(response_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)