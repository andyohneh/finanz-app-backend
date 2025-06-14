from flask import Flask, jsonify
from flask_cors import CORS
import requests
import os
import pandas as pd # NEU: Importiere pandas
import ta.trend # NEU: Importiere ta.trend für technische Indikatoren

app = Flask(__name__)
CORS(app)

# --- API-SCHLÜSSEL SICHER VERWALTEN ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'Rk3PgYQk5cRd3MFZggibVSygaT3t1g9GvxVukLHDC6OZToinRhGDH5UQ29YtnCgw')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'cvlpf2G8qODAQ55iJQ6ZV8BfLzhCocGg5DR14ecDEBRGKKBvffBwii7o3ZhBC2Sk')
FMP_API_KEY = os.getenv('FMP_API_KEY', 'hbWngJ2fn18YpGJqH6R2lk5vHTx7pv1j')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', '2a152b2fb77743cda7c0066278e4ef37') # ERSETZE DIES!

# Basis-URLs der APIs
BINANCE_API_BASE_URL = "https://api.binance.com/api/v3"
FMP_API_BASE_URL = "https://financialmodelingprep.com/api/v3"
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

# NEU: Historische Bitcoin-Preise von Binance holen
def get_bitcoin_historical_prices(interval='1h', limit=100):
    try:
        # Endpunkt für Candlestick-Daten (Klines)
        response = requests.get(f"{BINANCE_API_BASE_URL}/klines?symbol=BTCUSDT&interval={interval}&limit={limit}")
        response.raise_for_status()
        data = response.json()
        # Binance Klines-Format: [timestamp, open, high, low, close, volume, ...]
        # Wir brauchen die 'close'-Preise (Index 4)
        close_prices = [float(kline[4]) for kline in data]
        return pd.Series(close_prices) # Wandelt Liste in Pandas Series um
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen historischer Bitcoin-Preise: {e}")
        return None

def get_gold_price():
    try:
        response = requests.get(f"{TWELVEDATA_API_BASE_URL}/price?symbol=XAU/USD&apikey={TWELVEDATA_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'price' in data:
            return float(data['price'])
        else:
            print(f"XAU/USD Preis nicht im erwarteten Format gefunden von Twelve Data: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen des XAU/USD Preises von Twelve Data: {e}")
        return None
    except Exception as e:
        print(f"Unbekannter Fehler beim Abrufen des XAU/USD Preises: {e}")
        return None

# NEU: Historische Gold-Preise von Twelve Data holen
def get_gold_historical_prices(interval='1min', outputsize=100):
    try:
        # Twelve Data Endpunkt für historische Kurse (Time Series)
        # Für XAU/USD kann die Intervalleinschränkung im Free Tier anders sein.
        # '1min' ist oft für Echtzeitdaten, '1day' für längere Historie.
        response = requests.get(f"{TWELVEDATA_API_BASE_URL}/time_series?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'values' in data and data['values']:
            close_prices = [float(entry['close']) for entry in data['values']]
            # Die neuesten Daten sind am Anfang der Liste, wir wollen sie chronologisch
            return pd.Series(close_prices[::-1]) # Umkehren für chronologische Reihenfolge
        else:
            print(f"Historische XAU/USD Preise nicht im erwarteten Format gefunden von Twelve Data: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen historischer XAU/USD Preise von Twelve Data: {e}")
        return None
    except Exception as e:
        print(f"Unbekannter Fehler beim Abrufen historischer XAU/USD Preise: {e}")
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

# NEU: Historische Brent Oil Preise von FMP holen
def get_brent_oil_historical_prices(limit=100):
    try:
        # FMP Historical Prices for Commodities: /historical-price/{symbol}?apikey=...
        response = requests.get(f"{FMP_API_BASE_URL}/historical-price/BZ=F?apikey={FMP_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'historical' in data and data['historical']:
            # Die neuesten Daten sind am Anfang der Liste, wir brauchen die 'close'-Preise
            # Beschränken auf die 'limit' neuesten Preise
            close_prices = [float(entry['close']) for entry in data['historical'][:limit]]
            return pd.Series(close_prices[::-1]) # Umkehren für chronologische Reihenfolge
        else:
            print(f"Historische Brent Oil Preise nicht im erwarteten Format gefunden von FMP: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen historischer Brent Oil Preise von FMP: {e}")
        return None
    except Exception as e:
        print(f"Unbekannter Fehler beim Abrufen historischer Brent Oil Preise: {e}")
        return None


# --- UNSERE "KI"-LOGIK (MIT SMA) ---
def calculate_trade_levels(current_price, historical_prices, asset_type):
    if current_price is None:
        return None, None, None

    entry_price = round(current_price, 2)

    # Berechne den SMA, wenn historische Daten verfügbar sind
    sma_period = 20 # Beispiel: SMA über die letzten 20 Datenpunkte (z.B. Stunden oder Tage)
    sma_value = None
    if historical_prices is not None and len(historical_prices) >= sma_period:
        # Füge den aktuellen Preis zu den historischen Preisen hinzu, um den neuesten SMA zu berechnen
        # (Dies ist wichtig, da der aktuelle Preis oft noch nicht in den "historischen" Klines enthalten ist)
        all_prices = pd.concat([historical_prices, pd.Series([current_price])])
        sma_value = ta.trend.sma_indicator(all_prices, window=sma_period).iloc[-1]
        print(f"Calculated SMA for {asset_type} ({sma_period} period): {sma_value:.2f}")


    # Beispiel-Logik: Buy/Sell-Signal basierend auf SMA und Prozentsätzen
    # Dies ist eine sehr vereinfachte Strategie!
    take_profit_factor = 1.01
    stop_loss_factor = 0.99

    if asset_type == "XAUUSD":
        take_profit_factor = 1.005 # 0.5%
        stop_loss_factor = 0.998  # 0.2%
        if sma_value is not None:
            if current_price > sma_value:
                print(f"XAUUSD: Current price ({current_price:.2f}) > SMA ({sma_value:.2f}) - Bullish bias, maybe higher TP.")
                take_profit_factor = 1.007 # Etwas höherer TP bei Aufwärtstrend
                stop_loss_factor = 0.997 # Etwas engere SL bei Aufwärtstrend
            else:
                print(f"XAUUSD: Current price ({current_price:.2f}) < SMA ({sma_value:.2f}) - Bearish bias.")
                # Hier könnte man auch ein Verkaufssignal oder ein engeres TP/SL für Short-Positionen machen
                pass # Für jetzt bleiben die Faktoren gleich

    elif asset_type == "Bitcoin (BTC)":
        take_profit_factor = 1.03 # 3%
        stop_loss_factor = 0.98  # 2%
        if sma_value is not None:
            if current_price > sma_value:
                print(f"Bitcoin: Current price ({current_price:.2f}) > SMA ({sma_value:.2f}) - Bullish bias, higher TP.")
                take_profit_factor = 1.04 # 4% TP
                stop_loss_factor = 0.975 # 2.5% SL
            else:
                print(f"Bitcoin: Current price ({current_price:.2f}) < SMA ({sma_value:.2f}) - Bearish bias.")
                pass

    elif asset_type == "Brent Oil (BBL)":
        take_profit_factor = 1.015 # 1.5%
        stop_loss_factor = 0.99  # 1%
        if sma_value is not None:
            if current_price > sma_value:
                print(f"Brent Oil: Current price ({current_price:.2f}) > SMA ({sma_value:.2f}) - Bullish bias.")
                take_profit_factor = 1.02 # 2% TP
            else:
                print(f"Brent Oil: Current price ({current_price:.2f}) < SMA ({sma_value:.2f}) - Bearish bias.")
                pass

    take_profit = round(current_price * take_profit_factor, 2)
    stop_loss = round(current_price * stop_loss_factor, 2)

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
    btc_historical_prices = get_bitcoin_historical_prices()
    btc_entry, btc_tp, btc_sl = calculate_trade_levels(btc_price, btc_historical_prices, "Bitcoin (BTC)")
    response_data.append({
        "asset": "Bitcoin (BTC)",
        "currentPrice": f"{btc_price:.2f}" if btc_price is not None else "N/A",
        "entry": f"{btc_entry:.2f}" if btc_entry is not None else "N/A",
        "takeProfit": f"{btc_tp:.2f}" if btc_tp is not None else "N/A",
        "stopLoss": f"{btc_sl:.2f}" if btc_sl is not None else "N/A"
    })

    # --- XAUUSD (Gold) Daten ---
    gold_price = get_gold_price()
    gold_historical_prices = get_gold_historical_prices()
    gold_entry, gold_tp, gold_sl = calculate_trade_levels(gold_price, gold_historical_prices, "XAUUSD")
    response_data.append({
        "asset": "XAUUSD",
        "currentPrice": f"{gold_price:.2f}" if gold_price is not None else "N/A",
        "entry": f"{gold_entry:.2f}" if gold_entry is not None else "N/A",
        "takeProfit": f"{gold_tp:.2f}" if gold_tp is not None else "N/A",
        "stopLoss": f"{gold_sl:.2f}" if gold_sl is not None else "N/A"
    })

    # --- Brent Oil Daten ---
    brent_oil_price = get_brent_oil_price()
    brent_oil_historical_prices = get_brent_oil_historical_prices()
    brent_entry, brent_tp, brent_sl = calculate_trade_levels(brent_oil_price, brent_oil_historical_prices, "Brent Oil (BBL)")
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