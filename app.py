from flask import Flask, jsonify
from flask_cors import CORS
import requests
import os
import pandas as pd
import ta.trend # Importiere ta.trend für technische Indikatoren
import ta.momentum # NEU: Importiere ta.momentum für RSI etc. (vorbereitet)

app = Flask(__name__)
CORS(app)

# --- API-SCHLÜSSEL SICHER VERWALTEN ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'Rk3PgYQk5cRd3MFZggibVSygaT3t1g9GvxVukLHDC6OZToinRhGDH5UQ29YtnCgw')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'cvlpf2G8qODAQ55iJQ6ZV8BfLzhCocGg5DR14ecDEBRGKKBvffBwii7o3ZhBC2Sk')
FMP_API_KEY = os.getenv('FMP_API_KEY', 'hbWngJ2fn18YpGJqH6R2lk5vHTx7pv1j')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', '2a152b2fb77743cda7c0066278e4ef37') # Dein Key ist jetzt hier eingetragen

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

def get_bitcoin_historical_prices(interval='1h', limit=100):
    try:
        response = requests.get(f"{BINANCE_API_BASE_URL}/klines?symbol=BTCUSDT&interval={interval}&limit={limit}")
        response.raise_for_status()
        data = response.json()
        close_prices = [float(kline[4]) for kline in data]
        return pd.Series(close_prices)
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

def get_gold_historical_prices(interval='1min', outputsize=100):
    try:
        response = requests.get(f"{TWELVEDATA_API_BASE_URL}/time_series?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'values' in data and data['values']:
            close_prices = [float(entry['close']) for entry in data['values']]
            return pd.Series(close_prices[::-1])
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

def get_brent_oil_historical_prices(limit=100):
    try:
        response = requests.get(f"{FMP_API_BASE_URL}/historical-price/BZ=F?apikey={FMP_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'historical' in data and data['historical']:
            close_prices = [float(entry['close']) for entry in data['historical'][:limit]]
            return pd.Series(close_prices[::-1])
        else:
            print(f"Historische Brent Oil Preise nicht im erwarteten Format gefunden von FMP: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen historischer Brent Oil Preise von FMP: {e}")
        return None
    except Exception as e:
        print(f"Unbekannter Fehler beim Abrufen historischer Brent Oil Preise: {e}")
        return None


# --- UNSERE "KI"-LOGIK (MIT SMA Crossover) ---
def calculate_trade_levels(current_price, historical_prices, asset_type):
    if current_price is None:
        return None, None, None

    entry_price = round(current_price, 2)

    # Definiere Perioden für schnelle und langsame SMAs
    # Typische Swing Trading SMAs könnten 10/20, 20/50 oder 50/100 sein
    # Wir nehmen hier 10 und 20 für ein schnelleres Signal
    fast_sma_period = 10
    slow_sma_period = 20

    fast_sma_value = None
    slow_sma_value = None
    trade_signal = "NEUTRAL" # Kann "BUY", "SELL", "HOLD" sein

    if historical_prices is not None and len(historical_prices) >= slow_sma_period:
        # Füge den aktuellen Preis zu den historischen Preisen hinzu für die Berechnung
        all_prices = pd.concat([historical_prices, pd.Series([current_price])]).reset_index(drop=True)

        # Berechne den schnellen SMA
        if len(all_prices) >= fast_sma_period:
            fast_sma_value = ta.trend.sma_indicator(all_prices, window=fast_sma_period).iloc[-1]
            # print(f"{asset_type}: Fast SMA ({fast_sma_period} period): {fast_sma_value:.2f}")

        # Berechne den langsamen SMA
        if len(all_prices) >= slow_sma_period:
            slow_sma_value = ta.trend.sma_indicator(all_prices, window=slow_sma_period).iloc[-1]
            # print(f"{asset_type}: Slow SMA ({slow_sma_period} period): {slow_sma_value:.2f}")

        # Swing Trading Logik: SMA Crossover
        if fast_sma_value is not None and slow_sma_value is not None:
            # Überprüfe den letzten und vorletzten SMA-Wert, um ein echtes Kreuz zu erkennen
            # Benötigt mindestens 2 Perioden mehr als der längste SMA für präzise Crossover-Erkennung
            if len(all_prices) >= slow_sma_period + 1: # Brauchen mindestens einen vorherigen Punkt
                fast_sma_prev = ta.trend.sma_indicator(all_prices, window=fast_sma_period).iloc[-2]
                slow_sma_prev = ta.trend.sma_indicator(all_prices, window=slow_sma_period).iloc[-2]

                # Golden Cross (Kaufsignal): Schneller SMA kreuzt langsamen von unten nach oben
                if fast_sma_prev < slow_sma_prev and fast_sma_value >= slow_sma_value:
                    trade_signal = "BUY"
                    print(f"!!! {asset_type}: BUY Signal (Golden Cross) !!!")
                # Death Cross (Verkaufssignal): Schneller SMA kreuzt langsamen von oben nach unten
                elif fast_sma_prev > slow_sma_prev and fast_sma_value <= slow_sma_value:
                    trade_signal = "SELL"
                    print(f"!!! {asset_type}: SELL Signal (Death Cross) !!!")
                else:
                    trade_signal = "HOLD" # Kein Kreuz, Trend fortgesetzt
                    if current_price > slow_sma_value:
                        trade_signal = "HOLD_BULLISH" # Weiterhin über langem SMA
                    else:
                        trade_signal = "HOLD_BEARISH" # Weiterhin unter langem SMA
            else: # Nicht genug Daten für präzise Crossover-Erkennung, aber Trend kann angezeigt werden
                if fast_sma_value > slow_sma_value:
                    trade_signal = "HOLD_BULLISH"
                elif fast_sma_value < slow_sma_value:
                    trade_signal = "HOLD_BEARISH"

    # Anpassung der Take Profit / Stop Loss Faktoren basierend auf dem Signal
    take_profit_factor = 1.01
    stop_loss_factor = 0.99

    if asset_type == "XAUUSD":
        take_profit_factor = 1.005
        stop_loss_factor = 0.998
        if trade_signal == "BUY":
            take_profit_factor = 1.007 # Etwas aggressiver bei Kauf
            stop_loss_factor = 0.997  # Etwas engerer SL
        elif trade_signal == "SELL":
            take_profit_factor = 0.995 # Gewinn bei Short-Position
            stop_loss_factor = 1.002  # Verlust bei Short-Position (umgekehrt)
        elif trade_signal == "HOLD_BULLISH":
             take_profit_factor = 1.006
        elif trade_signal == "HOLD_BEARISH":
            stop_loss_factor = 0.997

    elif asset_type == "Bitcoin (BTC)":
        take_profit_factor = 1.03
        stop_loss_factor = 0.98
        if trade_signal == "BUY":
            take_profit_factor = 1.04
            stop_loss_factor = 0.975
        elif trade_signal == "SELL":
            take_profit_factor = 0.97 # Gewinn bei Short-Position
            stop_loss_factor = 1.025 # Verlust bei Short-Position (umgekehrt)
        elif trade_signal == "HOLD_BULLISH":
            take_profit_factor = 1.035
        elif trade_signal == "HOLD_BEARISH":
            stop_loss_factor = 0.985

    elif asset_type == "Brent Oil (BBL)":
        take_profit_factor = 1.015
        stop_loss_factor = 0.99
        if trade_signal == "BUY":
            take_profit_factor = 1.02
            stop_loss_factor = 0.985
        elif trade_signal == "SELL":
            take_profit_factor = 0.98 # Gewinn bei Short-Position
            stop_loss_factor = 1.015 # Verlust bei Short-Position (umgekehrt)
        elif trade_signal == "HOLD_BULLISH":
            take_profit_factor = 1.017
        elif trade_signal == "HOLD_BEARISH":
            stop_loss_factor = 0.995

    # Wenn ein SELL-Signal, können wir Take Profit und Stop Loss umkehren, um eine Short-Position darzustellen
    if trade_signal == "SELL":
        calculated_tp = round(current_price * take_profit_factor, 2)
        calculated_sl = round(current_price * stop_loss_factor, 2)
        # Für Short-Positionen ist TP unter Einstieg, SL über Einstieg
        # Wir haben die Faktoren bereits umgekehrt, sodass tp < entry und sl > entry ist
        # Keine Umkehrung der Faktoren, sondern der Logik im Return
        return entry_price, calculated_tp, calculated_sl
    else: # BUY, HOLD, oder andere Signale (long position)
        calculated_tp = round(current_price * take_profit_factor, 2)
        calculated_sl = round(current_price * stop_loss_factor, 2)
        return entry_price, calculated_tp, calculated_sl

# --- FLASK-ROUTEN ---

@app.route('/')
def home():
    return "Hallo von deinem Backend-Server!"

@app.route('/api/finance_data')
def get_finance_data():
    response_data = []

    # --- Bitcoin Daten ---
    btc_price = get_bitcoin_price()
    # Hinweis: Binance 'klines' für BTCUSDT sind sehr zuverlässig.
    btc_historical_prices = get_bitcoin_historical_prices(interval='1h', limit=50) # Genug für 20-Perioden SMA
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
    # Hinweis: Twelve Data Free Tier hat oft Einschränkungen bei historischen Daten (outputsize/interval)
    # Für Swing Trading wären Tages- oder 4-Stunden-Intervalle besser, aber die sind im Free Tier begrenzt.
    # Wir nutzen hier 1-Minuten-Daten für kurzfristige Tests.
    gold_historical_prices = get_gold_historical_prices(interval='1min', outputsize=50) # Genug für 20-Perioden SMA
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
    # FMP Historical Prices sind oft Tagesdaten, gut für längere SMAs
    brent_oil_historical_prices = get_brent_oil_historical_prices(limit=50) # Genug für 20-Perioden SMA
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