from flask import Flask, jsonify
from flask_cors import CORS
import requests
import os
import pandas as pd
import ta.trend
import ta.momentum

app = Flask(__name__)
CORS(app)

# --- API-SCHLÜSSEL SICHER VERWALTEN ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'Rk3PgYQk5cRd3MFZggibVSygaT3t1g9GvxVukLHDC6OZToinRhGDH5UQ29YtnCgw')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'cvlpf2G8qODAQ55iJQ6ZV8BfLzhCocGg5DR14ecDEBRGKKBvffBwii7o3ZhBC2Sk')
FMP_API_KEY = os.getenv('FMP_API_KEY', 'hbWngJ2fn18YpGJqH6R2lk5vHTx7pv1j')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', '2a152b2fb77743cda7c0066278e4ef37')

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

def get_bitcoin_historical_prices(interval='1h', limit=150):
    try:
        response = requests.get(f"{BINANCE_API_BASE_URL}/klines?symbol=BTCUSDT&interval={interval}&limit={limit}")
        response.raise_for_status()
        data = response.json()
        close_prices = [float(kline[4]) for kline in data]
        return pd.Series(close_prices).iloc[::-1].reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen historischer Bitcoin-Preise: {e}")
        return None
    except Exception as e:
        print(f"Unbekannter Fehler beim Abrufen historischer Bitcoin-Preise: {e}")
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

def get_gold_historical_prices(interval='1min', outputsize=150):
    try:
        response = requests.get(f"{TWELVEDATA_API_BASE_URL}/time_series?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'values' in data and data['values']:
            close_prices = [float(entry['close']) for entry in data['values']]
            return pd.Series(close_prices).iloc[::-1].reset_index(drop=True)
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
    # Aktuellen Brent-Preis weiterhin von FMP holen, da das funktioniert hat.
    try:
        response = requests.get(f"{FMP_API_BASE_URL}/quote/BZ=F?apikey={FMP_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0 and 'price' in data[0]:
            return float(data[0]['price'])
        else:
            print("Brent Oil (BZ=F) Preis nicht im erwarteten Format gefunden oder API-Antwort leer von FMP.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen des Brent Oil Preises von FMP: {e}")
        return None
    except KeyError:
        print("Brent Oil (BZ=F) Preis nicht im erwarteten Format gefunden von FMP.")
        return None

# AKTUELLE ÄNDERUNG: Nutzt Twelve Data für historische Brent Oil Preise
# Beachte: BRENT Symbol ist NICHT im Free Tier von Twelve Data. Wir fangen Fehler ab.
def get_brent_oil_historical_prices(interval='1min', outputsize=100):
    try:
        # Twelve Data Endpunkt für historische Rohstoffkurse
        # Symbol 'BRENT' ist im Free Tier nicht verfügbar. Erwarte Fehler.
        response = requests.get(f"{TWELVEDATA_API_BASE_URL}/time_series?symbol=BRENT&interval={interval}&outputsize={outputsize}&apikey={TWELVEDATA_API_KEY}")
        response.raise_for_status()
        data = response.json()
        if 'values' in data and data['values']:
            close_prices = [float(entry['close']) for entry in data['values']]
            return pd.Series(close_prices).iloc[::-1].reset_index(drop=True)
        else:
            print(f"Historische Brent Oil Preise (BRENT) nicht im erwarteten Format gefunden von Twelve Data: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen historischer Brent Oil Preise von Twelve Data: {e}")
        return None
    except Exception as e:
        print(f"Unbekannter Fehler beim Abrufen historischer Brent Oil Preise von Twelve Data: {e}")
        return None

# --- UNSERE "KI"-LOGIK (MIT SMA Crossover, RSI & MACD) ---
def calculate_trade_levels(current_price, historical_prices, asset_type):
    if current_price is None:
        return None, None, None

    entry_price = round(current_price, 2)

    # Standard-Faktoren (Fallback)
    take_profit_factor = 1.01
    stop_loss_factor = 0.99

    # Spezifische Standard-Faktoren für jedes Asset, wenn keine Indikatoren berechnet werden können
    if asset_type == "XAUUSD":
        take_profit_factor = 1.005
        stop_loss_factor = 0.998
    elif asset_type == "Bitcoin (BTC)":
        take_profit_factor = 1.03
        stop_loss_factor = 0.98
    elif asset_type == "Brent Oil (BBL)":
        take_profit_factor = 1.015
        stop_loss_factor = 0.99

    # Indikator-Parameter
    fast_sma_period = 10
    slow_sma_period = 20
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    macd_fast_period = 12
    macd_slow_period = 26
    macd_signal_period = 9

    trade_signal = "NEUTRAL"
    rsi_value = None
    macd_line = None
    macd_signal_line = None

    # NEUE LOGIK: Nur berechnen, wenn genügend historische Daten vorhanden sind
    # Und nur für Assets, die keine Probleme mit API-Limits haben (aktuell BTC und XAUUSD)
    if historical_prices is not None and asset_type in ["Bitcoin (BTC)", "XAUUSD"]: # Nur für diese Assets Indikatoren berechnen
        all_prices = pd.concat([historical_prices, pd.Series([current_price])]).reset_index(drop=True)

        # Berechne SMAs
        if len(all_prices) >= slow_sma_period:
            fast_sma_value = ta.trend.sma_indicator(all_prices, window=fast_sma_period).iloc[-1]
            slow_sma_value = ta.trend.sma_indicator(all_prices, window=slow_sma_period).iloc[-1]

            if len(all_prices) >= slow_sma_period + 1:
                fast_sma_prev = ta.trend.sma_indicator(all_prices, window=fast_sma_period).iloc[-2]
                slow_sma_prev = ta.trend.sma_indicator(all_prices, window=slow_sma_period).iloc[-2]

                if fast_sma_prev < slow_sma_prev and fast_sma_value >= slow_sma_value:
                    trade_signal = "BUY"
                    print(f"!!! {asset_type}: BUY Signal (Golden Cross) !!!")
                elif fast_sma_prev > slow_sma_prev and fast_sma_value <= slow_sma_value:
                    trade_signal = "SELL"
                    print(f"!!! {asset_type}: SELL Signal (Death Cross) !!!")
                else:
                    trade_signal = "HOLD"
                    if current_price > slow_sma_value:
                        trade_signal = "HOLD_BULLISH"
                    else:
                        trade_signal = "HOLD_BEARISH"
            else:
                if fast_sma_value > slow_sma_value:
                    trade_signal = "HOLD_BULLISH"
                elif fast_sma_value < slow_sma_value:
                    trade_signal = "HOLD_BEARISH"

        # Berechne RSI
        if len(all_prices) >= rsi_period:
            rsi_value = ta.momentum.rsi(all_prices, window=rsi_period).iloc[-1]
            print(f"{asset_type}: RSI ({rsi_period} period): {rsi_value:.2f}")

        # Berechne MACD
        if len(all_prices) >= macd_slow_period:
            macd_series = ta.trend.macd(all_prices, window_fast=macd_fast_period, window_slow=macd_slow_period)
            macd_line = macd_series.iloc[-1]
            macd_signal_line = ta.trend.macd_signal(all_prices, window_fast=macd_fast_period, window_slow=macd_slow_period, window_sign=macd_signal_period).iloc[-1]
            macd_histogram = ta.trend.macd_diff(all_prices, window_fast=macd_fast_period, window_slow=macd_slow_period, window_sign=macd_signal_period).iloc[-1]
            print(f"{asset_type}: MACD Line: {macd_line:.2f}, Signal Line: {macd_signal_line:.2f}, Histogram: {macd_histogram:.2f}")

            if len(all_prices) >= macd_slow_period + 1 and macd_line is not None and macd_signal_line is not None:
                macd_series_prev_calc = ta.trend.macd(all_prices.iloc[:-1], window_fast=macd_fast_period, window_slow=macd_slow_period)
                macd_line_prev = macd_series_prev_calc.iloc[-1] if not macd_series_prev_calc.empty else None
                
                macd_signal_series_prev_calc = ta.trend.macd_signal(all_prices.iloc[:-1], window_fast=macd_fast_period, window_slow=macd_slow_period, window_sign=macd_signal_period)
                macd_signal_line_prev = macd_signal_series_prev_calc.iloc[-1] if not macd_signal_series_prev_calc.empty else None

                if macd_line_prev is not None and macd_signal_line_prev is not None:
                    if macd_line_prev < macd_signal_line_prev and macd_line >= macd_signal_line:
                        print(f"!!! {asset_type}: MACD Cross Up (Confirm BUY) !!!")
                        if trade_signal == "NEUTRAL" or trade_signal == "HOLD_BULLISH":
                            trade_signal = "BUY"
                    elif macd_line_prev > macd_signal_line_prev and macd_line <= macd_signal_line:
                        print(f"!!! {asset_type}: MACD Cross Down (Confirm SELL) !!!")
                        if trade_signal == "NEUTRAL" or trade_signal == "HOLD_BEARISH":
                            trade_signal = "SELL"


    # Anpassung der Take Profit / Stop Loss Faktoren basierend auf dem Signal und Indikatoren
    # Diese Logik wird NUR ausgeführt, wenn historische Daten vorhanden und Indikatoren berechnet wurden
    if historical_prices is not None and asset_type in ["Bitcoin (BTC)", "XAUUSD"]:
        if asset_type == "XAUUSD":
            take_profit_factor = 1.005
            stop_loss_factor = 0.998
            if trade_signal == "BUY":
                take_profit_factor = 1.007
                stop_loss_factor = 0.997
                if rsi_value is not None and rsi_value < rsi_oversold:
                    print(f"XAUUSD: RSI ({rsi_value:.2f}) confirms BUY from oversold.")
                    take_profit_factor = 1.009
                if macd_line is not None and macd_signal_line is not None and macd_line > macd_signal_line:
                    print(f"XAUUSD: MACD ({macd_line:.2f}) above Signal ({macd_signal_line:.2f}) confirms BUY strength.")
                    take_profit_factor = max(take_profit_factor, 1.01)
            elif trade_signal == "SELL":
                take_profit_factor = 0.995
                stop_loss_factor = 1.002
                if rsi_value is not None and rsi_value > rsi_overbought:
                    print(f"XAUUSD: RSI ({rsi_value:.2f}) confirms SELL from overbought.")
                    take_profit_factor = 0.993
                if macd_line is not None and macd_signal_line is not None and macd_line < macd_signal_line:
                    print(f"XAUUSD: MACD ({macd_line:.2f}) below Signal ({macd_signal_line:.2f}) confirms SELL strength.")
                    take_profit_factor = min(take_profit_factor, 0.99)
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
                if rsi_value is not None and rsi_value < rsi_oversold:
                    print(f"Bitcoin: RSI ({rsi_value:.2f}) confirms BUY from oversold.")
                    take_profit_factor = 1.05
                if macd_line is not None and macd_signal_line is not None and macd_line > macd_signal_line:
                    print(f"Bitcoin: MACD ({macd_line:.2f}) above Signal ({macd_signal_line:.2f}) confirms BUY strength.")
                    take_profit_factor = max(take_profit_factor, 1.06)
            elif trade_signal == "SELL":
                take_profit_factor = 0.97
                stop_loss_factor = 1.025
                if rsi_value is not None and rsi_value > rsi_overbought:
                    print(f"Bitcoin: RSI ({rsi_value:.2f}) confirms SELL from overbought.")
                    take_profit_factor = 0.96
                if macd_line is not None and macd_signal_line is not None and macd_line < macd_signal_line:
                    print(f"Bitcoin: MACD ({macd_line:.2f}) below Signal ({macd_signal_line:.2f}) confirms SELL strength.")
                    take_profit_factor = min(take_profit_factor, 0.95)
            elif trade_signal == "HOLD_BULLISH":
                take_profit_factor = 1.035
            elif trade_signal == "HOLD_BEARISH":
                stop_loss_factor = 0.985
    # WICHTIG: Brent Oil fällt HIER auf die initialen Faktoren zurück,
    # da es nicht in den obigen 'if historical_prices is not None'-Block fällt.
    # Die Initialisierung ganz oben stellt sicher, dass es immer Werte gibt.

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
    btc_historical_prices = get_bitcoin_historical_prices(interval='1h', limit=150)
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
    gold_historical_prices = get_gold_historical_prices(interval='1min', outputsize=150)
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
    # Hier rufen wir historische Daten ab, wissen aber, dass es wahrscheinlich fehlschlägt (Free Tier).
    # Die calculate_trade_levels Funktion fängt dies ab.
    brent_oil_historical_prices = get_brent_oil_historical_prices(interval='1min', outputsize=150)
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