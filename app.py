from flask import Flask, jsonify, request 
from flask_cors import CORS
import requests
import os
import pandas as pd
import ta.trend
import ta.momentum

app = Flask(__name__)
CORS(app)

# --- API-SCHLÜSSEL SICHER VERWALTEN ---
# ERSETZE DIESE PLATZHALTER MIT DEINEN ECHTEN SCHLÜSSELN!
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

# --- get_api_usage() Funktion ist jetzt entfernt ---

# --- UNSERE "KI"-LOGIK (MIT SMA Crossover, RSI & MACD & kombiniertem Signal) ---
def calculate_trade_levels(current_price, historical_prices, asset_type, params):
    if current_price is None:
        return None, None, None, "N/A", "gray", "question"

    # Default-Faktoren (für den Fall, dass keine Indikatoren berechnet werden oder HALTEN (Neutral))
    take_profit_factor_long = 1.01
    stop_loss_factor_long = 0.99
    take_profit_factor_short = 0.99
    stop_loss_factor_short = 1.01

    if asset_type == "XAUUSD":
        take_profit_factor_long = 1.005 # 0.5% TP long
        stop_loss_factor_long = 0.998  # 0.2% SL long
        take_profit_factor_short = 0.995 # 0.5% TP short
        stop_loss_factor_short = 1.002 # 0.2% SL short
    elif asset_type == "Bitcoin (BTC)":
        take_profit_factor_long = 1.03 # 3% TP long
        stop_loss_factor_long = 0.98  # 2% SL long
        take_profit_factor_short = 0.97 # 3% TP short
        stop_loss_factor_short = 1.02 # 2% SL short

    # Indikator-Parameter von params übernehmen oder Standardwerte verwenden
    fast_sma_period = params.get('fast_sma_period', 10)
    slow_sma_period = params.get('slow_sma_period', 20)
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)
    macd_fast_period = params.get('macd_fast_period', 12)
    macd_slow_period = params.get('macd_slow_period', 26)
    macd_signal_period = params.get('macd_signal_period', 9)
    
    # Zusätzlicher Parameter für Signalstärke/Volatilitätsfilter
    min_price_deviation_from_sma = params.get('min_price_deviation_from_sma', 0.002) # 0.2% Abweichung vom langsamen SMA
    # NEU: Parameter für dynamischen Einstiegspreis
    entry_price_correction_factor = params.get('entry_price_correction_factor', 0.001) # 0.1% Abweichung für Einstieg

    # Initialisiere Signale
    sma_signal = "NEUTRAL"
    rsi_signal_status = "NEUTRAL"
    macd_crossover_signal = "NEUTRAL"
    final_trade_signal = "HALTEN (Neutral)"
    signal_color = "gray"
    signal_icon = "minus" # Standard ist neutral
    
    # Der Einstiegspreis wird initial auf den aktuellen Preis gesetzt und dann angepasst
    entry_price = round(current_price, 2) 

    rsi_value = None
    macd_line = None
    macd_signal_line = None
    slow_sma_value = None 

    # Indikatoren nur für Bitcoin und XAUUSD berechnen, wenn genügend historische Daten vorhanden sind
    if historical_prices is not None and asset_type in ["Bitcoin (BTC)", "XAUUSD"]:
        all_prices = pd.concat([historical_prices, pd.Series([current_price])]).reset_index(drop=True)

        # SMA Crossover
        if len(all_prices) >= slow_sma_period + 1:
            fast_sma_value = ta.trend.sma_indicator(all_prices, window=fast_sma_period).iloc[-1]
            slow_sma_value = ta.trend.sma_indicator(all_prices, window=slow_sma_period).iloc[-1]
            fast_sma_prev = ta.trend.sma_indicator(all_prices, window=fast_sma_period).iloc[-2]
            slow_sma_prev = ta.trend.sma_indicator(all_prices, window=slow_sma_period).iloc[-2]

            if fast_sma_prev < slow_sma_prev and fast_sma_value >= slow_sma_value:
                sma_signal = "KAUF" # Golden Cross
                print(f"!!! {asset_type}: SMA KAUF Signal (Golden Cross) !!!")
            elif fast_sma_prev > slow_sma_prev and fast_sma_value <= slow_sma_value:
                sma_signal = "VERKAUF" # Death Cross
                print(f"!!! {asset_type}: SMA VERKAUF Signal (Death Cross) !!!")
            elif current_price > slow_sma_value:
                sma_signal = "TREND_AUF" # Preis über langem SMA
            elif current_price < slow_sma_value:
                sma_signal = "TREND_AB" # Preis unter langem SMA
            print(f"{asset_type}: Fast SMA: {fast_sma_value:.2f}, Slow SMA: {slow_sma_value:.2f}")

        # RSI
        if len(all_prices) >= rsi_period:
            rsi_value = ta.momentum.rsi(all_prices, window=rsi_period).iloc[-1]
            print(f"{asset_type}: RSI ({rsi_period} period): {rsi_value:.2f}")
            if rsi_value < rsi_oversold:
                rsi_signal_status = "ÜBERVERKAUFT"
            elif rsi_value > rsi_overbought:
                rsi_signal_status = "ÜBERKAUFT"

        # MACD
        if len(all_prices) >= max(macd_fast_period, macd_slow_period, macd_signal_period) + 1:
            macd_series = ta.trend.macd(all_prices, window_fast=macd_fast_period, window_slow=macd_slow_period)
            macd_line = macd_series.iloc[-1]
            macd_signal_line = ta.trend.macd_signal(all_prices, window_fast=macd_fast_period, window_slow=macd_slow_period, window_sign=macd_signal_period).iloc[-1]
            macd_histogram = ta.trend.macd_diff(all_prices, window_fast=macd_fast_period, window_slow=macd_slow_period, window_sign=macd_signal_period).iloc[-1]
            print(f"{asset_type}: MACD Line: {macd_line:.2f}, Signal Line: {macd_signal_line:.2f}, Histogram: {macd_histogram:.2f}")

            macd_series_prev_calc = ta.trend.macd(all_prices.iloc[:-1], window_fast=macd_fast_period, window_slow=macd_slow_period)
            macd_line_prev = macd_series_prev_calc.iloc[-1] if not macd_series_prev_calc.empty else None
            
            macd_signal_series_prev_calc = ta.trend.macd_signal(all_prices.iloc[:-1], window_fast=macd_fast_period, window_slow=macd_slow_period, window_sign=macd_signal_period)
            macd_signal_line_prev = macd_signal_series_prev_calc.iloc[-1] if not macd_signal_series_prev_calc.empty else None

            if macd_line_prev is not None and macd_signal_line_prev is not None:
                if macd_line_prev < macd_signal_line_prev and macd_line >= macd_signal_line:
                    macd_crossover_signal = "KAUF" # MACD Cross Up
                    print(f"!!! {asset_type}: MACD KAUF Signal (Cross Up) !!!")
                elif macd_line_prev > macd_signal_line_prev and macd_line <= macd_signal_line:
                    macd_crossover_signal = "VERKAUF" # MACD Cross Down
                    print(f"!!! {asset_type}: MACD VERKAUF Signal (Cross Down) !!!")


        # --- Kombinierte Swing-Trading-Strategie zur finalen Signalgenerierung ---
        # Priorisiere klare Signale und nutze andere Indikatoren zur Bestätigung
        
        is_above_sma_deviation = (current_price > slow_sma_value * (1 + min_price_deviation_from_sma)) if slow_sma_value is not None else False
        is_below_sma_deviation = (current_price < slow_sma_value * (1 - min_price_deviation_from_sma)) if slow_sma_value is not None else False

        # STARKES KAUF-Signal (striktere Bedingungen)
        if (sma_signal == "KAUF" or macd_crossover_signal == "KAUF") and \
           rsi_signal_status != "ÜBERKAUFT" and \
           is_above_sma_deviation: # Zusätzliche Bedingung
            final_trade_signal = "KAUFEN"
            signal_color = "green"
            signal_icon = "arrow-up"
            # Dynamischer Einstiegspreis für Kauf-Signal (leicht unter dem aktuellen Preis)
            entry_price = round(current_price * (1 - entry_price_correction_factor), 2)
            
        # STARKES VERKAUF-Signal (striktere Bedingungen)
        elif (sma_signal == "VERKAUF" or macd_crossover_signal == "VERKAUF") and \
             rsi_signal_status != "ÜBERVERKAUFT" and \
             is_below_sma_deviation: # Zusätzliche Bedingung
            final_trade_signal = "VERKAUFEN"
            signal_color = "red"
            signal_icon = "arrow-down"
            # Dynamischer Einstiegspreis für Verkauf-Signal (leicht über dem aktuellen Preis)
            entry_price = round(current_price * (1 + entry_price_correction_factor), 2)

        # HALTEN / VORSICHT-Signale, wenn keine klaren Kauf/Verkauf-Signale
        else:
            # Hier ist der Einstiegspreis weiterhin der aktuelle Preis,
            # da man nicht aktiv einsteigt, sondern nur den Trend beobachtet.
            final_trade_signal = "HALTEN (Neutral)" # Default
            signal_color = "gray"
            signal_icon = "minus"

            if sma_signal == "TREND_AUF":
                final_trade_signal = "HALTEN (Aufwärtstrend)"
                signal_color = "darkgreen"
                signal_icon = "chevron-up"
            elif sma_signal == "TREND_AB":
                final_trade_signal = "HALTEN (Abwärtstrend)"
                signal_color = "darkred"
                signal_icon = "chevron-down"
            elif rsi_signal_status == "ÜBERKAUFT":
                final_trade_signal = "VORSICHT (Überkauft)"
                signal_color = "orange"
                signal_icon = "exclamation"
            elif rsi_signal_status == "ÜBERVERKAUFT":
                final_trade_signal = "VORSICHT (Überverkauft)"
                signal_color = "lightblue"
                signal_icon = "exclamation"


        # Anpassung der Take Profit / Stop Loss Faktoren basierend auf dem finalen Signal
        # ACHTUNG: Die Faktoren sind jetzt immer relativ zum dynamisch berechneten entry_price
        if final_trade_signal == "KAUFEN": # Für Long-Position
            if asset_type == "XAUUSD":
                take_profit = round(entry_price * 1.015, 2) # TP ist ÜBER dem dynamischen Einstieg
                stop_loss = round(entry_price * 0.994, 2)  # SL ist UNTER dem dynamischen Einstieg
            elif asset_type == "Bitcoin (BTC)":
                take_profit = round(entry_price * 1.07, 2)
                stop_loss = round(entry_price * 0.96, 2)
        elif final_trade_signal == "VERKAUFEN": # Für Short-Position
            if asset_type == "XAUUSD":
                take_profit = round(entry_price * 0.988, 2) # TP ist UNTER dem dynamischen Einstieg
                stop_loss = round(entry_price * 1.004, 2)  # SL ist ÜBER dem dynamischen Einstieg
            elif asset_type == "Bitcoin (BTC)":
                take_profit = round(entry_price * 0.94, 2)
                stop_loss = round(entry_price * 1.05, 2)
        else: # HALTEN oder VORSICHT (neutrale oder trendbasierte Faktoren)
            # Hier bleiben TP/SL relativ zum aktuellen Preis, oder leicht angepasst
            if asset_type == "XAUUSD":
                if final_trade_signal == "HALTEN (Aufwärtstrend)":
                    take_profit = round(current_price * 1.004, 2)
                    stop_loss = round(current_price * 0.999, 2)
                elif final_trade_signal == "HALTEN (Abwärtstrend)":
                    take_profit = round(current_price * 0.996, 2) # Short-Bias für TP
                    stop_loss = round(current_price * 1.001, 2) # Short-Bias für SL
                else: # Neutral
                    take_profit = round(current_price * 1.003, 2)
                    stop_loss = round(current_price * 0.999, 2)
            elif asset_type == "Bitcoin (BTC)":
                if final_trade_signal == "HALTEN (Aufwärtstrend)":
                    take_profit = round(current_price * 1.015, 2)
                    stop_loss = round(current_price * 0.988, 2)
                elif final_trade_signal == "HALTEN (Abwärtstrend)":
                    take_profit = round(current_price * 0.985, 2) # Short-Bias
                    stop_loss = round(current_price * 1.015, 2) # Short-Bias
                else: # Neutral
                    take_profit = round(current_price * 1.015, 2)
                    stop_loss = round(current_price * 0.988, 2)

    return entry_price, take_profit, stop_loss, final_trade_signal, signal_color, signal_icon

# --- FLASK-ROUTEN ---
@app.route('/')
def home():
    return "Hallo von deinem Backend-Server!"

@app.route('/api/finance_data')
def get_finance_data():
    assets_data = [] 

    # Parameter aus der URL auslesen
    params = {
        'fast_sma_period': int(request.args.get('fast_sma_period', 10)),
        'slow_sma_period': int(request.args.get('slow_sma_period', 20)),
        'rsi_period': int(request.args.get('rsi_period', 14)),
        'rsi_overbought': int(request.args.get('rsi_overbought', 70)),
        'rsi_oversold': int(request.args.get('rsi_oversold', 30)),
        'macd_fast_period': int(request.args.get('macd_fast_period', 12)),
        'macd_slow_period': int(request.args.get('macd_slow_period', 26)),
        'macd_signal_period': int(request.args.get('macd_signal_period', 9)),
        'min_price_deviation_from_sma': float(request.args.get('min_price_deviation_from_sma', 0.002)), 
        'entry_price_correction_factor': float(request.args.get('entry_price_correction_factor', 0.001)) # NEU: Parameter für Einstiegskorrektur
    }
    print(f"Verwendete Indikator-Parameter: {params}")

    # --- Bitcoin Daten ---
    btc_price = get_bitcoin_price()
    btc_historical_prices = get_bitcoin_historical_prices(interval='1h', limit=max(150, params['slow_sma_period'] + params['macd_slow_period'] + params['macd_signal_period'] + 10))
    btc_entry, btc_tp, btc_sl, btc_signal, btc_color, btc_icon = calculate_trade_levels(btc_price, btc_historical_prices, "Bitcoin (BTC)", params)
    assets_data.append({
        "asset": "Bitcoin (BTC)",
        "currentPrice": f"{btc_price:.2f}" if btc_price is not None else "N/A",
        "entry": f"{btc_entry:.2f}" if btc_entry is not None else "N/A",
        "takeProfit": f"{btc_tp:.2f}" if btc_tp is not None else "N/A",
        "stopLoss": f"{btc_sl:.2f}" if btc_sl is not None else "N/A",
        "signal": btc_signal,
        "color": btc_color,
        "icon": btc_icon
    })

    # --- XAUUSD (Gold) Daten ---
    gold_price = get_gold_price()
    gold_historical_prices = get_gold_historical_prices(interval='1min', outputsize=max(150, params['slow_sma_period'] + params['macd_slow_period'] + params['macd_signal_period'] + 10))
    gold_entry, gold_tp, gold_sl, gold_signal, gold_color, gold_icon = calculate_trade_levels(gold_price, gold_historical_prices, "XAUUSD", params)
    assets_data.append({
        "asset": "XAUUSD",
        "currentPrice": f"{gold_price:.2f}" if gold_price is not None else "N/A",
        "entry": f"{gold_entry:.2f}" if gold_entry is not None else "N/A",
        "takeProfit": f"{gold_tp:.2f}" if gold_tp is not None else "N/A",
        "stopLoss": f"{gold_sl:.2f}" if gold_sl is not None else "N/A",
        "signal": gold_signal,
        "color": gold_color,
        "icon": gold_icon
    })
    
    return jsonify(assets_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)