# predictor_longshort.py (FINALE VERSION: Sucht nach Long- und Short-Signalen und sendet Push-Nachrichten)
import os
import json
import pandas as pd
import numpy as np
import joblib
import ta
import requests
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data_daily, predictions, push_subscriptions
from dotenv import load_dotenv
from datetime import datetime
from pywebpush import webpush, WebPushException

# --- Initialisierung & Konfiguration ---
load_dotenv()
VAPID_PRIVATE_KEY = os.getenv('VAPID_PRIVATE_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
DATA_LIMIT_FOR_FEATURES = 201

# Parameter aus unseren Backtests
CONFIDENCE_LONG = 0.75
TREND_PERIOD_LONG = 150
CONFIDENCE_SHORT = 0.60
TREND_PERIOD_SHORT = 50

# Risk-Management
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['sma_fast'] = ta.trend.sma_indicator(df_copy['close'], window=20); df_copy['sma_slow'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd = ta.trend.MACD(df_copy['close'], window_slow=26, window_fast=12, window_sign=9); df_copy['macd'] = macd.macd(); df_copy['macd_signal'] = macd.macd_signal()
    df_copy['atr'] = ta.volatility.AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df_copy['close'], window=20, window_dev=2)
    df_copy['bb_high'] = bollinger.bollinger_hband(); df_copy['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14, smooth_window=3)
    df_copy['stoch_k'] = stoch.stoch(); df_copy['stoch_d'] = stoch.stoch_signal()
    df_copy['sma_trend'] = ta.trend.sma_indicator(df_copy['close'], window=trend_sma_period)
    df_copy.dropna(inplace=True)
    return df_copy

def send_push_notification(subscription_info_json, payload_str):
    if not VAPID_PRIVATE_KEY:
        print("FEHLER: VAPID_PRIVATE_KEY nicht konfiguriert.")
        return
    try:
        webpush(
            subscription_info=json.loads(subscription_info_json),
            data=payload_str,
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={"sub": "mailto:deine-email@example.com"} # Ersetze dies mit deiner E-Mail
        )
        print("Push-Nachricht erfolgreich gesendet.")
    except WebPushException as ex:
        print(f"Fehler beim Senden der Push-Nachricht: {ex}")

def run_longshort_prediction_cycle():
    print(f"\n--- Starte 'Allwetter'-Analyse um {datetime.now()} ---")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Verarbeite {symbol} ---")
            try:
                # 1. Neueste Tages-Daten holen und speichern
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=1&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=15); response.raise_for_status()
                data = response.json()
                if not (data.get('status') == 'ok' and 'values' in data): continue
                latest_candle = data['values'][0]
                record = {'symbol': symbol, 'timestamp': datetime.strptime(latest_candle['datetime'], '%Y-%m-%d'), 'open': float(latest_candle['open']), 'high': float(latest_candle['high']), 'low': float(latest_candle['low']), 'close': float(latest_candle['close']), 'volume': float(latest_candle.get('volume', 0))}
                stmt = insert(historical_data_daily).values(record)
                stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                conn.execute(stmt); conn.commit()
                print(f"Tages-Datenpunkt für {symbol} vom {record['timestamp'].date()} gespeichert.")

                # 2. Modelle laden
                symbol_filename = symbol.replace('/', '')
                model_long = joblib.load(os.path.join(MODEL_DIR, f'model_{symbol_filename}_swing.pkl'))
                model_short = joblib.load(os.path.join(MODEL_DIR, f'model_{symbol_filename}_short.pkl'))

                # 3. Daten für Analyse vorbereiten
                query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
                df = pd.read_sql_query(query, conn, params={'symbol': symbol, 'limit': DATA_LIMIT_FOR_FEATURES})
                df = df.iloc[::-1].reset_index(drop=True)
                if df.empty: continue

                # 4. Signale generieren
                df_long_features = add_features(df, TREND_PERIOD_LONG)
                df_short_features = add_features(df, TREND_PERIOD_SHORT)
                latest_data_long = df_long_features.iloc[[-1]]
                latest_data_short = df_short_features.iloc[[-1]]
                long_proba = model_long.predict_proba(latest_data_long[model_long.feature_names_in_])[0, np.where(model_long.classes_ == 1)[0][0]]
                short_proba = model_short.predict_proba(latest_data_short[model_short.feature_names_in_])[0, np.where(model_short.classes_ == -1)[0][0]]
                is_uptrend = latest_data_long['close'].iloc[0] > latest_data_long['sma_trend'].iloc[0]
                is_downtrend = latest_data_short['close'].iloc[0] < latest_data_short['sma_trend'].iloc[0]

                signal, take_profit, stop_loss, atr = "Halten", None, None, latest_data_long['atr'].iloc[0]
                entry_price = latest_data_long['close'].iloc[0]

                if is_uptrend and long_proba > CONFIDENCE_LONG:
                    signal = "Kaufen"
                    take_profit = entry_price + (atr * TAKE_PROFIT_ATR_MULTIPLIER)
                    stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
                elif is_downtrend and short_proba > CONFIDENCE_SHORT:
                    signal = "Verkaufen"
                    take_profit = entry_price - (atr * TAKE_PROFIT_ATR_MULTIPLIER)
                    stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
                
                print(f"-> FINALES Signal: {signal} (Long-Conf: {long_proba:.2f}, Short-Conf: {short_proba:.2f})")

                # 5. Ergebnis speichern und Nachrichten senden
                values_to_insert = { "symbol": symbol, "signal": signal, "entry_price": float(entry_price), "take_profit": float(take_profit) if take_profit is not None else None, "stop_loss": float(stop_loss) if stop_loss is not None else None }
                pred_stmt = insert(predictions).values(values_to_insert)
                pred_stmt = pred_stmt.on_conflict_do_update( index_elements=['symbol'], set_={ "signal": pred_stmt.excluded.signal, "entry_price": pred_stmt.excluded.entry_price, "take_profit": pred_stmt.excluded.take_profit, "stop_loss": pred_stmt.excluded.stop_loss, "last_updated": datetime.utcnow() })
                conn.execute(pred_stmt); conn.commit()
                print("Analyse-Ergebnis erfolgreich gespeichert.")

                if signal != "Halten":
                    payload = json.dumps({"title": f"Neues KI-Signal: {symbol}", "body": f"Signal: {signal} @ {entry_price:.2f}"})
                    subscriptions_query = text("SELECT subscription_json FROM push_subscriptions")
                    subscribers = conn.execute(subscriptions_query).fetchall()
                    for sub_row in subscribers:
                        send_push_notification(sub_row[0], payload)

            except Exception as e:
                print(f"Ein Fehler im Zyklus für {symbol} ist aufgetreten: {e}")

if __name__ == "__main__":
    run_longshort_prediction_cycle()
