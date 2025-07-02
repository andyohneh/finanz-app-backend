# predictor_genius.py (Finale, vollständige Version)
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
VAPID_CLAIMS = {"sub": "mailto:deine-email@example.com"} # Ersetze mit deiner E-Mail
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
DATA_LIMIT_FOR_FEATURES = 201
CONFIDENCE_THRESHOLD = 0.75
TREND_SMA_PERIOD = 150
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5
# Dummy-Sentiment für den Live-Predictor
LIVE_SENTIMENT_SCORE = 0.0

def add_features(df: pd.DataFrame, trend_sma_period: int, sentiment_score: float) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['sma_fast'] = ta.trend.sma_indicator(df_copy['close'], window=20); df_copy['sma_slow'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd = ta.trend.MACD(df_copy['close'], window_slow=26, window_fast=12, window_sign=9); df_copy['macd'] = macd.macd(); df_copy['macd_signal'] = macd.macd_signal()
    df_copy['atr'] = ta.volatility.AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df_copy['close'], window=20, window_dev=2); df_copy['bb_high'] = bollinger.bollinger_hband(); df_copy['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14, smooth_window=3); df_copy['stoch_k'] = stoch.stoch(); df_copy['stoch_d'] = stoch.stoch_signal()
    df_copy['sma_trend'] = ta.trend.sma_indicator(df_copy['close'], window=trend_sma_period)
    df_copy['sentiment'] = sentiment_score
    df_copy.dropna(inplace=True)
    return df_copy

def send_push_notification(subscription_info_json, payload_str):
    print(f"Simulation: Sende Push-Nachricht...")
    pass

def run_genius_prediction_cycle():
    print(f"\n--- Starte 'Genie'-Analyse um {datetime.now()} ---")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Verarbeite {symbol} ---")
            try:
                # 1. Daten holen & speichern
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=1&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=15); response.raise_for_status()
                data = response.json()
                if not (data.get('status') == 'ok' and 'values' in data): continue
                latest_candle = data['values'][0]
                record = {'symbol': symbol, 'timestamp': datetime.strptime(latest_candle['datetime'], '%Y-%m-%d'), 'open': float(latest_candle['open']), 'high': float(latest_candle['high']), 'low': float(latest_candle['low']), 'close': float(latest_candle['close']), 'volume': float(latest_candle.get('volume', 0))}
                stmt = insert(historical_data_daily).values(record)
                stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                conn.execute(stmt); conn.commit()
                
                # 2. Genie-Modell laden
                model_path = os.path.join(MODEL_DIR, f'model_{symbol.replace("/", "")}_genius.pkl')
                model = joblib.load(model_path)

                # 3. Daten für Analyse vorbereiten
                query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
                df = pd.read_sql_query(query, conn, params={'symbol': symbol, 'limit': DATA_LIMIT_FOR_FEATURES})
                df = df.iloc[::-1].reset_index(drop=True)
                if df.empty: continue
                
                df_with_features = add_features(df, TREND_SMA_PERIOD, LIVE_SENTIMENT_SCORE)
                latest_data = df_with_features.iloc[[-1]]
                
                # 4. Signal generieren
                X_live = latest_data[model.feature_names_in_]
                buy_proba = model.predict_proba(X_live)[0, np.where(model.classes_ == 2)[0][0]]

                signal, take_profit, stop_loss = "Halten", None, None
                entry_price = latest_data['close'].iloc[0]
                
                if (latest_data['close'].iloc[0] > latest_data['sma_trend'].iloc[0]) and (buy_proba > CONFIDENCE_THRESHOLD):
                    signal = "Kaufen"
                    atr = latest_data['atr'].iloc[0]
                    take_profit = entry_price + (atr * TAKE_PROFIT_ATR_MULTIPLIER)
                    stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
                
                print(f"-> FINALES 'Genie'-Signal: {signal}")

                # 5. Ergebnis speichern & Push senden
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
                print(f"Fehler im Zyklus für {symbol}: {e}")

if __name__ == "__main__":
    run_genius_prediction_cycle()