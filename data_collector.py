# data_collector.py (DIAMANT-STANDARD 1.1 - Robuste Version)
import os
import pandas as pd
import numpy as np
import joblib
import ta
import requests
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, historical_data, predictions
from dotenv import load_dotenv
from datetime import datetime

# --- KONFIGURATION ---
load_dotenv()
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
DATA_LIMIT_FOR_FEATURES = 251
OPTIMIZED_CONFIDENCE = 0.75
OPTIMIZED_TREND_PERIOD = 150
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    # ... unverändert ...
    # (Code wie bisher)
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['sma_trend'] = ta.trend.sma_indicator(df['close'], window=trend_sma_period)
    df.dropna(inplace=True)
    return df

def run_live_cycle():
    print(f"\n--- Starte Diamant-Zyklus um {datetime.now()} ---")
    with engine.connect() as conn:
        for symbol in SYMBOLS:
            print(f"\n--- Verarbeite {symbol} ---")
            try:
                url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=1&apikey={TWELVEDATA_API_KEY}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not (data.get('status') == 'ok' and 'values' in data):
                    print(f"Fehlerhafte API-Antwort für {symbol}. Überspringe.")
                    continue
                
                latest_candle = data['values'][0]
                record = {
                    'symbol': symbol,
                    'timestamp': datetime.strptime(latest_candle['datetime'], '%Y-%m-%d %H:%M:%S'),
                    'open': float(latest_candle['open']), 'high': float(latest_candle['high']),
                    'low': float(latest_candle['low']), 'close': float(latest_candle['close']),
                    # === HIER IST DIE KORREKTUR ===
                    # .get('volume', 0) versucht, 'volume' zu holen. Wenn es nicht da ist, wird 0 als Standardwert genommen.
                    'volume': float(latest_candle.get('volume', 0))
                }
                
                stmt = insert(historical_data).values(record)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'timestamp'],
                    set_={'close': stmt.excluded.close, 'open': stmt.excluded.open, 'high': stmt.excluded.high, 'low': stmt.excluded.low, 'volume': stmt.excluded.volume}
                )
                conn.execute(stmt)
                conn.commit()
                print(f"Neuester Datenpunkt für {symbol} um {record['timestamp']} gespeichert/aktualisiert.")

                # ... Der Rest der Funktion (Analyse etc.) bleibt unverändert ...
                print("Starte sofortige Analyse...")
                model_filename = symbol.replace('/', '')
                model_path = os.path.join(MODEL_DIR, f'model_{model_filename}.pkl')
                if not os.path.exists(model_path):
                    print("Kein Modell gefunden. Analyse übersprungen.")
                    continue
                model = joblib.load(model_path)
                query = text("SELECT * FROM historical_data WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
                df = pd.read_sql_query(query, conn, params={'symbol': symbol, 'limit': DATA_LIMIT_FOR_FEATURES})
                df = df.iloc[::-1].reset_index(drop=True)
                if df.empty or len(df) < OPTIMIZED_TREND_PERIOD:
                    print("Nicht genügend historische Daten für die Analyse.")
                    continue
                df_features = add_features(df, OPTIMIZED_TREND_PERIOD)
                if df_features.empty:
                    print("Daten-Frame nach Feature-Berechnung leer.")
                    continue
                latest_data = df_features.iloc[[-1]]
                feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
                latest_features = latest_data[feature_columns]
                probabilities = model.predict_proba(latest_features)
                buy_confidence = probabilities[0, np.where(model.classes_ == 1)[0][0]]
                entry_price = latest_data['close'].iloc[0]
                is_uptrend = entry_price > latest_data['sma_trend'].iloc[0]
                signal, take_profit, stop_loss = "Halten", None, None
                if is_uptrend and buy_confidence > OPTIMIZED_CONFIDENCE:
                    signal = "Kaufen"
                    atr_at_entry = latest_data['atr'].iloc[0]
                    take_profit = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER)
                    stop_loss = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
                print(f"-> Signal: {signal}, Konfidenz: {buy_confidence:.2f}, Signal-Preis: {entry_price}")
                values_to_insert = { "symbol": symbol, "signal": signal, "entry_price": float(entry_price), "take_profit": float(take_profit) if take_profit is not None else None, "stop_loss": float(stop_loss) if stop_loss is not None else None }
                pred_stmt = insert(predictions).values(values_to_insert)
                pred_stmt = pred_stmt.on_conflict_do_update(
                    index_elements=['symbol'],
                    set_={ "signal": pred_stmt.excluded.signal, "entry_price": pred_stmt.excluded.entry_price, "take_profit": pred_stmt.excluded.take_profit, "stop_loss": pred_stmt.excluded.stop_loss, "last_updated": datetime.utcnow() }
                )
                conn.execute(pred_stmt)
                conn.commit()
                print("Analyse-Ergebnis erfolgreich gespeichert.")

            except Exception as e:
                print(f"Ein Fehler im Zyklus für {symbol} ist aufgetreten: {e}")

if __name__ == "__main__":
    run_live_cycle()