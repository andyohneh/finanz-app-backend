# predictor.py (FINALE PLATIN-VERSION)
import os
import pandas as pd
import numpy as np
import joblib
import ta
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, predictions
from dotenv import load_dotenv

# --- FINALE OPTIMIERTE KONFIGURATION ---
load_dotenv()
MODEL_DIR = "models"
SYMBOLS_DB = ['BTC/USD', 'XAU/USD']
DATA_LIMIT_FOR_FEATURES = 251 # Etwas mehr, um sicher den 250er SMA berechnen zu können

# Beste gefundene Parameter aus unserer Optimierung
OPTIMIZED_CONFIDENCE = 0.75
OPTIMIZED_TREND_PERIOD = 150

# Parameter für das Trade Management (Risk/Reward-Ratio)
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

def load_latest_data(symbol: str, limit: int) -> pd.DataFrame:
    # ... unverändert ...
    print(f"Lade die letzten {limit} Datenpunkte für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT :limit")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol, 'limit': limit})
            if not df.empty:
                df = df.iloc[::-1].reset_index(drop=True)
            print(f"Erfolgreich {len(df)} Datenpunkte für {symbol} geladen.")
            return df
    except Exception as e:
        print(f"Fehler beim Laden der Daten für {symbol}: {e}")
        return pd.DataFrame()


def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    # ... unverändert ...
    print("Füge Features zu den Daten hinzu...")
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

def make_and_store_predictions():
    """Hauptfunktion: Wendet die finale, profitable Strategie auf Live-Daten an."""
    print("\n--- Starte den finalen Vorhersage-Prozess ---")
    
    for symbol_db in SYMBOLS_DB:
        print(f"\n--- Verarbeite {symbol_db} ---")
        
        model_filename = symbol_db.replace('/', '')
        model_path = os.path.join(MODEL_DIR, f'model_{model_filename}.pkl')
        if not os.path.exists(model_path): continue
        model = joblib.load(model_path)

        df = load_latest_data(symbol_db, DATA_LIMIT_FOR_FEATURES)
        if df.empty or len(df) < OPTIMIZED_TREND_PERIOD: continue
        
        df_features = add_features(df, OPTIMIZED_TREND_PERIOD)
        if df_features.empty: continue
            
        latest_data = df_features.iloc[[-1]]
        
        feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
        latest_features = latest_data[feature_columns]

        probabilities = model.predict_proba(latest_features)
        buy_confidence = probabilities[0, np.where(model.classes_ == 1)[0][0]]

        # FINALE LOGIK
        entry_price = latest_data['close'].iloc[0]
        is_uptrend = entry_price > latest_data['sma_trend'].iloc[0]
        signal, take_profit, stop_loss = "Halten", None, None
        
        if is_uptrend and buy_confidence > OPTIMIZED_CONFIDENCE:
            signal = "Kaufen"
            atr_at_entry = latest_data['atr'].iloc[0]
            take_profit = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER)
            stop_loss = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
        
        print(f"Signal: {signal}, Konfidenz: {buy_confidence:.2f}, TP: {take_profit or 'N/A'}, SL: {stop_loss or 'N/A'}")

        values_to_insert = {
            "symbol": symbol_db, "signal": signal,
            "entry_price": float(entry_price),
            "take_profit": float(take_profit) if take_profit is not None else None,
            "stop_loss": float(stop_loss) if stop_loss is not None else None
        }
        insert_stmt = insert(predictions).values(values_to_insert)
        update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=['symbol'],
            set_={"signal": insert_stmt.excluded.signal, "entry_price": insert_stmt.excluded.entry_price,
                  "take_profit": insert_stmt.excluded.take_profit, "stop_loss": insert_stmt.excluded.stop_loss,
                  "last_updated": pd.to_datetime('now', utc=True)}
        )
        with engine.connect() as conn:
            conn.execute(update_stmt)
            conn.commit()
        print(f"Vorhersage für {symbol_db} erfolgreich in der Datenbank gespeichert/aktualisiert.")

    print("\n--- Vorhersage-Prozess beendet. ---")

if __name__ == "__main__":
    make_and_store_predictions()