# backtester.py (FINALE PLATIN-VERSION mit Trade Management)
import pandas as pd
import numpy as np
import joblib
import os
import ta
from sqlalchemy import text
from database import engine

# --- FINALE OPTIMIERTE KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
INITIAL_CAPITAL = 100
TRANSACTION_COST_PERCENT = 0.001

# Beste gefundene Parameter aus unserer Optimierung
OPTIMIZED_CONFIDENCE = 0.75
OPTIMIZED_TREND_PERIOD = 150

# Parameter für das Trade Management (Risk/Reward-Ratio)
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

def load_all_data(symbol: str) -> pd.DataFrame:
    # ... unverändert ...
    print(f"Lade alle historischen Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            print(f"Erfolgreich {len(df)} Datenpunkte für {symbol} geladen.")
            return df
    except Exception as e:
        print(f"Fehler beim Laden der Daten für {symbol}: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    # ... unverändert ...
    print(f"Füge Features hinzu (Trend-Periode: {trend_sma_period})...")
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

def run_final_backtest(symbol: str):
    """Führt den finalen Backtest mit den optimierten Parametern und Trade Management durch."""
    print(f"\n--- Starte Finalen Backtest für {symbol} ---")
    print(f"Parameter: Konfidenz={OPTIMIZED_CONFIDENCE}, Trend={OPTIMIZED_TREND_PERIOD}, R/R={TAKE_PROFIT_ATR_MULTIPLIER}:{STOP_LOSS_ATR_MULTIPLIER}")

    model_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{model_filename}.pkl')
    model = joblib.load(model_path)

    df_full = load_all_data(symbol)
    df = add_features(df_full.copy(), OPTIMIZED_TREND_PERIOD)
    
    feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
    X_full = df[feature_columns]

    probabilities = model.predict_proba(X_full)
    df['buy_proba'] = probabilities[:, np.where(model.classes_ == 1)[0][0]]

    # --- FINALE TRADING-SIMULATION MIT TP/SL ---
    capital = INITIAL_CAPITAL
    position_open = False
    trades = []
    
    take_profit_price = 0
    stop_loss_price = 0

    for i in range(len(df)):
        # Wenn eine Position offen ist, prüfe zuerst, ob sie geschlossen werden muss
        if position_open:
            if df['high'].iloc[i] >= take_profit_price:
                exit_price = take_profit_price
                profit = (exit_price - entry_price) / entry_price
                trades.append({'profit_percent': profit * 100})
                capital *= (1 + profit * (1-TRANSACTION_COST_PERCENT))
                position_open = False
                continue
            elif df['low'].iloc[i] <= stop_loss_price:
                exit_price = stop_loss_price
                profit = (exit_price - entry_price) / entry_price
                trades.append({'profit_percent': profit * 100})
                capital *= (1 + profit * (1-TRANSACTION_COST_PERCENT))
                position_open = False
                continue

        # Wenn keine Position offen ist, prüfe auf ein neues Kaufsignal
        is_uptrend = df['close'].iloc[i] > df['sma_trend'].iloc[i]
        buy_confidence = df['buy_proba'].iloc[i]

        if not position_open and is_uptrend and buy_confidence > OPTIMIZED_CONFIDENCE:
            entry_price = df['close'].iloc[i]
            position_open = True
            
            atr_at_entry = df['atr'].iloc[i]
            take_profit_price = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER)
            stop_loss_price = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
            
    # --- ENDE DER SIMULATION ---

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0:
        print("Keine Trades wurden ausgeführt.")
        return
        
    winning_trades = [t for t in trades if t['profit_percent'] > 0]
    win_rate = len(winning_trades) / num_trades * 100
    
    print("\n--- Finale Backtest-Ergebnisse ---")
    print(f"Startkapital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Endkapital: ${capital:,.2f}")
    print(f"Gesamtrendite: {total_return:.2f}%")
    print(f"Anzahl abgeschlossener Trades: {num_trades}")
    print(f"Gewinnrate: {win_rate:.2f}%")

if __name__ == "__main__":
    for symbol in SYMBOLS:
        run_final_backtest(symbol)