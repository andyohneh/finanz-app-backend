# backtester_swing.py (DIAMANT-STANDARD mit realistischem Position Sizing)
import pandas as pd
import numpy as np
import joblib
import os
import ta
from sqlalchemy import text
from database import engine, historical_data_daily

# --- KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
INITIAL_CAPITAL = 100  # Wir starten mit einem fiktiven Gesamtkapital von 100€
TRADE_SIZE_USD = 10     # JEDER Trade wird mit einer festen Größe von 10€ getätigt
TRANSACTION_COST_PERCENT = 0.001

# Beste gefundene Parameter
CONFIDENCE_THRESHOLD = 0.75
TREND_SMA_PERIOD = 100

# Risk-Management-Parameter
TAKE_PROFIT_ATR_MULTIPLIER = 3.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

def load_all_data(symbol: str) -> pd.DataFrame:
    # ... unverändert ...
    print(f"Lade alle Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            return df
    except Exception as e:
        print(f"Fehler beim Laden der Daten für {symbol}: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    # ... unverändert ...
    print(f"Füge Features hinzu (Trend-Periode: {trend_sma_period} Tage)...")
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

def run_true_swing_backtest(symbol: str):
    """Führt den finalen Backtest mit realistischem Position Sizing durch."""
    print(f"\n--- Starte finalen, realistischen SWING Backtest für {symbol} ---")

    model_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{model_filename}_swing.pkl')
    model = joblib.load(model_path)

    df_full = load_all_data(symbol)
    if df_full.empty: return
    df = add_features(df_full.copy(), TREND_SMA_PERIOD)
    
    model_features = model.feature_names_in_
    X_full = df[model_features]

    probabilities = model.predict_proba(X_full)
    df['buy_proba'] = probabilities[:, np.where(model.classes_ == 1)[0][0]]

    capital = INITIAL_CAPITAL
    position_open = False
    trades = []
    
    for i in range(len(df)):
        if position_open:
            # TP-Check
            if df['high'].iloc[i] >= take_profit_price:
                exit_price = take_profit_price
                profit_usd = (exit_price - entry_price) * position_size_asset
                profit_usd -= (TRADE_SIZE_USD + profit_usd) * TRANSACTION_COST_PERCENT # Kosten abziehen
                capital += profit_usd
                trades.append({'profit_usd': profit_usd})
                position_open = False
            # SL-Check
            elif df['low'].iloc[i] <= stop_loss_price:
                exit_price = stop_loss_price
                profit_usd = (exit_price - entry_price) * position_size_asset
                profit_usd -= (TRADE_SIZE_USD + abs(profit_usd)) * TRANSACTION_COST_PERCENT # Kosten abziehen
                capital += profit_usd
                trades.append({'profit_usd': profit_usd})
                position_open = False
            
        if not position_open:
            is_uptrend = df['close'].iloc[i] > df['sma_trend'].iloc[i]
            buy_confidence = df['buy_proba'].iloc[i]
            if is_uptrend and buy_confidence > CONFIDENCE_THRESHOLD:
                position_open = True
                entry_price = df['close'].iloc[i]
                position_size_asset = TRADE_SIZE_USD / entry_price
                
                atr_at_entry = df['atr'].iloc[i]
                take_profit_price = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER)
                stop_loss_price = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
            
    total_profit_usd = capital - INITIAL_CAPITAL
    total_return_percent = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0:
        print("\nKeine Trades wurden ausgeführt.")
        return
        
    winning_trades = [t for t in trades if t['profit_usd'] > 0]
    win_rate = len(winning_trades) / num_trades * 100
    
    print("\n--- Realistische Swing-Backtest-Ergebnisse ---")
    print(f"Startkapital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Einsatz pro Trade: ${TRADE_SIZE_USD:,.2f}")
    print(f"Endkapital: ${capital:,.2f}")
    print(f"Gesamtgewinn: ${total_profit_usd:,.2f}")
    print(f"Gesamtrendite (auf Startkapital): {total_return_percent:.2f}%")
    print(f"Anzahl abgeschlossener Trades: {num_trades}")
    print(f"Gewinnrate: {win_rate:.2f}%")

if __name__ == "__main__":
    for symbol in SYMBOLS:
        run_true_swing_backtest(symbol)