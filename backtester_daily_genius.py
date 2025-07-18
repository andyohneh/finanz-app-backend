# backtester_daily_genius.py (Testet unser Platin-"Genie"-Modell)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
from sqlalchemy import text
from database import engine, historical_data_daily

# --- KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
INITIAL_CAPITAL = 100
TRADE_SIZE_USD = 45
TRANSACTION_COST_PERCENT = 0.001

# Startparameter für die Strategie - wir nehmen die besten, die wir kennen
CONFIDENCE_THRESHOLD = 0.75
TREND_SMA_PERIOD = 150

# Risk-Management
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

# Dummy-Sentiment für den Backtest (da wir keine historischen News-Daten haben)
# In einem echten, fortgeschrittenen Backtest würde man hier historische Sentiment-Daten laden.
# Für unsere Zwecke testen wir, wie das Modell mit einer neutralen Stimmung umgeht.
BACKTEST_SENTIMENT_SCORE = 0.0

def load_all_data(symbol: str) -> pd.DataFrame:
    """Lädt ALLE TAGES-Daten für ein Symbol."""
    print(f"Lade alle Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception as e:
        return pd.DataFrame()

def add_all_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    """Fügt alle technischen Features UND das Sentiment-Feature hinzu."""
    print(f"Füge alle Features für 'Genie'-Backtest hinzu...")
    df_copy = df.copy()
    df_copy['sma_fast'] = ta.trend.sma_indicator(df_copy['close'], window=20); df_copy['sma_slow'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd = ta.trend.MACD(df_copy['close'], window_slow=26, window_fast=12, window_sign=9); df_copy['macd'] = macd.macd(); df_copy['macd_signal'] = macd.macd_signal()
    df_copy['atr'] = ta.volatility.AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df_copy['close'], window=20, window_dev=2); df_copy['bb_high'] = bollinger.bollinger_hband(); df_copy['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14, smooth_window=3); df_copy['stoch_k'] = stoch.stoch(); df_copy['stoch_d'] = stoch.stoch_signal()
    df_copy['sma_trend'] = ta.trend.sma_indicator(df_copy['close'], window=trend_sma_period)
    
    # Füge das Sentiment-Feature hinzu
    df_copy['sentiment'] = BACKTEST_SENTIMENT_SCORE
    
    df_copy.dropna(inplace=True)
    return df_copy

def run_genius_backtest(symbol: str):
    """Führt den finalen Backtest für das 'Genie'-Modell durch."""
    print(f"\n--- Starte finalen GENIE Backtest für {symbol} ---")

    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_genius.pkl')
    if not os.path.exists(model_path):
        print(f"'Genie'-Modell nicht gefunden."); return None
    model = joblib.load(model_path)

    df_full = load_all_data(symbol)
    if df_full.empty: return None
    df = add_all_features(df_full.copy(), TREND_SMA_PERIOD)
    
    X_full = df[model.feature_names_in_]
    probabilities = model.predict_proba(X_full)
    df['buy_proba'] = probabilities[:, np.where(model.classes_ == 2)[0][0]]

    # Trading-Simulation
    capital = INITIAL_CAPITAL
    position_open = False
    trades = []
    equity_curve = [{'timestamp': df['timestamp'].iloc[0].strftime('%Y-%m-%d'), 'equity': INITIAL_CAPITAL}]
    entry_price, take_profit_price, stop_loss_price, position_size_asset = 0, 0, 0, 0

    for i in range(len(df)):
        current_timestamp = df['timestamp'].iloc[i]
        if position_open:
            if df['high'].iloc[i] >= take_profit_price:
                profit_usd = (take_profit_price - entry_price) * position_size_asset; capital += profit_usd - (TRADE_SIZE_USD * TRANSACTION_COST_PERCENT); trades.append({'profit_usd': profit_usd}); position_open = False; equity_curve.append({'timestamp': current_timestamp.strftime('%Y-%m-%d'), 'equity': capital})
            elif df['low'].iloc[i] <= stop_loss_price:
                profit_usd = (stop_loss_price - entry_price) * position_size_asset; capital += profit_usd - (TRADE_SIZE_USD * TRANSACTION_COST_PERCENT); trades.append({'profit_usd': profit_usd}); position_open = False; equity_curve.append({'timestamp': current_timestamp.strftime('%Y-%m-%d'), 'equity': capital})
        
        if not position_open:
            is_uptrend = df['close'].iloc[i] > df['sma_trend'].iloc[i]
            buy_confidence = df['buy_proba'].iloc[i]
            if is_uptrend and buy_confidence > CONFIDENCE_THRESHOLD:
                position_open = True; entry_price = df['close'].iloc[i]; position_size_asset = TRADE_SIZE_USD / entry_price
                atr_at_entry = df['atr'].iloc[i]
                take_profit_price = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER); stop_loss_price = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)

    # KPI-Berechnung
    total_return_percent = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0: return None
    win_rate = len([t for t in trades if t['profit_usd'] > 0]) / num_trades * 100
    returns = pd.Series([e['equity'] for e in equity_curve]).pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return {'Symbol': symbol, 'Gesamtrendite_%': round(total_return_percent, 2), 'Gewinnrate_%': round(win_rate, 2), 'Sharpe_Ratio': round(sharpe_ratio, 2), 'Anzahl_Trades': num_trades, 'equity_curve': equity_curve}

if __name__ == "__main__":
    all_results = []
    print("Starte finale Backtests für die 'GENIE'-Strategie...")
    for symbol in SYMBOLS:
        result = run_genius_backtest(symbol)
        if result:
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n\n--- 'GENIE'-STRATEGIE BACKTEST-ERGEBNISSE ---")
        print(results_df[['Symbol', 'Gesamtrendite_%', 'Gewinnrate_%', 'Sharpe_Ratio', 'Anzahl_Trades']].to_string())
        
        with open('backtest_results_genius.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("\nErfolgreich 'backtest_results_genius.json' gespeichert.")