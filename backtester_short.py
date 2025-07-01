# backtester_short.py (Version 1.3 - Finale Korrektur)
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
TRADE_SIZE_USD = 10
TRANSACTION_COST_PERCENT = 0.001
CONFIDENCE_THRESHOLD = 0.60 
TREND_SMA_PERIOD = 50
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

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
        print(f"Fehler beim Laden der Daten: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    """Fügt Indikatoren basierend auf TAGES-Daten hinzu."""
    print(f"Füge Features für Short-Backtest hinzu...")
    df_copy = df.copy()
    df_copy['sma_fast'] = ta.trend.sma_indicator(df_copy['close'], window=20)
    df_copy['sma_slow'] = ta.trend.sma_indicator(df_copy['close'], window=50)
    df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
    macd = ta.trend.MACD(df_copy['close'], window_slow=26, window_fast=12, window_sign=9)
    df_copy['macd'] = macd.macd(); df_copy['macd_signal'] = macd.macd_signal()
    df_copy['atr'] = ta.volatility.AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df_copy['close'], window=20, window_dev=2)
    df_copy['bb_high'] = bollinger.bollinger_hband(); df_copy['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=14, smooth_window=3)
    df_copy['stoch_k'] = stoch.stoch(); df_copy['stoch_d'] = stoch.stoch_signal()
    df_copy['sma_trend'] = ta.trend.sma_indicator(df_copy['close'], window=trend_sma_period)
    df_copy.dropna(inplace=True)
    return df_copy

def run_short_backtest(symbol: str):
    """Führt den finalen, realistischen Backtest für die Short-Strategie durch."""
    print(f"\n--- Starte finalen SHORT Backtest für {symbol} ---")
    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_short.pkl')
    if not os.path.exists(model_path):
        print(f"FEHLER: Short-Modell für {symbol} nicht gefunden.")
        return None
    model = joblib.load(model_path)

    df_full = load_all_data(symbol)
    if df_full.empty: return None
    df = add_features(df_full.copy(), TREND_SMA_PERIOD)
    
    X_full = df[model.feature_names_in_]
    probabilities = model.predict_proba(X_full)
    # Klasse -1 (unser Short-Signal) wurde beim Training zu 0 verschoben
    df['short_proba'] = probabilities[:, np.where(model.classes_ == 0)[0][0]] 

    capital = INITIAL_CAPITAL
    equity_curve = [{'timestamp': df['timestamp'].iloc[0].strftime('%Y-%m-%d'), 'equity': INITIAL_CAPITAL}]
    position_open = False
    trades = []
    entry_price, take_profit_price, stop_loss_price, position_size_asset = 0, 0, 0, 0

    for i in range(len(df)):
        current_timestamp = df['timestamp'].iloc[i]
        if position_open:
            if df['low'].iloc[i] <= take_profit_price:
                profit_usd = (entry_price - take_profit_price) * position_size_asset
                capital += profit_usd - (TRADE_SIZE_USD * TRANSACTION_COST_PERCENT)
                trades.append({'profit_usd': profit_usd})
                position_open = False
                equity_curve.append({'timestamp': current_timestamp.strftime('%Y-%m-%d'), 'equity': capital})
                continue
            elif df['high'].iloc[i] >= stop_loss_price:
                profit_usd = (entry_price - stop_loss_price) * position_size_asset
                capital += profit_usd - (TRADE_SIZE_USD * TRANSACTION_COST_PERCENT)
                trades.append({'profit_usd': profit_usd})
                position_open = False
                equity_curve.append({'timestamp': current_timestamp.strftime('%Y-%m-%d'), 'equity': capital})
                continue
            
        if not position_open:
            is_downtrend = df['close'].iloc[i] < df['sma_trend'].iloc[i]
            short_confidence = df['short_proba'].iloc[i]
            if is_downtrend and short_confidence > CONFIDENCE_THRESHOLD:
                position_open = True
                entry_price = df['close'].iloc[i]
                position_size_asset = TRADE_SIZE_USD / entry_price
                atr_at_entry = df['atr'].iloc[i]
                take_profit_price = entry_price - (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER)
                stop_loss_price = entry_price + (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
    
    total_profit_usd = capital - INITIAL_CAPITAL
    total_return_percent = total_profit_usd / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0: return None
        
    winning_trades = [t for t in trades if t['profit_usd'] > 0]
    win_rate = len(winning_trades) / num_trades * 100
    
    returns = pd.Series([e['equity'] for e in equity_curve]).pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return {
        'Symbol': symbol,
        'Gesamtrendite_%': round(total_return_percent, 2),
        'Gewinnrate_%': round(win_rate, 2),
        'Sharpe_Ratio': round(sharpe_ratio, 2),
        'Anzahl_Trades': num_trades,
        'equity_curve': equity_curve
    }

if __name__ == "__main__":
    all_results = []
    print("Starte finale Backtests für SHORT-Strategie...")
    for symbol in SYMBOLS:
        result = run_short_backtest(symbol)
        if result:
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n\n--- SHORT-STRATEGIE BACKTEST-ERGEBNISSE ---")
        print(results_df[['Symbol', 'Gesamtrendite_%', 'Gewinnrate_%', 'Sharpe_Ratio', 'Anzahl_Trades']].to_string())
        
        try:
            with open('backtest_results_daily_short.json', 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            print("\nErfolgreich 'backtest_results_daily_short.json' gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der JSON-Datei: {e}")