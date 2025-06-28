# backtester_daily.py (Finale Version für die Tages-Strategie mit Equity-Curve-Export)
import pandas as pd
import numpy as np
import joblib
import os
import ta
import json
from sqlalchemy import text
from database import engine, historical_data_daily

# --- KONFIGURATIONEN ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
INITIAL_CAPITAL = 100
TRADE_SIZE_USD = 45
TRANSACTION_COST_PERCENT = 0.001
CONFIDENCE_THRESHOLD = 0.75
TREND_SMA_PERIOD = 150 
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

def load_data_from_db(symbol: str):
    print(f"Lade alle Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception as e: return pd.DataFrame()

def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    print(f"Füge Features hinzu (Trend-Periode: {trend_sma_period} Tage)...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20); df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9); df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2); df['bb_high'] = bollinger.bollinger_hband(); df['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3); df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
    df['sma_trend'] = ta.trend.sma_indicator(df['close'], window=trend_sma_period)
    df.dropna(inplace=True)
    return df

def run_daily_backtest(symbol: str):
    print(f"\n--- Starte finalen DAILY Backtest für {symbol} ---")
    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_swing.pkl')
    if not os.path.exists(model_path): return None
    model = joblib.load(model_path)
    df_full = load_data_from_db(symbol)
    if df_full.empty: return None
    df = add_features(df_full.copy(), TREND_SMA_PERIOD)
    
    X_full = df[model.feature_names_in_]
    probabilities = model.predict_proba(X_full)
    df['buy_proba'] = probabilities[:, np.where(model.classes_ == 1)[0][0]]

    capital = INITIAL_CAPITAL
    position_open = False
    trades = []
    equity_curve = [{'timestamp': df['timestamp'].iloc[0].strftime('%Y-%m-%d'), 'equity': INITIAL_CAPITAL}]
    entry_price, take_profit_price, stop_loss_price, position_size_asset = 0, 0, 0, 0

    for i in range(len(df)):
        current_timestamp = df['timestamp'].iloc[i]
        if position_open:
            if df['high'].iloc[i] >= take_profit_price:
                profit_usd = (take_profit_price - entry_price) * position_size_asset; capital += profit_usd - (TRADE_SIZE_USD * TRANSACTION_COST_PERCENT * 2); trades.append({'profit_usd': profit_usd}); position_open = False; equity_curve.append({'timestamp': current_timestamp.strftime('%Y-%m-%d'), 'equity': capital})
            elif df['low'].iloc[i] <= stop_loss_price:
                profit_usd = (stop_loss_price - entry_price) * position_size_asset; capital += profit_usd - (TRADE_SIZE_USD * TRANSACTION_COST_PERCENT * 2); trades.append({'profit_usd': profit_usd}); position_open = False; equity_curve.append({'timestamp': current_timestamp.strftime('%Y-%m-%d'), 'equity': capital})
        if not position_open:
            if (df['close'].iloc[i] > df['sma_trend'].iloc[i]) and (df['buy_proba'].iloc[i] > CONFIDENCE_THRESHOLD):
                position_open = True; entry_price = df['close'].iloc[i]; position_size_asset = TRADE_SIZE_USD / entry_price; atr_at_entry = df['atr'].iloc[i]
                take_profit_price = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER); stop_loss_price = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
    
    total_profit_usd = capital - INITIAL_CAPITAL
    total_return_percent = total_profit_usd / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0: return None
    win_rate = len([t for t in trades if t['profit_usd'] > 0]) / num_trades * 100
    returns = pd.Series([e['equity'] for e in equity_curve]).pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return {
        'Symbol': symbol, 'Gesamtrendite_%': round(total_return_percent, 2),
        'Gewinnrate_%': round(win_rate, 2), 'Sharpe_Ratio': round(sharpe_ratio, 2),
        'Anzahl_Trades': num_trades, 'Endkapital_$': round(capital, 2),
        'equity_curve': equity_curve
    }

if __name__ == "__main__":
    all_results = []
    print("Starte finale Backtests mit Equity-Curve-Export...")
    for symbol in SYMBOLS:
        result = run_daily_backtest(symbol)
        if result:
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n\n--- TAGES-STRATEGIE BACKTEST-ERGEBNISSE ---")
        print(results_df[['Symbol', 'Gesamtrendite_%', 'Gewinnrate_%', 'Sharpe_Ratio', 'Anzahl_Trades']].to_string())
        try:
            with open('backtest_results_daily.json', 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            print("\nErfolgreich 'backtest_results_daily.json' mit Equity-Kurven gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der JSON-Datei: {e}")
    else:
        print("\nKeine Trades wurden ausgeführt.")