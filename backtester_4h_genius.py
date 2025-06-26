# backtester_4h_genius.py (Testet das 'Genie'-Modell mit Kontext-Wissen)
import pandas as pd
import numpy as np
import joblib
import os
import ta
from sqlalchemy import text
from database import engine, historical_data_4h, historical_data_daily

# --- KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
INITIAL_CAPITAL = 100
TRADE_SIZE_USD = 45
TRANSACTION_COST_PERCENT = 0.001

# Risk-Management-Parameter
TAKE_PROFIT_ATR_MULTIPLIER = 2.5
STOP_LOSS_ATR_MULTIPLIER = 1.5
TREND_SMA_PERIOD_DAILY = 50 # Der übergeordnete Kontext-Trend

def load_data(symbol: str):
    """Lädt SOWOHL die 4h-Daten als auch die Tages-Daten."""
    print(f"Lade 4h- und Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query_4h = text("SELECT * FROM historical_data_4h WHERE symbol = :symbol ORDER BY timestamp ASC")
            df_4h = pd.read_sql_query(query_4h, conn, params={'symbol': symbol})
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
            
            query_daily = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df_daily = pd.read_sql_query(query_daily, conn, params={'symbol': symbol})
            df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'])
            return df_4h, df_daily
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return pd.DataFrame(), pd.DataFrame()

def add_all_features(df_4h: pd.DataFrame, df_daily: pd.DataFrame, trend_sma_period_4h: int) -> pd.DataFrame:
    """Fügt alle technischen und kontextuellen Features hinzu."""
    print(f"Füge alle Features für 'Genie'-Backtest hinzu (4h-Trend: {trend_sma_period_4h})...")
    
    # 1. Kontext-Feature (Tages-Trend)
    df_daily['sma_trend_daily'] = ta.trend.sma_indicator(df_daily['close'], window=TREND_SMA_PERIOD_DAILY)
    df_daily['is_daily_uptrend'] = (df_daily['close'] > df_daily['sma_trend_daily']).astype(int)
    df_daily['date'] = df_daily['timestamp'].dt.date
    daily_context = df_daily[['date', 'is_daily_uptrend']].drop_duplicates(subset='date', keep='last')
    
    df_4h['date'] = df_4h['timestamp'].dt.date
    df_merged = pd.merge(df_4h, daily_context, on='date', how='left')
    df_merged['is_daily_uptrend'] = df_merged['is_daily_uptrend'].ffill()
    df_merged.drop(columns=['date'], inplace=True)
    
    # 2. Technische 4h-Features
    df = df_merged
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20); df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9); df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2); df['bb_high'] = bollinger.bollinger_hband(); df['bb_low'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3); df['stoch_k'] = stoch.stoch(); df['stoch_d'] = stoch.stoch_signal()
    df['sma_trend_4h'] = ta.trend.sma_indicator(df['close'], window=trend_sma_period_4h)
    
    df.dropna(inplace=True)
    return df

def run_genius_backtest(symbol: str, confidence_threshold: float, trend_sma_period_4h: int):
    """Führt einen einzelnen Backtest für das 'Genie'-Modell durch."""
    print(f"\n--- Teste GENIE {symbol} | Konfidenz: {confidence_threshold:.2f} | 4h-Trend: {trend_sma_period_4h} ---")

    symbol_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}_4h_genius.pkl')
    if not os.path.exists(model_path):
        print(f"'Genie'-Modell nicht gefunden."); return None
    model = joblib.load(model_path)

    df_4h, df_daily = load_data(symbol)
    if df_4h.empty or df_daily.empty: return None
    df = add_all_features(df_4h, df_daily, trend_sma_period_4h)
    
    X_full = df[model.feature_names_in_]
    probabilities = model.predict_proba(X_full)
    df['buy_proba'] = probabilities[:, np.where(model.classes_ == 1)[0][0]]

    capital = INITIAL_CAPITAL
    capital_over_time = [INITIAL_CAPITAL]
    position_open = False
    trades = []
    entry_price, take_profit_price, stop_loss_price = 0, 0, 0
    position_size_asset = 0

    for i in range(len(df)):
        # Trade-Management
        if position_open:
            if df['high'].iloc[i] >= take_profit_price:
                exit_price = take_profit_price
                profit_usd = (exit_price - entry_price) * position_size_asset
                profit_usd -= (TRADE_SIZE_USD + profit_usd) * TRANSACTION_COST_PERCENT
                capital += profit_usd
                trades.append({'profit_usd': profit_usd})
                position_open = False
                continue # Wichtig, um in derselben Kerze keinen neuen Trade zu eröffnen
            elif df['low'].iloc[i] <= stop_loss_price:
                exit_price = stop_loss_price
                profit_usd = (exit_price - entry_price) * position_size_asset
                profit_usd -= (TRADE_SIZE_USD + abs(profit_usd)) * TRANSACTION_COST_PERCENT
                capital += profit_usd
                trades.append({'profit_usd': profit_usd})
                position_open = False
                continue # Wichtig

        # Entry-Logik
        if not position_open:
            is_4h_uptrend = df['close'].iloc[i] > df['sma_trend_4h'].iloc[i]
            buy_confidence = df['buy_proba'].iloc[i]
            if is_4h_uptrend and buy_confidence > confidence_threshold:
                position_open = True
                entry_price = df['close'].iloc[i]
                position_size_asset = TRADE_SIZE_USD / entry_price
                atr_at_entry = df['atr'].iloc[i]
                take_profit_price = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER)
                stop_loss_price = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
        
        capital_over_time.append(capital)

    total_profit_usd = capital - INITIAL_CAPITAL
    total_return_percent = total_profit_usd / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0: return None
        
    winning_trades = [t for t in trades if t['profit_usd'] > 0]
    win_rate = len(winning_trades) / num_trades * 100
    returns = pd.Series(capital_over_time).pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(6 * 252) if np.std(returns) > 0 else 0
    
    return {
        'Symbol': symbol, 'Konfidenz': f"{confidence_threshold:.2f}",
        'Trend_Periode_4h': trend_sma_period_4h, 'Rendite_%': round(total_return_percent, 2),
        'Gewinnrate_%': round(win_rate, 2), 'Sharpe_Ratio': round(sharpe_ratio, 2),
        'Anzahl_Trades': num_trades
    }

if __name__ == "__main__":
    thresholds_to_test = [0.60, 0.65, 0.70]
    trend_periods_to_test = [50, 100]
    all_results = []
    
    print("Starte Parameter-Optimierung für die 'GENIE'-Strategie...")
    for symbol in SYMBOLS:
        for threshold in thresholds_to_test:
            for period in trend_periods_to_test:
                result = run_genius_backtest(symbol, confidence_threshold=threshold, trend_sma_period_4h=period)
                if result:
                    all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df_sorted = results_df.sort_values(by='Sharpe_Ratio', ascending=False)
        
        print("\n\n--- 'GENIE'-STRATEGIE OPTIMIERUNGS-ZUSAMMENFASSUNG ---")
        print("Beste Ergebnisse (nach Sharpe Ratio) oben")
        print(results_df_sorted.to_string())

        try:
            results_df_sorted.to_json('backtest_results_genius.json', orient='records', indent=4)
            print("\nErfolgreich 'backtest_results_genius.json' im Projektordner gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der JSON-Datei: {e}")
    else:
        print("\nKeine Trades wurden in den Tests ausgeführt.")