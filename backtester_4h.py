# backtester_4h.py (Version 1.1 - mit Ergebnis-Export)
import pandas as pd
import numpy as np
import joblib
import os
import ta
from sqlalchemy import text
from database import engine, historical_data_4h

# --- KONFIGURATION ---
MODEL_DIR = "models"
SYMBOLS = ['BTC/USD', 'XAU/USD']
INITIAL_CAPITAL = 100
TRADE_SIZE_USD = 10
TRANSACTION_COST_PERCENT = 0.001
TAKE_PROFIT_ATR_MULTIPLIER = 2.5
STOP_LOSS_ATR_MULTIPLIER = 1.5

def load_all_data(symbol: str) -> pd.DataFrame:
    # ... unverändert ...
    print(f"Lade alle 4-Stunden-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_4h WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            return df
    except Exception as e:
        print(f"Fehler beim Laden der Daten für {symbol}: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame, trend_sma_period: int) -> pd.DataFrame:
    # ... unverändert ...
    print(f"Füge Features hinzu (Trend-Periode: {trend_sma_period} * 4h)...")
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

def run_4h_backtest(symbol: str, confidence_threshold: float, trend_sma_period: int):
    # ... (Die komplette Logik dieser Funktion bleibt unverändert) ...
    print(f"\n--- Teste 4H {symbol} | Konfidenz: {confidence_threshold:.2f} | Trend: {trend_sma_period} * 4h ---")
    model_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{model_filename}_4h.pkl')
    if not os.path.exists(model_path): return None
    model = joblib.load(model_path)
    df_full = load_all_data(symbol)
    if df_full.empty: return None
    df = add_features(df_full.copy(), trend_sma_period)
    model_features = model.feature_names_in_
    X_full = df[model_features]
    probabilities = model.predict_proba(X_full)
    df['buy_proba'] = probabilities[:, np.where(model.classes_ == 1)[0][0]]
    capital = INITIAL_CAPITAL
    capital_over_time = [INITIAL_CAPITAL]
    position_open = False
    trades = []
    entry_price, take_profit_price, stop_loss_price = 0, 0, 0
    position_size_asset = 0
    for i in range(len(df)):
        if position_open:
            if df['high'].iloc[i] >= take_profit_price:
                profit_usd = (take_profit_price - entry_price) * position_size_asset; profit_usd -= (TRADE_SIZE_USD + profit_usd) * TRANSACTION_COST_PERCENT; capital += profit_usd; trades.append({'profit_usd': profit_usd}); position_open = False
            elif df['low'].iloc[i] <= stop_loss_price:
                profit_usd = (stop_loss_price - entry_price) * position_size_asset; profit_usd -= (TRADE_SIZE_USD + abs(profit_usd)) * TRANSACTION_COST_PERCENT; capital += profit_usd; trades.append({'profit_usd': profit_usd}); position_open = False
        if not position_open:
            is_uptrend = df['close'].iloc[i] > df['sma_trend'].iloc[i]
            buy_confidence = df['buy_proba'].iloc[i]
            if is_uptrend and buy_confidence > confidence_threshold:
                position_open = True; entry_price = df['close'].iloc[i]; position_size_asset = TRADE_SIZE_USD / entry_price; atr_at_entry = df['atr'].iloc[i]; take_profit_price = entry_price + (atr_at_entry * TAKE_PROFIT_ATR_MULTIPLIER); stop_loss_price = entry_price - (atr_at_entry * STOP_LOSS_ATR_MULTIPLIER)
        capital_over_time.append(capital)
    total_profit_usd = capital - INITIAL_CAPITAL
    total_return_percent = total_profit_usd / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0: return None
    winning_trades = [t for t in trades if t['profit_usd'] > 0]
    win_rate = len(winning_trades) / num_trades * 100
    returns = pd.Series(capital_over_time).pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(6 * 252) if np.std(returns) > 0 else 0
    return { 'Symbol': symbol, 'Konfidenz': f"{confidence_threshold:.2f}", 'Trend_Periode': trend_sma_period, 'Rendite_%': round(total_return_percent, 2), 'Gewinnrate_%': round(win_rate, 2), 'Sharpe_Ratio': round(sharpe_ratio, 2), 'Anzahl_Trades': num_trades, 'Endkapital_$': round(capital, 2) }

if __name__ == "__main__":
    thresholds_to_test = [0.60, 0.65, 0.70, 0.75]
    trend_periods_to_test = [20, 50, 100]
    all_results = []
    print("Starte Parameter-Optimierung für die 4-STUNDEN-Strategie...")
    for symbol in SYMBOLS:
        for threshold in thresholds_to_test:
            for period in trend_periods_to_test:
                result = run_4h_backtest(symbol, confidence_threshold=threshold, trend_sma_period=period)
                if result:
                    all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df_sorted = results_df.sort_values(by='Rendite_%', ascending=False)
        print("\n\n--- 4-STUNDEN-STRATEGIE OPTIMIERUNGS-ZUSAMMENFASSUNG ---")
        print(results_df_sorted.to_string())

        # === HIER IST DIE FEHLENDE/KORRIGIERTE LOGIK ===
        try:
            # Wichtig: Wir speichern die Ergebnisse für die 4h-Strategie in einer eigenen Datei
            results_df_sorted.to_json('backtest_results_4h.json', orient='records', indent=4)
            print("\nErfolgreich 'backtest_results_4h.json' im Projektordner gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der JSON-Datei: {e}")
        # === ENDE DER KORREKTUR ===

    else:
        print("\nKeine Trades wurden in den Tests ausgeführt.")