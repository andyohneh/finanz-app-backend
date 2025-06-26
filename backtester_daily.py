# backtester.py (Finale Version für TAGES-Daten mit sauberem Code und JSON-Export)
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
INITIAL_CAPITAL = 100
TRADE_SIZE_USD = 45
TRANSACTION_COST_PERCENT = 0.001

# Beste gefundene Parameter für die TAGES-Strategie
CONFIDENCE_THRESHOLD = 0.75
TREND_SMA_PERIOD = 150 

# Risk-Management-Parameter
TAKE_PROFIT_ATR_MULTIPLIER = 2.0
STOP_LOSS_ATR_MULTIPLIER = 1.5

def load_all_data(symbol: str) -> pd.DataFrame:
    """Lädt ALLE TAGES-Daten für ein Symbol."""
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
    """Fügt Indikatoren basierend auf TAGES-Daten hinzu."""
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

def run_daily_backtest(symbol: str):
    """Führt den finalen, realistischen Backtest für die Tages-Strategie durch."""
    print(f"\n--- Starte finalen, realistischen DAILY Backtest für {symbol} ---")

    model_filename = symbol.replace('/', '')
    model_path = os.path.join(MODEL_DIR, f'model_{model_filename}_swing.pkl')
    model = joblib.load(model_path)

    df_full = load_all_data(symbol)
    if df_full.empty: return None
    df = add_features(df_full.copy(), TREND_SMA_PERIOD)
    
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

    # === HIER IST DIE SAUBER FORMATIERTE LOGIK ===
    for i in range(len(df)):
        # Wenn eine Position offen ist, prüfe zuerst auf Ausstieg
        if position_open:
            # Take Profit Check
            if df['high'].iloc[i] >= take_profit_price:
                exit_price = take_profit_price
                profit_usd = (exit_price - entry_price) * position_size_asset
                profit_usd -= (TRADE_SIZE_USD + profit_usd) * TRANSACTION_COST_PERCENT
                capital += profit_usd
                trades.append({'profit_usd': profit_usd})
                position_open = False
            # Stop Loss Check
            elif df['low'].iloc[i] <= stop_loss_price:
                exit_price = stop_loss_price
                profit_usd = (exit_price - entry_price) * position_size_asset
                profit_usd -= (TRADE_SIZE_USD + abs(profit_usd)) * TRANSACTION_COST_PERCENT
                capital += profit_usd
                trades.append({'profit_usd': profit_usd})
                position_open = False
        
        # Wenn keine Position offen ist, prüfe auf neuen Einstieg
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
        
        capital_over_time.append(capital)
    # === ENDE DER SAUBEREN FORMATIERUNG ===

    total_profit_usd = capital - INITIAL_CAPITAL
    total_return_percent = total_profit_usd / INITIAL_CAPITAL * 100
    num_trades = len(trades)
    if num_trades == 0: return None
        
    winning_trades = [t for t in trades if t['profit_usd'] > 0]
    win_rate = len(winning_trades) / num_trades * 100
    
    returns = pd.Series(capital_over_time).pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return {
        'Symbol': symbol,
        'Konfidenz': f"{CONFIDENCE_THRESHOLD:.2f}",
        'Trend_Periode': TREND_SMA_PERIOD,
        'Rendite_%': round(total_return_percent, 2),
        'Gewinnrate_%': round(win_rate, 2),
        'Sharpe_Ratio': round(sharpe_ratio, 2),
        'Anzahl_Trades': num_trades,
        'Endkapital_$': round(capital, 2)
    }

if __name__ == "__main__":
    all_results = []
    print("Starte finale Backtests für TAGES-Strategie zum Speichern der Ergebnisse...")
    for symbol in SYMBOLS:
        result = run_daily_backtest(symbol)
        if result:
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n\n--- TAGES-STRATEGIE BACKTEST-ERGEBNISSE ---")
        print(results_df.to_string())

        try:
            # HIER DIE WICHTIGE ZEILE: Speichert unter dem korrekten Namen
            results_df.to_json('backtest_results_daily.json', orient='records', indent=4)
            print("\nErfolgreich 'backtest_results_daily.json' im Projektordner gespeichert.")
        except Exception as e:
            print(f"\nFehler beim Speichern der JSON-Datei: {e}")
    else:
        print("\nKeine Trades wurden ausgeführt.")