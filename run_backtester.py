# run_backtester.py
import pandas as pd
import numpy as np
import joblib
import json
from sqlalchemy import text
from database import engine

# Importiere die Feature-Funktionen aus den jeweiligen Trainern
from ki_trainer_daily import add_features as add_daily_features
from ki_trainer_swing import add_features as add_swing_features
from ki_trainer_genius import add_features as add_genius_features

# --- KONFIGURATION ---
INITIAL_CAPITAL = 100
SYMBOLS = ['BTC/USD', 'XAU/USD']
STRATEGIES = {
    'daily': {
        'model_prefix': 'model_daily_',
        'add_features': add_daily_features,
        'features': ['sma_fast', 'sma_slow', 'rsi', 'macd_diff', 'bb_width', 'stoch_k', 'roc', 'atr', 'adx', 'cci', 'sentiment_score'],
        'tp_factor': 2.0,
        'sl_factor': 1.0
    },
    'swing': {
        'model_prefix': 'model_swing_',
        'add_features': add_swing_features,
        'features': ['sma_fast', 'sma_slow', 'rsi', 'macd_diff', 'atr', 'sentiment_score'],
        'tp_factor': 2.5,
        'sl_factor': 1.5
    },
    'genius': {
        'model_prefix': 'model_genius_',
        'add_features': add_genius_features,
        'features': ['sma_fast', 'sma_slow', 'rsi', 'macd_diff', 'atr', 'sentiment_score'],
        'tp_factor': 2.5,
        'sl_factor': 1.2
    }
}

def load_data_from_db(symbol: str) -> pd.DataFrame:
    """Lädt historische Preisdaten UND die dazugehörigen Sentiment-Scores."""
    with engine.connect() as conn:
        query = text("""
            SELECT h.*, s.sentiment_score
            FROM historical_data_daily h
            LEFT JOIN daily_sentiment s ON h.symbol = s.asset AND DATE(h.timestamp) = DATE(s.date)
            WHERE h.symbol = :symbol ORDER BY h.timestamp ASC
        """)
        df = pd.read_sql_query(query, conn, params={'symbol': symbol})
        df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

def run_backtest(symbol, strategy_name):
    print(f"Starte Backtest für {symbol} mit Strategie '{strategy_name}'...")
    
    config = STRATEGIES[strategy_name]
    model_path = f"models/{config['model_prefix']}{symbol.replace('/', '')}.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Modell {model_path} nicht gefunden. Überspringe Backtest.")
        return None

    df = load_data_from_db(symbol)
    df_featured = config['add_features'](df.copy())
    
    capital = INITIAL_CAPITAL
    trades = []
    position = None

    for i in range(len(df_featured)):
        current_row = df_featured.iloc[i]
        
        if position is None: # Keine offene Position, suche nach Einstieg
            X_live = pd.DataFrame([current_row[config['features']]])
            prediction = model.predict(X_live)[0]
            
            if prediction != 0: # Einstiegssignal
                position = {'type': prediction, 'entry_price': current_row['close'], 'entry_date': current_row['timestamp']}
                if prediction == 1: # Long
                    position['stop_loss'] = current_row['close'] - (current_row['atr'] * config['sl_factor'])
                    position['take_profit'] = current_row['close'] + (current_row['atr'] * config['tp_factor'])
                else: # Short
                    position['stop_loss'] = current_row['close'] + (current_row['atr'] * config['sl_factor'])
                    position['take_profit'] = current_row['close'] - (current_row['atr'] * config['tp_factor'])

        else: # Offene Position, prüfe auf Ausstieg
            exit_reason = None
            if position['type'] == 1: # Long
                if current_row['low'] <= position['stop_loss']: exit_reason = 'Stop Loss'
                elif current_row['high'] >= position['take_profit']: exit_reason = 'Take Profit'
            else: # Short
                if current_row['high'] >= position['stop_loss']: exit_reason = 'Stop Loss'
                elif current_row['low'] <= position['take_profit']: exit_reason = 'Take Profit'
            
            if exit_reason:
                exit_price = position['stop_loss'] if 'Stop Loss' in exit_reason else position['take_profit']
                profit_percent = ((exit_price - position['entry_price']) / position['entry_price']) * position['type']
                trades.append({'profit_percent': profit_percent})
                capital *= (1 + profit_percent)
                position = None

    # Performance-Metriken berechnen
    num_trades = len(trades)
    if num_trades == 0: return {'Symbol': symbol, 'Gesamtrendite_%': 0, 'Gewinnrate_%': 0, 'Anzahl_Trades': 0}

    win_rate = len([t for t in trades if t['profit_percent'] > 0]) / num_trades * 100
    total_return = (capital / INITIAL_CAPITAL - 1) * 100

    return {
        'Symbol': symbol,
        'Gesamtrendite_%': round(total_return, 2),
        'Gewinnrate_%': round(win_rate, 2),
        'Anzahl_Trades': num_trades
    }

if __name__ == "__main__":
    final_results = {}
    for strategy in STRATEGIES.keys():
        strategy_results = []
        for symbol in SYMBOLS:
            result = run_backtest(symbol, strategy)
            if result:
                strategy_results.append(result)
        final_results[strategy] = strategy_results

    # Speichere die Ergebnisse in einer einzigen JSON-Datei
    with open('backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    print("\nBacktest abgeschlossen. 'backtest_results.json' wurde erfolgreich erstellt.")
    print("Dein Dashboard hat jetzt wieder Daten!")