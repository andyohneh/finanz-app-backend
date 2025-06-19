# predictor.py
import os
import pandas as pd
import joblib
import ta
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, predictions
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()

# --- KONFIGURATION ---
# Die Symbole, für die Vorhersagen gemacht werden sollen.
# WICHTIG: Hier das Format für die Datenbank verwenden (z.B. 'BTC/USD')
SYMBOLS_DB = ['BTC/USD', 'XAU/USD']
# Pfad, wo die Modelle gespeichert sind
MODEL_DIR = "models"
# Wieviele Datenpunkte für die Feature-Berechnung geladen werden sollen
DATA_LIMIT_FOR_FEATURES = 200

def load_latest_data(symbol: str, limit: int = DATA_LIMIT_FOR_FEATURES) -> pd.DataFrame:
    """Lädt die neuesten Daten für ein Symbol aus der Datenbank."""
    print(f"Lade die letzten {limit} Datenpunkte für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT * FROM historical_data 
                WHERE symbol = :symbol 
                ORDER BY timestamp DESC 
                LIMIT :limit
            """)
            df = pd.read_sql_query(query, conn, params={'symbol': symbol, 'limit': limit})
            if not df.empty:
                # Daten sind absteigend sortiert, für die Feature-Berechnung müssen sie aufsteigend sein
                df = df.iloc[::-1].reset_index(drop=True)
            print(f"Erfolgreich {len(df)} Datenpunkte für {symbol} geladen.")
            return df
    except Exception as e:
        print(f"Fehler beim Laden der Daten für {symbol}: {e}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt die technischen Indikatoren als Features hinzu.
    WICHTIG: Diese Funktion muss exakt die gleichen Features wie in ki_trainer.py berechnen!
    """
    print("Füge Features zu den Daten hinzu...")
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    # ATR ist unser Schlüssel für das Risikomanagement
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # Bollinger Bänder
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df.dropna(inplace=True)
    return df

def make_and_store_predictions():
    """Hauptfunktion: Lädt Daten, Modelle und erzeugt/speichert Vorhersagen."""
    print("\n--- Starte den Vorhersage-Prozess ---")
    
    for symbol_db in SYMBOLS_DB:
        print(f"\n--- Verarbeite {symbol_db} ---")
        
        # 1. Modell laden
        symbol_filename = symbol_db.replace('/', '') # z.B. aus 'BTC/USD' wird 'BTCUSD'
        model_path = os.path.join(MODEL_DIR, f'model_{symbol_filename}.pkl')
        
        if not os.path.exists(model_path):
            print(f"FEHLER: Kein Modell unter {model_path} gefunden. Bitte zuerst ki_trainer.py ausführen.")
            continue
            
        try:
            model = joblib.load(model_path)
            print(f"Modell für {symbol_db} erfolgreich geladen.")
        except Exception as e:
            print(f"FEHLER beim Laden des Modells {model_path}: {e}")
            continue

        # 2. Daten laden und Features berechnen
        df = load_latest_data(symbol_db)
        if df.empty or len(df) < 50: # Brauchen genug Daten für Features
             print("Nicht genügend Daten für die Feature-Berechnung vorhanden.")
             continue
        
        df_features = add_features(df)
        if df_features.empty:
            print("Daten-Frame ist nach Feature-Berechnung und dropna() leer.")
            continue
            
        # 3. Vorhersage für den LETZTEN Datenpunkt treffen
        latest_data = df_features.iloc[[-1]] # Wichtig: iloc[[-1]] behält die DataFrame-Struktur
        
        # Features für das Modell auswählen (exakt wie im Training)
        feature_columns = ['sma_fast', 'sma_slow', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d']
        latest_features = latest_data[feature_columns]

        prediction_numeric = model.predict(latest_features)[0]
        
        # Numerische Vorhersage in Text umwandeln (angenommen: 1=Kaufen, -1=Verkaufen, 0=Halten)
        signal_map = {1: "Kaufen", -1: "Verkaufen", 0: "Halten"}
        signal = signal_map.get(prediction_numeric, "Unbekannt")
        print(f"Rohe Vorhersage: {prediction_numeric} -> Signal: {signal}")

        # 4. Take Profit / Stop Loss berechnen
        entry_price = latest_data['close'].iloc[0]
        latest_atr = latest_data['atr'].iloc[0]
        take_profit, stop_loss = None, None
        
        # Dies ist dein strategischer Hebel! Du kannst die Multiplikatoren anpassen.
        # Ein Risk/Reward-Ratio von 2:1 ist ein gängiger Startpunkt.
        RISK_REWARD_RATIO = 2.0
        STOP_LOSS_ATR_MULTIPLIER = 1.5

        if signal == "Kaufen":
            stop_loss = entry_price - (latest_atr * STOP_LOSS_ATR_MULTIPLIER)
            take_profit = entry_price + (latest_atr * STOP_LOSS_ATR_MULTIPLIER * RISK_REWARD_RATIO)
        elif signal == "Verkaufen":
            stop_loss = entry_price + (latest_atr * STOP_LOSS_ATR_MULTIPLIER)
            take_profit = entry_price - (latest_atr * STOP_LOSS_ATR_MULTIPLIER * RISK_REWARD_RATIO)
        
        print(f"Einstiegspreis: {entry_price:.2f}, TP: {take_profit or 'N/A'}, SL: {stop_loss or 'N/A'}")

        # --- KORREKTUR HIER ---
        # 5. Vorhersage in die Datenbank schreiben (mit Typ-Umwandlung)
        
        # Die Werte in Standard-Python-floats umwandeln, bevor wir sie übergeben
        values_to_insert = {
            "symbol": symbol_db,
            "signal": signal,
            "entry_price": float(entry_price),
            # Wichtig: Prüfen, ob TP/SL existieren, bevor wir sie umwandeln
            "take_profit": float(take_profit) if take_profit is not None else None,
            "stop_loss": float(stop_loss) if stop_loss is not None else None
        }

        insert_stmt = insert(predictions).values(values_to_insert)
        
        update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=['symbol'],
            set_={
                "signal": insert_stmt.excluded.signal,
                "entry_price": insert_stmt.excluded.entry_price,
                "take_profit": insert_stmt.excluded.take_profit,
                "stop_loss": insert_stmt.excluded.stop_loss,
                "last_updated": pd.to_datetime('now', utc=True)
            }
        )
        
        try:
            with engine.connect() as conn:
                conn.execute(update_stmt)
                conn.commit()
            print(f"Vorhersage für {symbol_db} erfolgreich in der Datenbank gespeichert/aktualisiert.")
        except Exception as e:
            print(f"FEHLER beim Speichern der Vorhersage in der DB: {e}")

    print("\n--- Vorhersage-Prozess beendet. ---")

if __name__ == "__main__":
    make_and_store_predictions()