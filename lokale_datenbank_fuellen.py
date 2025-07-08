# lokale_datenbank_fuellen.py (Die endgültige, korrekte Lösung)
import yfinance as yf
import pandas as pd
import os
from sqlalchemy import create_engine, text

# --- DEINE DATENBANK-ADRESSE ---
# FÜGE HIER DEINEN "EXTERNAL CONNECTION STRING" VON RENDER EIN
DATABASE_URL = "postgresql://krypto_settings_db_zzf3_user:aRUhOfiB1jiUV4ASBFkSYJIJdFXvmV2s@dpg-d17t3hruibrs7383vov0-a.frankfurt-postgres.render.com/krypto_settings_db_zzf3"

# --- SYMBOLE, DIE WIR LADEN WOLLEN ---
SYMBOLS_TO_FETCH = {
    "BTC-USD": "BTC/USD",
    "GC=F": "XAU/USD"
}

def fill_database_from_local():
    """
    Verbindet sich von deinem Computer zur Render-Datenbank, lädt die Daten
    und konvertiert die Spaltennamen korrekt, bevor sie eingefügt werden.
    """
    print("Verbinde mit der externen Datenbank (finaler Versuch)...")
    if DATABASE_URL == "DEIN_EXTERNAL_CONNECTION_STRING_HIER":
        print("FEHLER: Bitte trage zuerst deine Datenbank-URL in das Skript ein.")
        return
        
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            print("✅ Erfolgreich mit der Datenbank verbunden.")
            
            for ticker, db_symbol in SYMBOLS_TO_FETCH.items():
                print(f"\n--- Verarbeite Symbol: {ticker} ---")
                try:
                    data = yf.download(ticker, period="max", interval="1d", progress=False)
                    if data.empty:
                        continue
                    
                    # === DIE FINALE, ENTSCHEIDENDE KORREKTUR ===
                    # Wenn die Spalten eine komplexe Struktur haben (MultiIndex),
                    # wandeln wir sie in eine einfache Struktur um.
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    # Jetzt, wo die Spaltennamen einfach sind, können wir sicher arbeiten.
                    data.reset_index(inplace=True)
                    
                    data.rename(columns={
                        'Date': 'timestamp', 'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    }, inplace=True)
                    
                    data['symbol'] = db_symbol
                    
                    required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                    final_data = data[required_columns].dropna()
                    
                    records = final_data.to_dict(orient='records')
                    if not records:
                        continue
                        
                    print(f"Füge {len(records)} Datensätze für {db_symbol} in die Datenbank ein...")
                    
                    trans = conn.begin()
                    for record in records:
                        stmt = text("""
                            INSERT INTO historical_data_daily (timestamp, symbol, open, high, low, close, volume)
                            VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume)
                            ON CONFLICT (timestamp, symbol) DO NOTHING
                        """)
                        conn.execute(stmt, record)
                    trans.commit()
                    print(f"Daten für {db_symbol} erfolgreich importiert.")
                    
                except Exception as e_inner:
                    # Bei einem Fehler innerhalb der Schleife, rolle die Transaktion zurück
                    if 'trans' in locals() and trans.is_active:
                        trans.rollback()
                    print(f"Ein FEHLER ist bei {ticker} aufgetreten: {e_inner}")

    except Exception as e_outer:
        print(f"Ein FEHLER bei der Datenbankverbindung ist aufgetreten: {e_outer}")

    print("\n✅ DATEN-TANKSTELLE ABGESCHLOSSEN!")

if __name__ == "__main__":
    fill_database_from_local()