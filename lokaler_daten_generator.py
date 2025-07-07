# lokaler_daten_generator.py
import yfinance as yf
import os

# Definiere die Symbole, die du brauchst
SYMBOLS = {
    "BTC-USD": "BTC-USD.csv",
    "GC=F": "XAU-USD.csv"
}

# Erstelle den Ordner, falls er nicht existiert
output_dir = "data/daily"
os.makedirs(output_dir, exist_ok=True)

print("Starte den lokalen Download der Marktdaten...")

for ticker, filename in SYMBOLS.items():
    try:
        print(f"Lade Daten für {ticker}...")
        # Lade die maximalen historischen Daten
        data = yf.download(ticker, period="max", interval="1d")
        
        if data.empty:
            print(f"Keine Daten für {ticker} gefunden.")
            continue
            
        # Speichere die Daten in der CSV-Datei
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath)
        
        print(f"✅ Daten erfolgreich in '{filepath}' gespeichert.")
        
    except Exception as e:
        print(f"Ein Fehler ist bei {ticker} aufgetreten: {e}")

print("\nAlle Daten wurden erfolgreich als CSV-Dateien gespeichert!")