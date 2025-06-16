import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column, 
                        Integer, String, Float, DateTime)
from datetime import datetime



# --- DATENBANK-VERBINDUNG ---
# Diese Zeilen MÜSSEN auf der obersten Ebene stehen (nicht eingerückt).
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError("Keine DATABASE_URL in den Umgebungsvariablen gefunden!")

engine = create_engine(DATABASE_URL)
meta = MetaData()

# --- TABELLEN-DEFINITION ---
# Auch diese Definition MUSS auf der obersten Ebene stehen.
historical_data = Table(
   'historical_data', meta, 
   Column('id', Integer, primary_key=True),
   Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False, default=datetime.utcnow),
   Column('open', Float, nullable=False),
   Column('high', Float, nullable=False),
   Column('low', Float, nullable=False),
   Column('close', Float, nullable=False),
   Column('volume', Float)
)

# --- FUNKTION ZUM ERSTELLEN DER TABELLE ---
# Diese Funktion ist korrekt eingerückt.
def create_tables():
    """Erstellt alle definierten Tabellen in der Datenbank, falls sie noch nicht existieren."""
    try:
        print("Versuche, Tabellen zu erstellen...")
        meta.create_all(engine)
        print("Tabellen erfolgreich überprüft/erstellt.")
    except Exception as e:
        print(f"Ein Fehler beim Erstellen der Tabellen ist aufgetreten: {e}")

# Dieser Block wird nur ausgeführt, wenn man das Skript direkt startet.
if __name__ == "__main__":
    create_tables()