# database.py (Version für die "Königsklasse")
import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column, 
                        Integer, String, Float, DateTime, UniqueConstraint)
from datetime import datetime

# Lädt die .env Datei für lokale Tests
load_dotenv()

# --- DATENBANK-VERBINDUNG ---
# Stellt sicher, dass die DATABASE_URL gesetzt ist
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("Keine DATABASE_URL in den Umgebungsvariablen gefunden!")

engine = create_engine(DATABASE_URL)
meta = MetaData()

# --- TABELLEN-DEFINITIONEN ---

# Tabelle 1: Für die gesammelten Rohdaten (bleibt unverändert)
historical_data = Table(
   'historical_data', meta, 
   Column('id', Integer, primary_key=True),
   Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False),
   Column('open', Float, nullable=False),
   Column('high', Float, nullable=False),
   Column('low', Float, nullable=False),
   Column('close', Float, nullable=False),
   Column('volume', Float),
   # Stellt sicher, dass jede Kerze (Symbol + Zeit) nur einmal existiert
   UniqueConstraint('symbol', 'timestamp', name='uq_symbol_timestamp')
)

# Tabelle 2: Für die fertigen Vorhersagen unserer KI (STARK ERWEITERT)
predictions = Table(
    'predictions', meta,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(10), nullable=False, unique=True),
    Column('signal', String(10), nullable=False), # "Kaufen", "Verkaufen" oder "Halten"
    Column('entry_price', Float, nullable=False), # Kurs bei Signalerstellung
    Column('take_profit', Float, nullable=True),  # Kursziel für Gewinn; kann NULL sein bei "Halten"
    Column('stop_loss', Float, nullable=True),    # Kursziel für Verlust; kann NULL sein bei "Halten"
    Column('last_updated', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)


def create_tables():
    """Erstellt alle oben definierten Tabellen in der Datenbank, falls sie noch nicht existieren."""
    try:
        print("Versuche, Tabellen zu erstellen oder zu aktualisieren...")
        meta.create_all(engine)
        print("Tabellen erfolgreich überprüft/erstellt.")
    except Exception as e:
        print(f"Ein Fehler beim Erstellen der Tabellen ist aufgetreten: {e}")

if __name__ == '__main__':
    # Wenn du dieses Skript direkt ausführst, werden die Tabellen erstellt.
    create_tables()