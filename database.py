# database.py (Finale, korrekte und ausführbare Version)
import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column, 
                        Integer, String, Float, DateTime, UniqueConstraint)
from datetime import datetime

# Lädt die .env Datei für lokale Tests
load_dotenv()

# --- DATENBANK-VERBINDUNG ---\
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("Keine DATABASE_URL in den Umgebungsvariablen gefunden!")

engine = create_engine(DATABASE_URL)
meta = MetaData()

# --- TABELLEN-DEFINITIONEN ---\

# Tabelle 1: Für die gesammelten Rohdaten
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

# Tabelle 2: Für die fertigen Vorhersagen unserer KI
predictions = Table(
    'predictions', meta,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(10), nullable=False, unique=True),
    Column('signal', String(10), nullable=False),
    Column('price', Float, nullable=False),
    Column('last_updated', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)


def create_tables():
    """Erstellt alle oben definierten Tabellen in der Datenbank, falls sie noch nicht existieren."""
    try:
        print("Versuche, Tabellen zu erstellen...")
        meta.create_all(engine)
        print("Tabellen erfolgreich überprüft/erstellt.")
    except Exception as e:
        print(f"Ein Fehler beim Erstellen der Tabellen ist aufgetreten: {e}")

# Dieser Block sorgt dafür, dass create_tables() aufgerufen wird,
# wenn wir 'python database.py' im Terminal ausführen.
if __name__ == '__main__':
    create_tables()