# database.py (Finale Diamant-Version mit allen Tabellen)
import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column, 
                        Integer, String, Float, DateTime, UniqueConstraint, Text)
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("Keine DATABASE_URL in den Umgebungsvariablen gefunden!")

engine = create_engine(DATABASE_URL)
meta = MetaData()

# Tabelle 1: Für die 1-Minuten-Daten (bleibt für die Zukunft)
historical_data = Table('historical_data', meta, 
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_symbol_timestamp')
)

# Tabelle 2: Für die Tages-Daten (unser Swing-Modell)
historical_data_daily = Table('historical_data_daily', meta, 
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_daily_symbol_timestamp')
)

# Tabelle 3: Für die 4-Stunden-Daten (unser zweites Swing-Modell)
historical_data_4h = Table(
   'historical_data_4h', meta, 
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_4h_symbol_timestamp')
)

# Tabelle 4: Für die Live-Vorhersagen der aktiven Strategie
predictions = Table('predictions', meta, 
    Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False, unique=True),
    Column('signal', String(10), nullable=False), Column('entry_price', Float, nullable=False),
    Column('take_profit', Float, nullable=True), Column('stop_loss', Float, nullable=True),
    Column('last_updated', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)

# Tabelle 5: Für die Push-Abonnements
push_subscriptions = Table(
   'push_subscriptions', meta, 
   Column('id', Integer, primary_key=True),
   Column('subscription_json', Text, nullable=False, unique=True)
)

def create_tables():
    """Erstellt alle definierten Tabellen in der Datenbank, falls sie noch nicht existieren."""
    try:
        print("Versuche, Tabellen zu erstellen oder zu aktualisieren...")
        meta.create_all(engine)
        print("Tabellen erfolgreich überprüft/erstellt.")
    except Exception as e:
        print(f"Ein Fehler beim Erstellen der Tabellen ist aufgetreten: {e}")

if __name__ == '__main__':
    create_tables()