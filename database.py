# database.py (mit 4h-Tabelle)
import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column, 
                        Integer, String, Float, DateTime, UniqueConstraint)
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("Keine DATABASE_URL in den Umgebungsvariablen gefunden!")

engine = create_engine(DATABASE_URL)
meta = MetaData()

# Tabelle für die 1-Minuten-Daten (bleibt für die Zukunft)
historical_data = Table('historical_data', meta, #... (Inhalt wie bisher)
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_symbol_timestamp')
)
# Tabelle für die Tages-Daten (unser Swing-Modell)
historical_data_daily = Table('historical_data_daily', meta, #... (Inhalt wie bisher)
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_daily_symbol_timestamp')
)
# NEUE TABELLE 3: Für die 4-Stunden-Daten
historical_data_4h = Table(
   'historical_data_4h', meta, 
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_4h_symbol_timestamp')
)
# Tabelle für die Live-Vorhersagen (unverändert)
predictions = Table('predictions', meta, #... (Inhalt wie bisher)
    Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False, unique=True),
    Column('signal', String(10), nullable=False), Column('entry_price', Float, nullable=False),
    Column('take_profit', Float, nullable=True), Column('stop_loss', Float, nullable=True),
    Column('last_updated', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)

def create_tables():
    try:
        print("Versuche, Tabellen zu erstellen oder zu aktualisieren...")
        meta.create_all(engine)
        print("Tabellen erfolgreich überprüft/erstellt.")
    except Exception as e:
        print(f"Ein Fehler beim Erstellen der Tabellen ist aufgetreten: {e}")

if __name__ == '__main__':
    create_tables()