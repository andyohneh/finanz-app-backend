# database.py (Finale, aufgeräumte Version)
import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column, 
                        Integer, String, Float, DateTime, UniqueConstraint, Text)
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
meta = MetaData()

# Nur die wirklich genutzten Tabellen
historical_data_daily = Table(
   'historical_data_daily', meta, 
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_daily_symbol_timestamp')
)
predictions = Table('predictions', meta, 
    Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False, unique=True),
    Column('signal', String(10), nullable=False), Column('entry_price', Float, nullable=False),
    Column('take_profit', Float, nullable=True), Column('stop_loss', Float, nullable=True),
    Column('last_updated', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
)
push_subscriptions = Table(
   'push_subscriptions', meta, 
   Column('id', Integer, primary_key=True),
   Column('subscription_json', Text, nullable=False, unique=True)
)

def create_tables():
    meta.create_all(engine)

if __name__ == '__main__':
    create_tables()