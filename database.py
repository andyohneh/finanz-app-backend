# database.py (Finale, aufgeräumte Version + Sentiment-Tabelle)
import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column, 
                        Integer, String, Float, DateTime, UniqueConstraint, Text)
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
meta = MetaData()

# Bestehende Tabelle für historische Daten
historical_data_daily = Table(
   'historical_data_daily', meta, 
   Column('id', Integer, primary_key=True), Column('symbol', String(10), nullable=False),
   Column('timestamp', DateTime, nullable=False), Column('open', Float, nullable=False),
   Column('high', Float, nullable=False), Column('low', Float, nullable=False),
   Column('close', Float, nullable=False), Column('volume', Float),
   UniqueConstraint('symbol', 'timestamp', name='uq_daily_symbol_timestamp')
)

# Bestehende Tabelle für die letzten Vorhersagen
predictions = Table('predictions', meta,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(10), nullable=False),
    Column('strategy', String(50), nullable=False), # NEUE SPALTE
    Column('signal', String(10), nullable=False),
    Column('entry_price', Float, nullable=False),
    Column('take_profit', Float, nullable=True),
    Column('stop_loss', Float, nullable=True),
    Column('last_updated', DateTime, default=datetime.utcnow),
    UniqueConstraint('symbol', 'strategy', name='uq_symbol_strategy') # NEUER UNIQUE KEY
)

# Bestehende Tabelle für Push-Benachrichtigungen
push_subscriptions = Table(
   'push_subscriptions', meta, 
   Column('id', Integer, primary_key=True),
   Column('subscription_json', Text, nullable=False)
)

# NEUE Tabelle für die täglichen Sentiment-Scores
daily_sentiment = Table(
   'daily_sentiment', meta, 
   Column('id', Integer, primary_key=True),
   Column('asset', String(10), nullable=False),
   Column('date', DateTime, nullable=False),
   Column('sentiment_score', Float, nullable=False),
   UniqueConstraint('asset', 'date', name='uq_sentiment_asset_date')
)