# backend/database.py (Finale Version mit Trades-Tabelle)
import os
from dotenv import load_dotenv
from sqlalchemy import (create_engine, MetaData, Table, Column,
                        Integer, String, Float, DateTime, UniqueConstraint, Text, Date)
from datetime import datetime

# --- DATENBANK-SETUP ---
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL nicht in den Umgebungsvariablen gefunden!")

engine = create_engine(DATABASE_URL)
meta = MetaData()

# --- TABELLEN-DEFINITIONEN ---

# Tabelle für historische Kursdaten
historical_data_daily = Table('historical_data_daily', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('timestamp', DateTime, nullable=False),
    Column('symbol', String(20), nullable=False),
    Column('open', Float, nullable=False),
    Column('high', Float, nullable=False),
    Column('low', Float, nullable=False),
    Column('close', Float, nullable=False),
    Column('volume', Float, nullable=False),
    Column('vix', Float, nullable=True),
    UniqueConstraint('timestamp', 'symbol', name='uq_timestamp_symbol_daily')
)

# Tabelle für die letzten KI-Vorhersagen
predictions = Table('predictions', meta,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(20), nullable=False),
    Column('strategy', String(50), nullable=False),
    Column('signal', String(10), nullable=False),
    Column('confidence', Float, nullable=True),
    Column('entry_price', Float, nullable=True),
    Column('take_profit', Float, nullable=True),
    Column('stop_loss', Float, nullable=True),
    Column('position_size', Float, nullable=True),
    Column('last_updated', DateTime, default=datetime.utcnow),
    UniqueConstraint('symbol', 'strategy', name='uq_symbol_strategy')
)

# Tabelle für Push-Benachrichtigungen
push_subscriptions = Table('push_subscriptions', meta,
   Column('id', Integer, primary_key=True),
   Column('subscription_json', Text, unique=True, nullable=False)
)

# Tabelle für die täglichen Sentiment-Scores
daily_sentiment = Table('daily_sentiment', meta,
    Column('id', Integer, primary_key=True),
    Column('asset', String(20), nullable=False),
    Column('date', Date, nullable=False),
    Column('sentiment_score', Float, nullable=False),
    UniqueConstraint('asset', 'date', name='uq_asset_date_sentiment')
)

# NEU: Das Orderbuch für unser Paper-Trading-Portfolio
trades = Table('trades', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('symbol', String(20), nullable=False),
    Column('strategy', String(50), nullable=False),
    Column('status', String(10), nullable=False, default='OPEN'), # z.B. OPEN, CLOSED
    Column('direction', String(10), nullable=False), # KAUFEN oder VERKAUFEN
    Column('entry_price', Float, nullable=False),
    Column('exit_price', Float, nullable=True),
    Column('size', Float, nullable=False),
    Column('pnl_pct', Float, nullable=True), # Profit & Loss in Prozent
    Column('entry_time', DateTime, default=datetime.utcnow),
    Column('exit_time', DateTime, nullable=True)
)