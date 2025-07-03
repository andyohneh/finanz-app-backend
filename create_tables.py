# create_tables.py

from database import engine, meta
from sqlalchemy import text

print("Stelle Verbindung zur Datenbank her...")
with engine.connect() as conn:
    print("Erstelle alle fehlenden Tabellen...")
    # Erstellt alle Tabellen, die in 'meta' definiert sind, aber noch nicht existieren.
    meta.create_all(conn)
    conn.commit()
    print("Fertig. Alle Tabellen sind jetzt in der Datenbank vorhanden.")