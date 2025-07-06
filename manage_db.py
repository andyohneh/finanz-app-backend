# backend/manage_db.py
import argparse
from sqlalchemy import inspect
from database import engine, meta

def reset_database():
    """
    Löscht alle bekannten Tabellen und erstellt sie neu.
    Dies ist nützlich, um Schema-Änderungen anzuwenden.
    """
    print("WARNUNG: Dieser Vorgang wird alle Tabellen löschen und neu erstellen.")
    confirm = input("Bist du sicher, dass du fortfahren möchtest? (ja/nein): ")
    
    if confirm.lower() != 'ja':
        print("Aktion abgebrochen.")
        return

    try:
        print("Lösche alle bekannten Tabellen...")
        # Löscht alle Tabellen, die im 'meta'-Objekt definiert sind
        meta.drop_all(engine)
        print("Tabellen erfolgreich gelöscht.")
        
        print("Erstelle alle Tabellen mit dem neuen Schema...")
        # Erstellt alle Tabellen neu basierend auf der aktuellen Definition in database.py
        meta.create_all(engine)
        print("✅ Datenbank erfolgreich zurückgesetzt und mit neuem Schema erstellt!")
        
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

def create_tables():
    """
    Erstellt alle Tabellen, falls sie noch nicht existieren.
    """
    print("Erstelle Tabellen (falls nicht vorhanden)...")
    try:
        meta.create_all(engine)
        print("✅ Tabellen erfolgreich erstellt.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Datenbank-Verwaltungsskript.")
    parser.add_argument('action', choices=['reset', 'create'], help="Aktion, die ausgeführt werden soll: 'reset' zum Löschen und Neuerstellen, 'create' zum Erstellen.")
    
    args = parser.parse_args()

    if args.action == 'reset':
        reset_database()
    elif args.action == 'create':
        create_tables()