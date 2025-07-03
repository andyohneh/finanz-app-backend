# run_weekly_training.py (Korrigierte und verbesserte Version)
import subprocess
import sys

# --- KONFIGURATION ---
# Wir listen hier alle unsere spezialisierten Trainer auf.
# WICHTIG: Wir rufen jetzt den neuen 'swing' Trainer auf!
TRAINER_SKRIPTE = [
    'ki_trainer_daily.py',
    'ki_trainer_swing.py', # ERSETZT 'ki_trainer_short.py'
    'ki_trainer_genius.py'
]

def run_training_cycle():
    """
    Führt alle definierten Trainer-Skripte nacheinander aus.
    """
    print("==============================================")
    print("=== STARTE WÖCHENTLICHEN TRAINING-ZYKLUS ===")
    print("==============================================")

    # Wir benutzen den gleichen Python-Interpreter, mit dem dieses Skript gestartet wurde.
    python_executable = sys.executable

    for skript in TRAINER_SKRIPTE:
        print(f"\n--- Starte Trainer: {skript} ---")
        try:
            # Führe das Skript als separaten Prozess aus.
            # Das ist robust und stellt sicher, dass jedes Skript frisch startet.
            result = subprocess.run(
                [python_executable, skript], 
                check=True,         # Wirft einen Fehler, wenn das Skript fehlschlägt
                capture_output=True, # Fängt die Ausgabe des Skripts auf
                text=True           # Kodiert die Ausgabe als Text
            )
            # Gib die Ausgabe des erfolgreichen Skripts aus
            print(result.stdout)
            print(f"--- Trainer {skript} erfolgreich abgeschlossen. ---")
        except FileNotFoundError:
            print(f"FEHLER: Das Skript '{skript}' wurde nicht gefunden.")
        except subprocess.CalledProcessError as e:
            # Wenn das Skript einen Fehler hat, geben wir die Fehlermeldung aus
            print(f"FEHLER: Das Skript '{skript}' wurde mit einem Fehler beendet.")
            print("\n--- FEHLERMELDUNG ---")
            print(e.stderr)
            print("--------------------")
        except Exception as e:
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == '__main__':
    run_training_cycle()