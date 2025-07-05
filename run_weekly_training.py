# run_weekly_training.py (Diamant-Version mit Auto-Backtest)
import subprocess
import sys

# --- KONFIGURATION ---
# Wir fügen den Backtester als letzten Schritt hinzu
TRAINER_SKRIPTE = [
    'ki_trainer_daily.py',
    'ki_trainer_swing.py',
    'ki_trainer_genius.py',
    'run_backtester.py' # NEU: Der Backtester läuft nach dem Training
]

def run_training_cycle():
    """
    Führt alle definierten Trainer-Skripte und den Backtester nacheinander aus.
    """
    print("==============================================")
    print("=== STARTE WÖCHENTLICHEN TRAINING-ZYKLUS ===")
    print("==============================================")

    python_executable = sys.executable

    for skript in TRAINER_SKRIPTE:
        print(f"\n--- Starte Prozess: {skript} ---")
        try:
            result = subprocess.run(
                [python_executable, skript], 
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            print(f"--- Prozess {skript} erfolgreich abgeschlossen. ---")
        except FileNotFoundError:
            print(f"FEHLER: Das Skript '{skript}' wurde nicht gefunden.")
        except subprocess.CalledProcessError as e:
            print(f"FEHLER: Das Skript '{skript}' wurde mit einem Fehler beendet.")
            print("\n--- FEHLERMELDUNG ---")
            print(e.stderr)
            print("--------------------")
        except Exception as e:
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == '__main__':
    run_training_cycle()