# run_weekly_training.py
# Dieses Skript ist der "Generalunternehmer" für unser wöchentliches Training.
# Es ruft nacheinander alle nötigen Trainer auf, um die Modelle für den Live-Betrieb zu aktualisieren.

print("==============================================")
print("=== STARTE WÖCHENTLICHEN TRAINING-ZYKLUS ===")
print("==============================================")

# Wir importieren die Hauptfunktionen aus unseren spezialisierten Trainer-Skripten
# und geben ihnen klare, verständliche Namen.
try:
    from ki_trainer_daily import run_training_pipeline as train_long_models
    from ki_trainer_short import run_training_pipeline as train_short_models
except ImportError as e:
    print(f"FEHLER: Ein Trainer-Modul konnte nicht importiert werden: {e}")
    print("Stelle sicher, dass 'ki_trainer_daily.py' und 'ki_trainer_short.py' existieren.")
    exit()

# --- Trainingsschritt 1: Long-Modelle ---
print("\n>>> TEIL 1: Trainiere die LONG-Modelle (Daily)...")
try:
    train_long_models()
    print(">>> LONG-Modell-Training ERFOLGREICH abgeschlossen.")
except Exception as e:
    print(f"!!! FEHLER beim Training der LONG-Modelle: {e} !!!")


# --- Trainingsschritt 2: Short-Modelle ---
print("\n\n>>> TEIL 2: Trainiere die SHORT-Modelle (Daily)...")
try:
    train_short_models()
    print(">>> SHORT-Modell-Training ERFOLGREICH abgeschlossen.")
except Exception as e:
    print(f"!!! FEHLER beim Training der SHORT-Modelle: {e} !!!")


print("\n\n=============================================")
print("=== WÖCHENTLICHER TRAINING-ZYKLUS BEENDET ===")
print("=============================================")