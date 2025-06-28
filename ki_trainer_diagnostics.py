# ki_trainer_diagnostics.py (Platin-Standard: Analysiert die Feature-Wichtigkeit)
import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import text
from database import engine, historical_data_daily
import matplotlib.pyplot as plt

# --- KONFIGURATION ---
MODEL_DIR = "models"
# WICHTIG: Wir analysieren das Modell, das profitabel war!
MODEL_FILENAME = "model_BTCUSD_swing.pkl" 

def load_data_from_db(symbol: str):
    """Lädt ALLE TAGES-Daten, um das Trainings-Set nachzubauen."""
    print(f"Lade alle Tages-Daten für {symbol}...")
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp ASC")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            return df
    except Exception as e:
        return pd.DataFrame()

def run_feature_analysis():
    """Lädt das Modell und analysiert die Wichtigkeit seiner Features."""
    print(f"Analysiere Modell: {MODEL_FILENAME}")
    
    # 1. Lade das trainierte Modell
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        print("Modell nicht gefunden! Bitte stelle sicher, dass das Diamant-Modell existiert.")
        return
        
    model = joblib.load(model_path)
    
    # 2. Extrahiere die Feature-Wichtigkeit
    # Das Modell hat gelernt, welchen Features es am meisten "vertraut"
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    
    # 3. Erstelle eine saubere Übersicht
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sortiere nach Wichtigkeit
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print("\n--- Feature Importance Ranking ---")
    print("Die wichtigsten Indikatoren für die Entscheidungen der KI:")
    print(feature_importance_df)
    
    # 4. Erstelle eine visuelle Grafik
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance für das Diamant-Modell')
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Wichtigkeit')
    plt.gca().invert_yaxis() # Wichtigstes Feature oben
    plt.tight_layout()
    # Speichere die Grafik als Bild-Datei im Projektordner
    plt.savefig('feature_importance.png')
    print("\nGrafik 'feature_importance.png' wurde im Projektordner gespeichert.")
    plt.show()


if __name__ == '__main__':
    # Wir brauchen matplotlib, das müssen wir installieren
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib nicht gefunden. Bitte führe 'pip install matplotlib' in deinem Terminal aus.")
        exit()
        
    run_feature_analysis()