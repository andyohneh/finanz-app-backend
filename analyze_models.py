# backend/analyze_models.py
import joblib
import os
import pandas as pd

# Importiere die Konfiguration, damit wir wissen, welche Modelle wir haben
from master_controller import STRATEGIES, SYMBOLS, MODELS_DIR

def analyze_feature_importance():
    """
    Lädt alle trainierten Modelle und analysiert die Wichtigkeit der Features.
    """
    print("=== STARTE FEATURE-IMPORTANCE-ANALYSE ===")
    
    all_importances = {}

    for strategy_name in STRATEGIES.keys():
        print(f"\n--- Strategie: {strategy_name.upper()} ---")
        
        # Wir sammeln die Feature-Wichtigkeiten über beide Symbole
        strategy_importances = pd.Series(dtype=float)

        for symbol in SYMBOLS:
            model_path = f"{MODELS_DIR}/model_{strategy_name}_{symbol.replace('/', '')}_model.pkl"
            
            if not os.path.exists(model_path):
                print(f"Modell für {symbol} nicht gefunden, überspringe.")
                continue

            # Lade das trainierte Modell
            model = joblib.load(model_path)
            
            # Hole die Feature-Namen und ihre Wichtigkeits-Werte
            # Der hasattr-Check ist eine Sicherheitsmaßnahme
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=model.feature_name_)
                strategy_importances = strategy_importances.add(importances, fill_value=0)
        
        if not strategy_importances.empty:
            # Berechne den Durchschnitt und sortiere
            strategy_importances = strategy_importances / len(SYMBOLS)
            all_importances[strategy_name] = strategy_importances.sort_values(ascending=False)
            
            print("Durchschnittliche Feature-Wichtigkeit (Top 5):")
            print(all_importances[strategy_name].head())

    print("\n=== ANALYSE ABGESCHLOSSEN ===")
    return all_importances

if __name__ == '__main__':
    analyze_feature_importance()