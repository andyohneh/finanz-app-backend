# backend/analyze_models.py (Finale Version mit korrekter Namens-Anzeige)
import joblib
import os
import pandas as pd
import json

# Importiere die Konfigurationen aus dem Master-Controller
from master_controller import STRATEGIES, SYMBOLS, MODELS_DIR

def analyze_feature_importance():
    """
    Lädt alle trainierten Modelle und analysiert die Wichtigkeit der Features,
    indem es die Namen aus der zugehörigen JSON-Datei liest.
    """
    print("=== STARTE FEATURE-IMPORTANCE-ANALYSE ===")
    
    all_importances = {}

    for strategy_name in STRATEGIES.keys():
        print(f"\n--- Strategie: {strategy_name.upper()} ---")
        
        strategy_importances = pd.Series(dtype=float)

        for symbol in SYMBOLS:
            base_path = f"{MODELS_DIR}/model_{strategy_name}_{symbol.replace('/', '')}"
            model_path = f"{base_path}_model.pkl"
            features_path = f"{base_path}_features.json"
            
            if not os.path.exists(model_path) or not os.path.exists(features_path):
                print(f"Modell-Dateien für {symbol} nicht gefunden, überspringe.")
                continue

            try:
                # Lade das Modell UND die Feature-Namen
                model = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    features = json.load(f)
                
                if hasattr(model, 'feature_importances_'):
                    importances = pd.Series(model.feature_importances_, index=features)
                    strategy_importances = strategy_importances.add(importances, fill_value=0)
            except Exception as e:
                print(f"Fehler beim Laden oder Analysieren von {symbol}: {e}")
        
        if not strategy_importances.empty:
            # Berechne den Durchschnitt und sortiere nach Wichtigkeit
            strategy_importances = strategy_importances / len(SYMBOLS)
            all_importances[strategy_name] = strategy_importances.sort_values(ascending=False)
            
            print("Durchschnittliche Feature-Wichtigkeit:")
            print(all_importances[strategy_name])

    print("\n=== ANALYSE ABGESCHLOSSEN ===")
    return all_importances

if __name__ == '__main__':
    analyze_feature_importance()