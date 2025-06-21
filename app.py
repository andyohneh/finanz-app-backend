# app.py (FINALE PWA-VERSION)
import os
import json # NEU: Importieren, um JSON-Dateien zu lesen
from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from database import engine

load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static') 
CORS(app)

# Routen
@app.route('/')
def index():
    return render_template('index.html')

# === NEUE ROUTE FÜR DAS DASHBOARD ===
@app.route('/dashboard')
def dashboard():
    """Liest die Backtest-Ergebnisse und zeigt sie auf einer neuen Seite an."""
    try:
        # Öffne die JSON-Datei mit den Ergebnissen
        with open('backtest_results.json', 'r') as f:
            results_data = json.load(f)
        print("Backtest-Ergebnisse erfolgreich geladen.")
    except FileNotFoundError:
        print("FEHLER: backtest_results.json nicht gefunden.")
        results_data = [] # Leere Liste, wenn die Datei nicht existiert
    except Exception as e:
        print(f"Ein Fehler beim Laden der Backtest-Ergebnisse ist aufgetreten: {e}")
        results_data = []
    
    # Übergib die Daten an das neue 'dashboard.html'-Template
    return render_template('dashboard.html', results=results_data)
# === ENDE NEUE ROUTE ===

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(app.root_path, 'manifest.json')

# NEU: Route für den Service Worker
@app.route('/sw.js')
def serve_sw():
    return send_from_directory(app.static_folder, 'sw.js')

@app.route('/api/assets')
def get_assets():
    print("API-Aufruf /api/assets: Lese fertige Signale aus der Datenbank.")
    assets_data = []
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM predictions ORDER BY symbol")
            results = conn.execute(query).fetchall()
            for row in results:
                prediction = row._asdict()
                color, icon = "grey", "circle"
                if prediction.get('signal') == "Kaufen":
                    color, icon = "green", "arrow-up"
                elif prediction.get('signal') == "Verkaufen":
                    color, icon = "red", "arrow-down"
                assets_data.append({ "asset": prediction['symbol'].replace('/', ''), "entry": f"{prediction.get('entry_price'):.2f}" if prediction.get('entry_price') else "N/A", "takeProfit": f"{prediction.get('take_profit'):.2f}" if prediction.get('take_profit') else "N/A", "stopLoss": f"{prediction.get('stop_loss'):.2f}" if prediction.get('stop_loss') else "N/A", "signal": prediction.get('signal'), "color": color, "icon": icon })
        return jsonify(assets_data)
    except Exception as e:
        print(f"Fehler beim Abrufen der Vorhersagen aus der Datenbank: {e}")
        return jsonify({"error": "Konnte keine Daten von der Datenbank abrufen."}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    query = text("SELECT timestamp, close FROM historical_data WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 200")
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
        result.reverse()
        labels = [row[0].strftime('%H:%M') for row in result]
        data_points = [row[1] for row in result]
        return jsonify({"labels": labels, "data": data_points})
    except Exception as e:
        return jsonify({"error": "Konnte Chart-Daten nicht laden."}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)