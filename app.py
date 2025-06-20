# app.py (Version mit PWA-Unterstützung)
import os
from flask import Flask, jsonify, render_template, send_from_directory # NEU: send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from database import engine

# --- GRUNDEINSTELLUNGEN ---
load_dotenv()
# NEU: 'static_folder' hinzugefügt, damit Flask unseren Ordner mit Icons findet
app = Flask(__name__, template_folder='templates', static_folder='static') 
CORS(app)

# --- ROUTEN FÜR DAS FRONTEND ---

@app.route('/')
def index():
    """Zeigt die Hauptseite (das Dashboard) an."""
    return render_template('index.html')

# NEU: Eine Route, um die manifest.json-Datei bereitzustellen
@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(app.root_path, 'manifest.json')

@app.route('/api/assets')
def get_assets():
    """Holt die fertigen Vorhersagen direkt aus der Datenbank."""
    # Diese Funktion bleibt unverändert...
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
                assets_data.append({
                    "asset": prediction['symbol'].replace('/', ''),
                    "currentPrice": f"{prediction.get('entry_price'):.2f}" if prediction.get('entry_price') else "N/A",
                    "entry": f"{prediction.get('entry_price'):.2f}" if prediction.get('entry_price') else "N/A",
                    "takeProfit": f"{prediction.get('take_profit'):.2f}" if prediction.get('take_profit') else "N/A",
                    "stopLoss": f"{prediction.get('stop_loss'):.2f}" if prediction.get('stop_loss') else "N/A",
                    "signal": prediction.get('signal'),
                    "color": color,
                    "icon": icon
                })
        return jsonify(assets_data)
    except Exception as e:
        print(f"Fehler beim Abrufen der Vorhersagen aus der Datenbank: {e}")
        return jsonify({"error": "Konnte keine Daten von der Datenbank abrufen."}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    """Holt die historischen Daten für die Charts."""
    # Diese Funktion bleibt unverändert...
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