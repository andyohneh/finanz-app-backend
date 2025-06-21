# app.py (Finale Version mit eingebettetem Analyse-Dashboard)
import os
import json
from flask import Flask, jsonify, render_template, send_from_directory 
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from database import engine

# --- GRUNDEINSTELLUNGEN ---
load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static') 
CORS(app)

# --- ROUTEN ---

@app.route('/')
def index():
    """Zeigt die Hauptseite (Live-Signale) an."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Zeigt die Backtest-Ergebnisse an. Die Daten sind jetzt fest im Code verankert."""
    
    # Die Daten aus unserem erfolgreichen Backtest, direkt hier als Python-Dictionary.
    results_data = [
        {
            "Symbol": "BTC/USD",
            "Gesamtrendite_%": 479.19,
            "Gewinnrate_%": 76.19,
            "Sharpe_Ratio": 1.77,
            "Anzahl_Trades": 147,
            "Endkapital_$": 579.19
        },
        {
            "Symbol": "XAU/USD",
            "Gesamtrendite_%": 157.84,
            "Gewinnrate_%": 76.92,
            "Sharpe_Ratio": 1.61,
            "Anzahl_Trades": 130,
            "Endkapital_$": 257.84
        }
    ]
    print("Backtest-Ergebnisse werden aus dem Code geladen und gerendert.")
    
    # Die Daten werden wie zuvor an das Template übergeben.
    return render_template('dashboard.html', results=results_data)

# ... Der Rest der app.py (PWA-Routen, API-Routen etc.) bleibt unverändert ...

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(app.root_path, 'manifest.json')

@app.route('/sw.js')
def serve_sw():
    return send_from_directory(app.static_folder, 'sw.js')

@app.route('/api/assets')
def get_assets():
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
                assets_data.append({ "asset": prediction['symbol'].replace('/', ''), "entry": f"{prediction.get('entry_price'):.2f}" if prediction.get('entry_price') else "N/A", "takeProfit": f"{prediction.get('take_profit'):.2f}" if prediction.get('take_profit') else "N/A", "stopLoss": f"{prediction.get('stop_loss'):.2f}" if prediction.get('stop_loss') else "N/A", "signal": prediction.get('signal'), "color": color, "icon": icon })
        return jsonify(assets_data)
    except Exception as e:
        return jsonify({"error": "Konnte keine Live-Daten abrufen."}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    query = text("SELECT timestamp, close FROM historical_data_daily WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 365")
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
        result.reverse()
        labels = [row[0].strftime('%Y-%m-%d') for row in result]
        data_points = [row[1] for row in result]
        return jsonify({"labels": labels, "data": data_points})
    except Exception as e:
        return jsonify({"error": "Konnte Chart-Daten nicht laden."}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)