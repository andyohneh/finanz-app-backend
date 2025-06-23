# app.py (Finale Diamant-Version 1.2 - mit Chart-Zoom)
import os
import json
from flask import Flask, jsonify, render_template, send_from_directory 
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from database import engine

load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static') 
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    results = {'daily': [], 'four_hour': []}
    try:
        with open('backtest_results_daily.json', 'r', encoding='utf-8') as f:
            results['daily'] = json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden von backtest_results_daily.json: {e}")
    try:
        with open('backtest_results_4h.json', 'r', encoding='utf-8') as f:
            results['four_hour'] = json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden von backtest_results_4h.json: {e}")
    return render_template('dashboard.html', results=results)

@app.route('/manifest.json')
def serve_manifest(): return send_from_directory(app.root_path, 'manifest.json')

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
                if prediction.get('signal') == "Kaufen": color, icon = "green", "arrow-up"
                assets_data.append({ 
                    "asset": prediction['symbol'].replace('/', ''), 
                    "entry": f"{prediction.get('entry_price'):.2f}" if prediction.get('entry_price') else "N/A", 
                    "takeProfit": f"{prediction.get('take_profit'):.2f}" if prediction.get('take_profit') else "N/A", 
                    "stopLoss": f"{prediction.get('stop_loss'):.2f}" if prediction.get('stop_loss') else "N/A", 
                    "signal": prediction.get('signal'), "color": color, "icon": icon,
                    "timestamp": prediction['last_updated'].strftime('%Y-%m-%d %H:%M:%S')
                })
        return jsonify(assets_data)
    except Exception as e:
        print(f"Fehler in /api/assets: {e}")
        return jsonify({"error": "Konnte keine Live-Daten abrufen."}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    """Holt die historischen 4H-Daten für die Charts."""
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    
    # === HIER IST DIE FINALE SCHÖNHEITSREPARATUR ===
    # Wir holen jetzt nur noch die letzten 60 Datenpunkte (entspricht 10 Tagen bei 4h-Intervall)
    query = text("SELECT timestamp, close FROM historical_data_4h WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 60")
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
        result.reverse()
        labels = [row[0].strftime('%Y-%m-%d %H:%M') for row in result]
        data_points = [row[1] for row in result]
        return jsonify({"labels": labels, "data": data_points})
    except Exception as e:
        print(f"Fehler beim Laden der Chart-Daten für {db_symbol}: {e}")
        return jsonify({"error": "Konnte Chart-Daten nicht laden."}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)