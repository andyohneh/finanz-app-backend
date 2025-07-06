# backend/app.py (Finale, saubere Version)
import os
import json
from flask import Flask, jsonify, render_template, request, send_from_directory
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

# Eigene Module importieren
from database import engine, push_subscriptions

# --- GRUNDEINSTELLUNGEN ---
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# --- ROUTEN ---

@app.route('/')
def index():
    """Zeigt die Hauptseite (Live-Signale & Cockpit) an."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Liest die Backtest-Ergebnisse und zeigt sie auf der Dashboard-Seite an."""
    results = {'daily': [], 'swing': [], 'genius': []}
    try:
        # Lädt die Ergebnisse aus der JSON-Datei, die vom Backtester erstellt wird
        with open('backtest_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Warnung: backtest_results.json nicht gefunden: {e}")

    return render_template('dashboard.html', results=results)

# --- API & PWA ROUTEN ---

@app.route('/api/assets')
def get_assets():
    """Holt die fertigen Live-Signale aus der 'predictions'-Tabelle."""
    assets_data = []
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM predictions ORDER BY symbol")
            db_results = conn.execute(query).fetchall()

            for row in db_results:
                prediction = row._asdict()
                color, icon = "grey", "circle"
                if prediction.get('signal') == "Kaufen":
                    color, icon = "green", "arrow-up"
                elif prediction.get('signal') == "Verkaufen":
                    color, icon = "red", "arrow-down"
                
                assets_data.append({
                    "asset": prediction.get('symbol', 'N/A').replace('/', ''),
                    "entry": f"{prediction.get('entry_price'):.4f}" if prediction.get('entry_price') else "N/A",
                    "takeProfit": f"{prediction.get('take_profit'):.4f}" if prediction.get('take_profit') else "N/A",
                    "stopLoss": f"{prediction.get('stop_loss'):.4f}" if prediction.get('stop_loss') else "N/A",
                    "signal": prediction.get('signal'),
                    "color": color,
                    "icon": icon,
                    "timestamp": prediction.get('last_updated').strftime('%Y-%m-%d %H:%M:%S') if prediction.get('last_updated') else "N/A"
                })
        return jsonify(assets_data)
    except Exception as e:
        print(f"KRITISCHER FEHLER in /api/assets: {e}")
        return jsonify({"error": "Konnte keine Live-Daten abrufen."}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    """Holt die aktuellen, historischen TÄGLICHEN Daten für die Lightweight Charts."""
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    query = text("SELECT timestamp, open, high, low, close FROM historical_data_daily WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 200")
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
            result.reverse()
            
            data_points = [
                {"time": row[0].strftime('%Y-%m-%d'), "open": row[1], "high": row[2], "low": row[3], "close": row[4]}
                for row in result
            ]
            return jsonify(data_points)
    except Exception as e:
        print(f"Fehler beim Laden der Chart-Daten für {db_symbol}: {e}")
        return jsonify({"error": "Konnte Chart-Daten nicht laden."}), 500

# --- PWA-spezifische Routen ---
@app.route('/manifest.json')
def manifest():
    return send_from_directory(app.static_folder, 'manifest.json')

@app.route('/sw.js')
def sw():
    return send_from_directory(app.static_folder, 'sw.js')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico')

# --- Main-Block ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)