# backend/app.py (Finale Version mit allen Funktionen)
import os
import json
from flask import Flask, jsonify, render_template, request
from sqlalchemy import text

# Eigene Module importieren
from database import engine

# --- GRUNDEINSTELLUNGEN ---
# Wir gehen davon aus, dass die Ordner im selben Verzeichnis wie die app.py liegen
app = Flask(__name__, template_folder='templates', static_folder='static')


# --- ROUTEN ---
@app.route('/')
def index():
    """Zeigt die Hauptseite (Live-Signale) an."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Zeigt das Dashboard an und liest die Backtest-Ergebnisse."""
    results = {'daily': [], 'swing': [], 'genius': []}
    try:
        # Die Datei wird im Hauptverzeichnis des Projekts erwartet
        with open('backtest_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Warnung: backtest_results.json nicht gefunden. Führe zuerst den Backtest aus.")
    
    return render_template('dashboard.html', results=results)

# --- API ROUTEN ---
@app.route('/api/assets')
def get_assets():
    """Stellt die Live-Signale inkl. Konfidenz bereit."""
    # Wir zeigen standardmäßig die Signale unserer neuen LSTM-Strategie
    strategy = request.args.get('strategy', 'genius_lstm') 
    assets_data = []
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM predictions WHERE strategy = :strategy ORDER BY symbol")
            db_results = conn.execute(query, {"strategy": strategy}).fetchall()

            for row in db_results:
                prediction = row._asdict()
                signal_text = prediction.get('signal', 'Halten')
                
                color, icon = "grey", "circle"
                if signal_text == "Kaufen":
                    color, icon = "green", "arrow-up"
                elif signal_text == "Verkaufen":
                    color, icon = "red", "arrow-down"
                
                assets_data.append({
                    "asset": prediction.get('symbol', 'N/A').replace('/', ''),
                    # NEU: Konfidenz-Wert wird hinzugefügt
                    "confidence": f"{prediction.get('confidence', 0.0):.2f}",
                    "entry": f"{prediction.get('entry_price'):.4f}" if prediction.get('entry_price') else "N/A",
                    "takeProfit": f"{prediction.get('take_profit'):.4f}" if prediction.get('take_profit') else "N/A",
                    "stopLoss": f"{prediction.get('stop_loss'):.4f}" if prediction.get('stop_loss') else "N/A",
                    "signal": signal_text,
                    "color": color,
                    "icon": icon,
                    "timestamp": prediction.get('last_updated').strftime('%Y-%m-%d %H:%M:%S') if prediction.get('last_updated') else "N/A"
                })
        return jsonify(assets_data)
    except Exception as e:
        print(f"KRITISCHER FEHLER in /api/assets: {e}")
        return jsonify({"error": "Live-Daten konnten nicht geladen werden."}), 500

# HIER IST DIE FEHLENDE FUNKTION FÜR DIE EQUITY CURVE
@app.route('/api/equity-curves')
def get_equity_curves():
    """Liest die Equity-Kurven-Daten aus der JSON-Datei."""
    try:
        with open('equity_curves.json', 'r', encoding='utf-8') as f:
            equity_data = json.load(f)
        return jsonify(equity_data)
    except FileNotFoundError:
        return jsonify({"error": "Equity-Daten nicht gefunden. Bitte zuerst den Backtest ausführen."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    """Stellt die Chart-Daten für das Modal-Fenster bereit."""
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    query = text("SELECT timestamp, open, high, low, close FROM historical_data_daily WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 400")
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
            result.reverse()
            data_points = [{"time": row[0].strftime('%Y-%m-%d'), "open": row[1], "high": row[2], "low": row[3], "close": row[4]} for row in result]
            return jsonify(data_points)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- PWA-Routen ---
# Diese sind für die Progressive Web App Funktionalität (optional)
@app.route('/manifest.json')
def manifest():
    from flask import send_from_directory
    return send_from_directory(app.static_folder, 'manifest.json')

@app.route('/sw.js')
def sw():
    from flask import send_from_directory
    return send_from_directory(app.static_folder, 'sw.js')

# --- Main-Block ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)