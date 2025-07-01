# app.py (Finale "Allwetter"-Version, perfektioniert)
import os
import json
from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from database import engine, push_subscriptions, historical_data_daily

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
    """Liest die Backtest-Ergebnisse der finalen Long- und Short-Tagesstrategien."""
    results = {'long': [], 'short': []}

    # Lade Ergebnisse der Long-Strategie
    try:
        # WICHTIG: Stellt sicher, dass diese Datei von backtester_daily.py erstellt wurde
        with open('backtest_results_daily.json', 'r', encoding='utf-8') as f:
            results['long'] = json.load(f)
    except FileNotFoundError:
        print("Warnung: backtest_results_daily.json nicht gefunden.")
    except Exception as e:
        print(f"Fehler beim Laden von daily_long.json: {e}")

    # Lade Ergebnisse der Short-Strategie
    try:
        # WICHTIG: Stellt sicher, dass diese Datei von backtester_short.py erstellt wurde
        with open('backtest_results_daily_short.json', 'r', encoding='utf-8') as f:
            results['short'] = json.load(f)
    except FileNotFoundError:
        print("Warnung: backtest_results_daily_short.json nicht gefunden.")
    except Exception as e:
        print(f"Fehler beim Laden von daily_short.json: {e}")

    return render_template('dashboard.html', results=results)

# PWA-Routen
@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(app.root_path, 'manifest.json')

@app.route('/sw.js')
def serve_sw():
    return send_from_directory(app.static_folder, 'sw.js')

# --- API ROUTEN ---

@app.route('/api/save-subscription', methods=['POST'])
def save_subscription():
    """Empfängt ein Push-Abonnement und speichert es in der Datenbank."""
    subscription_data = request.json
    if not subscription_data:
        return jsonify({'success': False, 'error': 'Keine Daten erhalten'}), 400
    try:
        with engine.connect() as conn:
            sub_json_string = json.dumps(subscription_data)
            stmt = insert(push_subscriptions).values(subscription_json=sub_json_string)
            stmt = stmt.on_conflict_do_nothing(index_elements=['subscription_json'])
            conn.execute(stmt)
            conn.commit()
            return jsonify({'success': True}), 201
    except Exception as e:
        print(f"Fehler beim Speichern des Abonnements: {e}")
        return jsonify({'success': False, 'error': 'Interner Serverfehler'}), 500

@app.route('/api/assets')
def get_assets():
    """Holt die fertigen Live-Signale aus der predictions-Tabelle."""
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
                    "entry": f"{prediction.get('entry_price'):.2f}" if prediction.get('entry_price') else "N/A",
                    "takeProfit": f"{prediction.get('take_profit'):.2f}" if prediction.get('take_profit') else "N/A",
                    "stopLoss": f"{prediction.get('stop_loss'):.2f}" if prediction.get('stop_loss') else "N/A",
                    "signal": prediction.get('signal'),
                    "color": color,
                    "icon": icon,
                    "timestamp": prediction['last_updated'].strftime('%Y-%m-%d %H:%M:%S')
                })
        return jsonify(assets_data)
    except Exception as e:
        print(f"Fehler in /api/assets: {e}")
        return jsonify({"error": "Konnte keine Live-Daten abrufen."}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    """Holt die historischen TAGES-Daten für die Charts."""
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    query = text("SELECT timestamp, close FROM historical_data_daily WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 30")
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
        result.reverse()
        labels = [row[0].strftime('%Y-%m-%d') for row in result]
        data_points = [row[1] for row in result]
        return jsonify({"labels": labels, "data": data_points})
    except Exception as e:
        print(f"Fehler beim Laden der Chart-Daten für {db_symbol}: {e}")
        return jsonify({"error": "Konnte Chart-Daten nicht laden."}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
