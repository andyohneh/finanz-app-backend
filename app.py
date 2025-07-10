# backend/app.py (Finale Version mit Admin-Reset-Route)
import os
import json
from flask import Flask, jsonify, render_template, request, send_from_directory
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

# Eigene Module importieren
from database import engine, meta # NEU: 'meta' importieren

# --- GRUNDEINSTELLUNGEN ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- ROUTEN ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    results = {'daily': [], 'swing': [], 'genius': []}
    try:
        with open('backtest_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Warnung: backtest_results.json nicht gefunden: {e}")
    return render_template('dashboard.html', results=results)

# --- API ROUTEN ---

@app.route('/api/assets')
def get_assets():
    """Holt die fertigen Live-Signale für eine gegebene Strategie."""
    strategy = request.args.get('strategy', 'daily')
    assets_data = []
    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM predictions WHERE strategy = :strategy ORDER BY symbol")
            db_results = conn.execute(query, {"strategy": strategy}).fetchall()

            for row in db_results:
                prediction = row._asdict()
                signal_text = prediction.get('signal', 'Halten')
                
                # HIER IST DIE KORRIGIERTE LOGIK
                color = "grey"
                icon = "circle"
                
                if signal_text == "Kaufen":
                    color, icon = "green", "arrow-up"
                elif signal_text == "Verkaufen":
                    color, icon = "red", "arrow-down"
                
                assets_data.append({
                    "asset": prediction.get('symbol', 'N/A').replace('/', ''),
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
        return jsonify({"error": "Konnte keine Live-Daten abrufen."}), 500

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    db_symbol = f"{symbol[:-3]}/{symbol[-3:]}"
    query = text("SELECT timestamp, open, high, low, close FROM historical_data_daily WHERE symbol = :symbol_param ORDER BY timestamp DESC LIMIT 200")
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
            result.reverse()
            data_points = [{"time": row[0].strftime('%Y-%m-%d'), "open": row[1], "high": row[2], "low": row[3], "close": row[4]} for row in result]
            return jsonify(data_points)
    except Exception as e:
        print(f"Fehler beim Laden der Chart-Daten für {db_symbol}: {e}")
        return jsonify({"error": "Konnte Chart-Daten nicht laden."}), 500

# --- NEU: ADMIN-ROUTE ZUM ZURÜCKSETZEN DER DATENBANK ---
@app.route('/admin/reset-database/<secret_key>')
def reset_database_route(secret_key):
    # Einfacher Schutz, damit nicht jeder die DB zurücksetzen kann
    if secret_key != 'Diamant777':
        return "Falscher Sicherheitsschlüssel!", 403
    
    try:
        print(">>> ADMIN-AKTION: Setze Datenbank über Web-Route zurück...")
        # Lösche alle Tabellen
        meta.drop_all(engine)
        print("Alte Tabellen erfolgreich gelöscht.")
        # Erstelle alle Tabellen mit dem neuen Schema neu
        meta.create_all(engine)
        print("✅ Neue Tabellen erfolgreich erstellt.")
        return "Datenbank wurde erfolgreich zurückgesetzt!", 200
    except Exception as e:
        print(f"FEHLER beim Zurücksetzen der Datenbank: {e}")
        return f"Ein Fehler ist aufgetreten: {e}", 500
    
    # In backend/app.py hinzufügen

@app.route('/api/equity-curves')
def get_equity_curves():
    try:
        with open('equity_curves.json', 'r', encoding='utf-8') as f:
            equity_data = json.load(f)
        return jsonify(equity_data)
    except FileNotFoundError:
        return jsonify({"error": "Equity-Daten nicht gefunden."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Main-Block ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)