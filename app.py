# app.py - KORRIGIERTER IMPORT-BEREICH

import os
import json
# FLASK-IMPORTE: send_from_directory hinzugefügt
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
# SQLALCHEMY-IMPORTE: 'insert' hinzugefügt
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from datetime import datetime, timedelta

# Eigene Module importieren
from database import engine, push_subscriptions, historical_data_daily
import predictor_daily
import predictor_swing
import predictor_genius

# --- GRUNDEINSTELLUNGEN ---
load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- ROUTEN ---

@app.route('/')
def index():
    """Zeigt die Hauptseite (Live-Signale) an."""
    return render_template('index.html')

@app.route('/api/assets')
def get_assets():
    """
    NEU: Ruft die Vorhersagen für alle drei Strategien ab und kombiniert sie.
    """
    assets_data = []
    symbols = ['BTC/USD', 'XAU/USD']
    
    # Lade die letzten 200 Tage an Daten als Basis für die Feature-Berechnung
    with engine.connect() as conn:
        for symbol in symbols:
            print(f"Lade Daten für {symbol} für alle Predictoren...")
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 200")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            if df.empty or len(df) < 60: # Brauchen genug Daten für Indikatoren
                print(f"Nicht genügend historische Daten für {symbol}.")
                continue

            # Erstelle eine Liste der Predictoren, die wir abfragen wollen
            predictors = {
                "Daily": predictor_daily,
                "Swing": predictor_swing,
                "Genius": predictor_genius
            }

            for strategy_name, predictor_module in predictors.items():
                print(f"Rufe Vorhersage ab für: {symbol} - Strategie: {strategy_name}")
                try:
                    # Der jeweilige Predictor wird mit seinen spezifischen Modellen und Features aufgerufen
                    prediction = predictor_module.get_prediction(symbol, df.copy(), predictor_module.MODEL_PATH_BTC if 'BTC' in symbol else predictor_module.MODEL_PATH_XAU)

                    if "error" in prediction:
                        print(f"Fehler vom {strategy_name}-Predictor für {symbol}: {prediction['error']}")
                        continue

                    # Bestimme Farbe und Icon basierend auf dem Signal
                    color = "grey"
                    icon = "fa-minus-circle"
                    if prediction['signal'] == 'Kaufen':
                        color = "green"
                        icon = "fa-arrow-up"
                    elif prediction['signal'] == 'Verkaufen':
                        color = "red"
                        icon = "fa-arrow-down"

                    # Füge die aufbereitete Vorhersage zur Liste hinzu
                    assets_data.append({
                        "name": f"{symbol} ({strategy_name})",
                        "signal": prediction.get('signal'),
                        "entry_price": f"{prediction.get('entry_price'):.2f}",
                        "take_profit": f"{prediction.get('take_profit'):.2f}",
                        "stop_loss": f"{prediction.get('stop_loss'):.2f}",
                        "color": color,
                        "icon": icon,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                except Exception as e:
                    print(f"Schwerer Fehler beim Ausführen des {strategy_name}-Predictors für {symbol}: {e}")

    return jsonify(assets_data)

# --- Die restlichen Routen für Charts, Dashboard, Push-Benachrichtigungen etc. bleiben bestehen ---

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    """Holt die historischen TAGES-Daten für die Charts."""
    db_symbol = f"{symbol.split('(')[0].strip()}"
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
        return jsonify({"labels": [], "data": []})


@app.route('/dashboard')
def dashboard():
    """Analyse-Dashboard. HINWEIS: Benötigt eine 'backtest_results.json'."""
    results = {"daily": [], "swing": [], "genius": []}
    try:
        with open('backtest_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Warnung: Konnte 'backtest_results.json' nicht laden. Dashboard ist leer. Fehler: {e}")
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