# app.py (Finale Diamant-Version 2.0)
import os
import json
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from datetime import datetime

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
    return render_template('index.html')

@app.route('/api/assets')
def get_assets():
    assets_data = []
    symbols = ['BTC/USD', 'XAU/USD']
    with engine.connect() as conn:
        for symbol in symbols:
            query = text("SELECT * FROM historical_data_daily WHERE symbol = :symbol ORDER BY timestamp DESC LIMIT 200")
            df = pd.read_sql_query(query, conn, params={'symbol': symbol})
            df = df.sort_values(by='timestamp').reset_index(drop=True)

            if df.empty or len(df) < 151: continue

            predictors = {"Daily": predictor_daily, "Swing": predictor_swing, "Genius": predictor_genius}
            for strategy_name, predictor_module in predictors.items():
                try:
                    model_path = predictor_module.MODEL_PATH_BTC if 'BTC' in symbol else predictor_module.MODEL_PATH_XAU
                    prediction = predictor_module.get_prediction(symbol, df.copy(), model_path)

                    if "error" in prediction: continue

                    color = "grey"
                    icon = "fa-minus-circle"
                    if prediction['signal'] == 'Kaufen':
                        color, icon = "green", "fa-arrow-up"
                    elif prediction['signal'] == 'Verkaufen':
                        color, icon = "red", "fa-arrow-down"

                    assets_data.append({
                        "name": f"{symbol} ({strategy_name})", "signal": prediction.get('signal'),
                        "entry_price": f"{prediction.get('entry_price'):.2f}", "take_profit": f"{prediction.get('take_profit'):.2f}",
                        "stop_loss": f"{prediction.get('stop_loss'):.2f}", "color": color, "icon": icon,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                except Exception as e:
                    print(f"Fehler beim Ausführen des {strategy_name}-Predictors für {symbol}: {e}")
    return jsonify(assets_data)

@app.route('/historical-data/<symbol>')
def get_historical_data(symbol):
    db_symbol = symbol.split('(')[0].strip()
    query = text("SELECT timestamp, open, high, low, close FROM historical_data_daily WHERE symbol = :symbol_param ORDER BY timestamp ASC LIMIT 200")
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol_param": db_symbol}).fetchall()
        data_points = [{"time": row[0].strftime('%Y-%m-%d'), "open": row[1], "high": row[2], "low": row[3], "close": row[4]} for row in result]
        return jsonify(data_points)
    except Exception as e:
        print(f"Fehler beim Laden der OHLC-Chart-Daten für {db_symbol}: {e}")
        return jsonify([])

@app.route('/dashboard')
def dashboard():
    results = {"daily": [], "swing": [], "genius": []}
    try:
        with open('backtest_results.json', 'r', encoding='utf-8') as f: results = json.load(f)
    except Exception as e:
        print(f"Warnung: Konnte 'backtest_results.json' nicht laden. Dashboard ist leer. Fehler: {e}")
    return render_template('dashboard.html', results=results)

# --- ROUTEN FÜR PWA-DATEIEN ---
@app.route('/sw.js')
def sw(): return send_from_directory(app.static_folder, 'sw.js')

@app.route('/manifest.json')
def manifest(): return send_from_directory(app.static_folder, 'manifest.json')

@app.route('/favicon.ico')
def favicon(): return send_from_directory(app.static_folder, 'favicon.ico')

# --- ROUTE FÜR PUSH-BENACHRICHTIGUNGEN ---
@app.route('/subscribe', methods=['POST'])
def subscribe():
    subscription_info = request.get_json()
    if not subscription_info: return jsonify({'error': 'Keine Subscription-Daten erhalten'}), 400
    try:
        with engine.connect() as conn:
            stmt = insert(push_subscriptions).values(subscription_json=json.dumps(subscription_info))
            conn.execute(stmt)
            conn.commit()
        return jsonify({'success': True}), 201
    except Exception as e:
        print(f"Fehler beim Speichern der Subscription: {e}")
        return jsonify({'error': 'Fehler beim Speichern'}), 500

# STARTPUNKT DER APP
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)