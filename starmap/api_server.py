# api_server.py (Final Version with All Features)
import struct
import os
import gevent.monkey
gevent.monkey.patch_all()

import json
import sqlite3
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
import experiment_runner
import re

# --- Operator ID Validation ---
VALID_OPERATOR_ID_PATTERN = re.compile(r'^ZNO-[A-Z0-9]{8}$')

# --- Yield Grade Configuration ---
YIELD_GRADES = {
    "Eschaton Grade": 0.9975,
    "Apocryphal Node": 0.985,
    "Anomalous Artifact": 0.95,
    "Aberrant Pattern": 0.88,
    "Variant Trace": 0.75,
    "Standard Signal": 0.55,
    "Residual Class": 0.0
}

def get_yield_grade(normalized_rarity: float) -> str:
    """Returns the correct thematic grade based on a rarity score."""
    for grade, threshold in YIELD_GRADES.items():
        if normalized_rarity >= threshold:
            return grade
    return "Residual Class" # Fallback

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# --- Database Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(SCRIPT_DIR, 'znou_exchange.db')

def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# --- WebSocket Global State ---
connected_clients = []

# --- Game Configuration ---
GAME_TARGET_LAYER = 5
MIN_KREDS = 100
MAX_KREDS = 25000

# --- WebSocket Endpoint ---
@sock.route('/live_feed')
def live_feed(ws):
    print("Received new WebSocket connection.")
    connected_clients.append(ws)
    try:
        while True:
            ws.receive(timeout=60)
    except Exception:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)

def broadcast_to_clients(prompt_text: str):
    message = json.dumps({"type": "scan_event", "prompt": prompt_text})
    for client_ws in connected_clients[:]:
        try:
            client_ws.send(message)
        except Exception:
            if client_ws in connected_clients:
                connected_clients.remove(client_ws)

# --- Kred Calculation Helper ---
def calculate_kreds_from_rarity(normalized_rarity: float) -> int:
    if not (0.0 <= normalized_rarity <= 1.0):
        return MIN_KREDS
    kreds = MIN_KREDS + (normalized_rarity * (MAX_KREDS - MIN_KREDS))
    return int(kreds)

# --- API Endpoint: /scan ---
@app.route("/scan", methods=['POST'])
def handle_scan():
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = request.json['prompt']
    protocol = request.json.get('protocol', 'explicit')
    mode = request.json.get('mode', 'resonance')
    
    broadcast_to_clients(prompt)
    
    neuron_id = experiment_runner.get_peak_neuron_for_prompt(
        prompt_text=prompt, layer_idx=GAME_TARGET_LAYER, protocol=protocol, mode=mode
    )
    
    if neuron_id is None: neuron_id = 0
    
    quadrant_key = f"{protocol[:3]}_{mode[0]}"
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT current_rarity_score FROM MasterHitCounts WHERE neuron_id = ? AND quadrant_key = ?",
        (neuron_id, quadrant_key)
    )
    rarity_row = cursor.fetchone()
    
    if rarity_row and rarity_row['current_rarity_score']:
        raw_bytes = rarity_row['current_rarity_score']
        try:
            current_rarity = struct.unpack('<f', raw_bytes)[0]
        except (struct.error, TypeError):
            current_rarity = 0.0
    else:
        current_rarity = 0.0
    
    potential_kred_value = calculate_kreds_from_rarity(current_rarity)
    
    event_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO Events (event_id, prompt, neuron_id, kred_value) VALUES (?, ?, ?, ?)",
        (event_id, prompt, neuron_id, potential_kred_value)
    )
    conn.commit()
    conn.close()
    
    return jsonify({ "neuron_id": neuron_id, "event_id": event_id })

# --- API Endpoint: /api/exchange/claim ---
@app.route("/api/exchange/claim", methods=['POST'])
def handle_claim():
    if not request.json or 'operator_id' not in request.json or 'event_ids' not in request.json:
        return jsonify({"error": "Missing 'operator_id' or 'event_ids' in request body"}), 400

    operator_id = request.json['operator_id']
    event_ids = request.json['event_ids']

    if not isinstance(operator_id, str):
        return jsonify({"error": "Operator ID must be a string."}), 400
    operator_id = operator_id.strip()
    if not VALID_OPERATOR_ID_PATTERN.fullmatch(operator_id):
        return jsonify({"error": "Invalid Operator ID format. Expected format: 'ZNO-XXXXXXXX'."}), 400

    if not isinstance(event_ids, list) or not event_ids:
        return jsonify({"error": "'event_ids' must be a non-empty array."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    total_kreds_awarded = 0
    claimed_event_ids_count = 0
    settlement_report = []

    cursor.execute("INSERT OR IGNORE INTO Operators (operator_id) VALUES (?)", (operator_id,))
    
    for event_id in event_ids:
        cursor.execute("SELECT * FROM Events WHERE event_id = ? AND is_claimed = 0", (event_id,))
        event = cursor.fetchone()
        
        if event:
            kreds_for_this_event = event['kred_value']
            total_kreds_awarded += kreds_for_this_event

            cursor.execute("UPDATE Events SET is_claimed = 1, claimed_by_operator_id = ? WHERE event_id = ?", (operator_id, event_id))
            
            temp_neuron_info = experiment_runner._get_all_neuron_ids_for_prompt(event['prompt'], GAME_TARGET_LAYER)
            
            found_quadrant = None
            if temp_neuron_info:
                for key, nid in temp_neuron_info.items():
                    short_key = f"{key[:3]}_{key.split('_')[1][0]}"
                    if nid == event['neuron_id']:
                        found_quadrant = short_key
                        break
            
            if found_quadrant:
                cursor.execute(
                    "UPDATE MasterHitCounts SET operator_hits = operator_hits + 1 WHERE neuron_id = ? AND quadrant_key = ?",
                    (event['neuron_id'], found_quadrant)
                )
            
            claimed_event_ids_count += 1

            rarity_score = (kreds_for_this_event - MIN_KREDS) / (MAX_KREDS - MIN_KREDS)
            rarity_percent = max(0.0, min(1.0, rarity_score)) * 100

            report_item = {
                "asset_id": f"J5-{event['neuron_id']}",
                "method": found_quadrant.replace("_", " ") if found_quadrant else "Unknown",
                "rarity_index_percent": f"{rarity_percent:.2f}%",
                "yield_grade": get_yield_grade(rarity_score),
                "kreds_yielded": kreds_for_this_event,
                "prompt": event['prompt']
            }
            settlement_report.append(report_item)

    settlement_report.sort(key=lambda item: item['kreds_yielded'], reverse=True)

    if total_kreds_awarded > 0:
        cursor.execute(
            "UPDATE Operators SET total_kreds_earned = total_kreds_earned + ? WHERE operator_id = ?",
            (total_kreds_awarded, operator_id)
        )

    cursor.execute("SELECT total_kreds_earned FROM Operators WHERE operator_id = ?", (operator_id,))
    new_balance_row = cursor.fetchone()
    new_balance = new_balance_row['total_kreds_earned'] if new_balance_row else 0

    conn.commit()
    conn.close()

    print(f"Operator {operator_id} claimed {claimed_event_ids_count} events for {total_kreds_awarded} Kreds. New balance: {new_balance}")

    return jsonify({
        "kreds_awarded": total_kreds_awarded,
        "new_balance": new_balance,
        "events_claimed_count": claimed_event_ids_count,
        "settlement_report": settlement_report
    })

# --- API Endpoint: /api/leaderboard ---
@app.route("/api/leaderboard", methods=['GET'])
def get_leaderboard():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT operator_id, total_kreds_earned FROM Operators ORDER BY total_kreds_earned DESC LIMIT 100")
    
    leaderboard = [
        {"rank": i + 1, "operator_id": row['operator_id'], "total_yield": row['total_kreds_earned']}
        for i, row in enumerate(cursor.fetchall())
    ]
    
    conn.close()
    return jsonify(leaderboard)

# --- Run the Server ---
if __name__ == "__main__":
    print("Starting Z-NOU Exchange API server (v1.2 with Reports)...")
    app.run(host='0.0.0.0', port=5000)