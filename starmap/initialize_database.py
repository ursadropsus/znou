import sqlite3
import os

# --- Use an absolute path to ensure we modify the correct DB ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(SCRIPT_DIR, 'znou_exchange.db')

print(f"Initializing database at: {DATABASE_FILE}")

# Connect to the database
conn = sqlite3.connect(DATABASE_FILE)
cursor = conn.cursor()

# --- Create the 'Events' Table ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS Events (
    event_id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    neuron_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_claimed BOOLEAN DEFAULT 0,
    claimed_by_operator_id TEXT,
    kred_value INTEGER DEFAULT 0
);
""")
print("Table 'Events' created or already exists.")

# --- Create the 'Operators' Table ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS Operators (
    operator_id TEXT PRIMARY KEY,
    total_kreds_earned INTEGER DEFAULT 0,
    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")
print("Table 'Operators' created or already exists.")

# Commit the changes and close the connection
conn.commit()
conn.close()

print(f"Database '{DATABASE_FILE}' initialized successfully.")