import sqlite3
import os

# Use the same absolute path logic as the main app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(SCRIPT_DIR, 'znou_exchange.db')

print(f"Attempting to inspect database at: {DATABASE_FILE}")

try:
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Query the master table to find all user-created tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Verification successful. Found tables: {tables}")
    
    conn.close()

except Exception as e:
    print(f"AN ERROR OCCURRED: {e}")