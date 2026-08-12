# market_ticker.py
#
# USAGE:
#   Run this script in its own terminal session from the main server directory,
#   after activating the virtual environment.
#   `python market_ticker.py`
#
# DESCRIPTION:
#   This script acts as the asynchronous heartbeat of the Z-NOU Exchange.
#   It runs in a continuous loop, waking up at a set interval to perform the
#   computationally expensive task of recalculating all neuron rarity scores.
#
#   It uses a zero-downtime "shadow table" method:
#     1. Reads the latest hit counts from the live database.
#     2. Performs all rarity calculations on a temporary table.
#     3. Atomically swaps the temporary table with the live one.
#
#   This ensures the main API server remains fast and responsive for players
#   while the market dynamically adjusts to their discoveries.

import os
import sqlite3
import time
import numpy as np
import schedule

# --- CONFIGURATION ---
DB_FILENAME = "znou_exchange.db"
RECALCULATION_INTERVAL_MINUTES = 5 # How often the market updates
QUADRANT_KEYS = ["exp_r", "exp_i", "imp_r", "imp_i"]
NEURON_DIMENSION = 3072
EPSILON = 1e-12

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILENAME)
    conn.row_factory = sqlite3.Row
    return conn

def calculate_rarity(hits_array: np.ndarray) -> np.ndarray:
    """Performs the core rarity calculation logic."""
    # This logic is identical to `calculate_rarity_index.py`
    smoothed_hits = hits_array.astype(np.float64) + 1
    total_hits = np.sum(smoothed_hits)
    probabilities = smoothed_hits / total_hits
    rarity_scores = -np.log(probabilities + EPSILON)
    
    min_rarity = np.min(rarity_scores)
    max_rarity = np.max(rarity_scores)
    
    if (max_rarity - min_rarity) < EPSILON:
        return np.zeros(NEURON_DIMENSION, dtype=np.float32)
    else:
        normalized_rarity = (rarity_scores - min_rarity) / (max_rarity - min_rarity)
        return normalized_rarity.astype(np.float32)

def run_market_recalibration():
    """The main job that recalculates all rarity scores."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting market recalibration cycle...")
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # --- Step 1: Create a temporary "shadow" table ---
        print("  - Creating shadow table for new calculations...")
        # Use a unique name to avoid conflicts if a previous run failed
        temp_table_name = f"MasterHitCounts_Temp_{int(time.time())}"
        cursor.execute(f"""
        CREATE TABLE {temp_table_name} (
            neuron_id INTEGER NOT NULL,
            quadrant_key TEXT NOT NULL,
            corpus_hits INTEGER NOT NULL,
            operator_hits INTEGER NOT NULL,
            current_rarity_score REAL NOT NULL,
            PRIMARY KEY (neuron_id, quadrant_key)
        );
        """)

        # --- Step 2: Fetch current data and perform calculations ---
        all_new_data = []
        for key in QUADRANT_KEYS:
            print(f"    - Calculating for quadrant: {key}")
            
            # Fetch the latest combined hit counts from the LIVE table
            cursor.execute(
                "SELECT neuron_id, corpus_hits, operator_hits FROM MasterHitCounts WHERE quadrant_key = ? ORDER BY neuron_id",
                (key,)
            )
            rows = cursor.fetchall()
            
            # This relies on the SELECT query being ordered by neuron_id
            combined_hits = np.array([row['corpus_hits'] + row['operator_hits'] for row in rows])

            # Recalculate rarity scores for this quadrant
            new_rarity_scores = calculate_rarity(combined_hits)

            # Prepare the full data for insertion into the shadow table
            for i, row in enumerate(rows):
                all_new_data.append((
                    row['neuron_id'],
                    key,
                    row['corpus_hits'],
                    row['operator_hits'],
                    new_rarity_scores[i]
                ))

        # --- Step 3: Populate the shadow table ---
        print(f"  - Populating shadow table with {len(all_new_data)} new records...")
        cursor.executemany(
            f"INSERT INTO {temp_table_name} VALUES (?, ?, ?, ?, ?)",
            all_new_data
        )
        conn.commit()

        # --- Step 4: The Atomic Swap (Zero-Downtime) ---
        print("  - Performing atomic swap of live and shadow tables...")
        old_table_name = f"MasterHitCounts_Old_{int(time.time())}"
        
        # This block is executed as a single transaction
        cursor.execute("BEGIN TRANSACTION;")
        cursor.execute(f"ALTER TABLE MasterHitCounts RENAME TO {old_table_name};")
        cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO MasterHitCounts;")
        cursor.execute(f"DROP TABLE {old_table_name};")
        conn.commit()
        
        print("  - Swap complete. New rarity scores are now live.")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Market recalibration cycle finished successfully.")

    except sqlite3.Error as e:
        print(f"[ERROR] A database error occurred during recalibration: {e}")
        if conn:
            conn.rollback() # Undo any partial changes
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

def main():
    """Main function to schedule and run the ticker job."""
    print("--- Z-NOU Exchange Market Ticker ---")
    print(f"Rarity scores will be recalculated every {RECALCULATION_INTERVAL_MINUTES} minutes.")
    print("Press Ctrl+C to stop the script.")
    
    # Run the job once immediately on startup
    run_market_recalibration()
    
    # Schedule the job to run at the configured interval
    schedule.every(RECALCULATION_INTERVAL_MINUTES).minutes.do(run_market_recalibration)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTicker stopped by user. Exiting.")

if __name__ == "__main__":
    # We need to install the 'schedule' library first.
    try:
        import schedule
    except ImportError:
        print("[ERROR] The 'schedule' library is not installed.")
        print("Please activate your virtual environment and run:")
        print("pip install schedule")
        exit()
        
    main()