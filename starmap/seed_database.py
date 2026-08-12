# seed_database.py
#
# USAGE:
#   Place this script in your main server directory alongside the
#   `rarity_index.json` file and the four `master_hits_*.bin` files.
#   Run `python seed_database.py` to generate the genesis database.
#
# DESCRIPTION:
#   This script creates the `znou_exchange.db` file from scratch. It builds
#   all necessary tables and populates the `MasterHitCounts` table with the
#   initial state of the Z-NOU economy, using the pre-compiled corpus data
#   and the calculated rarity scores.

import os
import json
import sqlite3
import numpy as np

# --- CONFIGURATION ---
DB_FILENAME = "znou_exchange.db"
RARITY_INDEX_FILE = "rarity_index.json"
HITS_FILE_TEMPLATE = "master_hits_{key}.bin"
QUADRANT_KEYS = ["exp_r", "exp_i", "imp_r", "imp_i"]
NEURON_DIMENSION = 3072
DATA_TYPE = np.uint32

def create_tables(cursor):
    """Creates all the necessary tables for the game."""
    print("Creating database tables...")
    # Table for player accounts and balances
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Operators (
        operator_id TEXT PRIMARY KEY,
        total_kreds_earned INTEGER NOT NULL DEFAULT 0,
        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # Table to log discovery events before they are claimed
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Events (
        event_id TEXT PRIMARY KEY,
        prompt TEXT NOT NULL,
        neuron_id INTEGER NOT NULL,
        kred_value INTEGER NOT NULL,
        is_claimed INTEGER NOT NULL DEFAULT 0,
        claimed_by_operator_id TEXT
    );
    """)
    # The new master table for the dynamic economy
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS MasterHitCounts (
        neuron_id INTEGER NOT NULL,
        quadrant_key TEXT NOT NULL,
        corpus_hits INTEGER NOT NULL DEFAULT 0,
        operator_hits INTEGER NOT NULL DEFAULT 0,
        current_rarity_score REAL NOT NULL DEFAULT 0.0,
        PRIMARY KEY (neuron_id, quadrant_key)
    );
    """)
    print("Tables created successfully.")

def seed_master_hit_counts(cursor, rarity_data):
    """Populates the MasterHitCounts table with genesis data."""
    print("Seeding the MasterHitCounts table with genesis data...")
    
    all_data_to_insert = []

    for key in QUADRANT_KEYS:
        print(f"  - Processing quadrant: {key}")
        
        # Load the raw corpus hits
        hits_filename = HITS_FILE_TEMPLATE.format(key=key)
        if not os.path.exists(hits_filename):
            print(f"    [ERROR] Missing hits file: {hits_filename}. Cannot seed.")
            return False
        
        corpus_hits_array = np.fromfile(hits_filename, dtype=DATA_TYPE)
        rarity_scores_list = rarity_data[key]

        for neuron_id in range(NEURON_DIMENSION):
            corpus_hits = int(corpus_hits_array[neuron_id])
            rarity_score = float(rarity_scores_list[neuron_id])
            
            # (neuron_id, quadrant, corpus_hits, operator_hits, rarity_score)
            all_data_to_insert.append(
                (neuron_id, key, corpus_hits, 0, rarity_score)
            )

    cursor.executemany(
        "INSERT INTO MasterHitCounts (neuron_id, quadrant_key, corpus_hits, operator_hits, current_rarity_score) VALUES (?, ?, ?, ?, ?)",
        all_data_to_insert
    )
    print(f"Seeding complete. Inserted {len(all_data_to_insert)} records.")
    return True


def main():
    """Main function to create and seed the database."""
    print("--- Z-NOU Exchange Genesis Database Seeder ---")

    # Safety check to prevent overwriting an existing database
    if os.path.exists(DB_FILENAME):
        overwrite = input(f"[WARNING] Database '{DB_FILENAME}' already exists. Overwrite it? (y/n): ").lower()
        if overwrite != 'y':
            print("Seeding aborted by user.")
            return
        os.remove(DB_FILENAME)
        print("Removed existing database.")

    # Load the rarity index JSON
    if not os.path.exists(RARITY_INDEX_FILE):
        print(f"[FATAL ERROR] '{RARITY_INDEX_FILE}' not found. Please place it in this directory.")
        return
        
    with open(RARITY_INDEX_FILE, 'r') as f:
        rarity_data = json.load(f)

    conn = None
    try:
        conn = sqlite3.connect(DB_FILENAME)
        cursor = conn.cursor()
        
        create_tables(cursor)
        success = seed_master_hit_counts(cursor, rarity_data)
        
        if success:
            conn.commit()
            print(f"\nGenesis database '{DB_FILENAME}' created successfully.")
        else:
            print("\nDatabase seeding failed. Rolling back.")
            conn.rollback()

    except sqlite3.Error as e:
        print(f"\n[FATAL ERROR] A database error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()