import json
import sys
import requests # We'll need the requests library
from tqdm import tqdm # For a nice progress bar

# --- Configuration ---
# Make sure this points to your raw, sentences-only JSON file
RAW_INPUT_FILE = './data/the_sea_raw.json' 

# This is the name of the final, pre-computed file it will create
COMPUTED_OUTPUT_FILE = './data/the_sea.json' 

# The local URL for the API server running on the same machine
API_URL = 'http://localhost:5000/scan'

# --- Main Script ---

def precompute_cache():
    print(f"Loading raw sentences from: {RAW_INPUT_FILE}")
    try:
        with open(RAW_INPUT_FILE, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Raw input file not found at '{RAW_INPUT_FILE}'. Please check the path.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Could not parse '{RAW_INPUT_FILE}'. Make sure it is a valid JSON array of strings.")
        sys.exit(1)

    print(f"Found {len(sentences)} sentences to process.")
    
    precomputed_results = []

    # Using tqdm for a progress bar, iterating over the sentences
    for sentence in tqdm(sentences, desc="Scanning Sentences"):
        if not isinstance(sentence, str) or not sentence.strip():
            continue # Skip empty or invalid entries

        try:
            # Make the API call to the local Flask server
            response = requests.post(API_URL, json={'prompt': sentence})
            
            if response.status_code == 200:
                data = response.json()
                neuron_id = data.get('neuron_id')
                
                if isinstance(neuron_id, int):
                    precomputed_results.append({
                        "sentence": sentence,
                        "neuron_id": neuron_id
                    })
                else:
                    print(f"\nWarning: Invalid neuron_id received for sentence: '{sentence[:50]}...'")
            else:
                print(f"\nWarning: API returned status {response.status_code} for sentence: '{sentence[:50]}...'")

        except requests.exceptions.RequestException as e:
            print(f"\nFATAL ERROR: Could not connect to the API at {API_URL}.")
            print("Please ensure the api_server.py is running before you run this script.")
            sys.exit(1)

    print(f"\nProcessed all sentences. Saving {len(precomputed_results)} results to {COMPUTED_OUTPUT_FILE}")
    
    try:
        with open(COMPUTED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(precomputed_results, f, indent=2)
    except Exception as e:
        print(f"\nFATAL ERROR: Could not write to output file. Error: {e}")
        sys.exit(1)
        
    print("\nPre-computation complete! The new data cache is ready.")


if __name__ == "__main__":
    print("--- Starting Data Cache Pre-computation ---")
    # You may need to install the 'requests' and 'tqdm' libraries:
    # pip install requests tqdm
    precompute_cache()