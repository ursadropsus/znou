import json
import sys
import time
from pathlib import Path
import torch
import nltk

# --- Local Project Imports ---
try:
    import experiment_runner
    import config
except ImportError:
    print("FATAL ERROR: Could not import project modules (experiment_runner, config).")
    print("Please ensure you are running this script from the project's root directory.")
    sys.exit(1)

# --- NLTK Sentence Tokenizer Download ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK sentence tokenizer data (punkt)...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")

# --- Configuration ---
TARGET_LAYER = 5

# --- I/O Configuration ---
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")

DATASET_FILENAME = "wiki103.txt"
# NEW FILENAMES for 4-quadrant data capture
OUTPUT_FILENAME = f"wiki103_l{TARGET_LAYER}_sentences_quad.jsonl"
CHECKPOINT_FILENAME = f"progress_state_l{TARGET_LAYER}_sentences_quad.json"

# --- Performance & Logging ---
LOG_INTERVAL = 2500       # Log every 2500 sentences (fewer, as each is more expensive)
CHECKPOINT_INTERVAL = 10000 # Save progress every 10000 sentences

# --- State Management Functions (Unchanged) ---

def load_checkpoint(filepath: Path) -> int:
    """Loads the last processed LINE index from the checkpoint file."""
    if not filepath.is_file():
        print("Checkpoint file not found. Starting from the beginning.")
        return 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        last_index = data.get("line_index", 0)
        print(f"Resuming from line index: {last_index}")
        return last_index
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read checkpoint file at {filepath}. Error: {e}")
        return 0

def save_checkpoint(filepath: Path, current_line_idx: int):
    """Saves the current LINE index to the checkpoint file."""
    state = {
        "dataset": DATASET_FILENAME,
        "layer": TARGET_LAYER,
        "line_index": current_line_idx,
        "timestamp": time.time()
    }
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
    except IOError as e:
        print(f"\nFATAL ERROR: Could not write checkpoint to {filepath}. Error: {e}", file=sys.stderr)


# --- REVISED Core Processing Function for 4 Quadrants ---

def get_all_quadrant_ids_for_sentence(prompt_text: str, layer_idx: int) -> dict[str, int] | None:
    """
    Calls the master function in experiment_runner to get all four neuron IDs.
    """
    clean_prompt = prompt_text.strip()
    if not clean_prompt:
        return None

    # This now uses the efficient master function that performs both passes.
    try:
        # Note: We are calling the internal function directly for efficiency in this script.
        all_ids = experiment_runner._get_all_neuron_ids_for_prompt(clean_prompt, layer_idx)
        return all_ids
    except Exception as e:
        print(f"\nError in mapper's core processing function: {e}", file=sys.stderr)
        return None


# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting GPT-2 Layer 5 QUADRANT Neuron Miner ---")
    print("(This will capture Implicit/Explicit Resonance and Inference IDs for each sentence)")

    dataset_path = DATA_DIR / DATASET_FILENAME
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    checkpoint_path = OUTPUT_DIR / CHECKPOINT_FILENAME

    if not dataset_path.is_file():
        print(f"FATAL ERROR: Dataset file not found at '{dataset_path}'")
        sys.exit(1)

    print("Loading model and tokenizer...")
    experiment_runner.load_model_and_tokenizer()
    print("Model loaded successfully.")

    start_line_index = load_checkpoint(checkpoint_path)
    total_sentences_processed = 0
    start_time = time.time()
    
    # Track coverage for the two main Resonance maps
    discovered_explicit_res = set()
    discovered_implicit_res = set()
    
    output_file_handle = None
    current_line_idx = -1 # Initialize before loop

    try:
        print(f"Opening output file for append: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_handle = open(output_path, 'a', encoding='utf-8')

        print(f"Starting to process dataset: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as dataset_file:
            for current_line_idx, line in enumerate(dataset_file):
                if current_line_idx < start_line_index:
                    continue

                sentences_in_line = nltk.sent_tokenize(line)

                for sentence in sentences_in_line:
                    neuron_ids = get_all_quadrant_ids_for_sentence(sentence, TARGET_LAYER)

                    if neuron_ids and all(val is not None for val in neuron_ids.values()):
                        # We use short keys to keep the file size down
                        result = {
                            "s": sentence.strip(),
                            "exp_r": neuron_ids["exp_res"],
                            "exp_i": neuron_ids["exp_inf"],
                            "imp_r": neuron_ids["imp_res"],
                            "imp_i": neuron_ids["imp_inf"]
                        }
                        json.dump(result, output_file_handle)
                        output_file_handle.write('\n')
                        
                        discovered_explicit_res.add(neuron_ids["exp_res"])
                        discovered_implicit_res.add(neuron_ids["imp_res"])

                    total_sentences_processed += 1
                    
                    if total_sentences_processed % LOG_INTERVAL == 0:
                        elapsed_time = time.time() - start_time
                        sps = total_sentences_processed / elapsed_time if elapsed_time > 0 else 0
                        print(f"\nProcessed: {total_sentences_processed} sentences (line {current_line_idx}) | Speed: {sps:.2f} sent/sec")
                        print(f"  - Explicit Resonance Coverage: {len(discovered_explicit_res)}/{config.DIMENSION}")
                        print(f"  - Implicit Resonance Coverage: {len(discovered_implicit_res)}/{config.DIMENSION}")

                    if total_sentences_processed % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(checkpoint_path, current_line_idx)
                        output_file_handle.flush()

        print("\n--- Reached end of dataset file. ---")

    except KeyboardInterrupt:
        print("\n\n--- Interruption detected. Shutting down gracefully. ---")
    except Exception as e:
        print(f"\n\n--- An unexpected error occurred: {e} ---", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        print("Finalizing...")
        if output_file_handle:
            output_file_handle.close()
            print(f"Output file closed: {output_path}")
        
        if current_line_idx > -1:
            save_checkpoint(checkpoint_path, current_line_idx)
            print(f"Final checkpoint saved for line index: {current_line_idx}")
            
        print("--- Miner shutdown complete. ---")