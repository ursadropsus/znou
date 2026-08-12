# live_mapper.py
# A persistent, stateful discovery engine for mapping neuron activations in GPT-2.
# Version 2: Now with file percentage progress indicator.

import json
import sys
import time
from pathlib import Path
from datetime import datetime

# --- Dependencies ---
try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    print("Error: NLTK library not found. Please install it with 'pip install nltk'")
    print("After installation, you must also download the 'punkt' package.")
    print("Run this in a Python interpreter: import nltk; nltk.download('punkt')")
    sys.exit(1)

# --- Local Project Imports ---
try:
    import experiment_runner
    import config
except ImportError:
    print("Error: Could not import 'experiment_runner.py'.")
    print("Please ensure 'live_mapper.py' is in the same directory as your other project files.")
    sys.exit(1)


# --- Configuration ---
STATE_FILE_PATH = Path("starmap_state.json")
TARGET_LAYERS = config.TARGET_LAYERS
TOTAL_NEURONS = config.DIMENSION
CHUNK_SIZE_BYTES = 1 * 1024 * 1024
SAVE_INTERVAL_SECONDS = 300

# --- Core Functions ---

def ping_model_for_peak_neurons(prompt_text: str) -> dict[int, int | None]:
    """
    A lightweight wrapper to get only the peak activating neuron IDs for a prompt.
    Returns a dictionary like {layer: neuron_id}.
    """
    peak_neurons = {}
    try:
        if not experiment_runner.tokenizer:
            print("Tokenizer not available in experiment_runner.", file=sys.stderr)
            return {layer: None for layer in TARGET_LAYERS}
        
        input_ids = experiment_runner.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(config.DEVICE)
        
        if input_ids.shape[1] == 0:
            return {layer: None for layer in TARGET_LAYERS}

        captured_acts = experiment_runner.capture_activations(input_ids, TARGET_LAYERS)
        summary = experiment_runner.summarize_activations(captured_acts)

        for layer in TARGET_LAYERS:
            # Bug fix from previous version: check for integer key `layer` directly.
            if summary and layer in summary and "error" not in summary[layer]:
                peak_neurons[layer] = summary[layer].get("max_activating_neuron_idx")
            else:
                peak_neurons[layer] = None

    except Exception as e:
        print(f"\n[ERROR] Could not process prompt '{prompt_text[:50]}...': {e}", file=sys.stderr)
        return {layer: None for layer in TARGET_LAYERS}
        
    return peak_neurons

def generate_report(state: dict, duration_seconds: float):
    """Generates and saves a final summary .txt report."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"live_mapper_report_{timestamp}.txt"
    report_filepath = Path.cwd() / report_filename

    total_discovered = 0
    report_content = []
    report_content.append("=" * 50)
    report_content.append("  Live Mapper Final Report")
    report_content.append("=" * 50 + "\n")
    report_content.append(f"Report Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Total Run Duration: {duration_seconds:.2f} seconds")
    report_content.append(f"Source Data File: {state.get('source_filepath')}\n")

    for layer in TARGET_LAYERS:
        starmap = state.get(f"starmap_L{layer}", {})
        discovered_count = len(starmap)
        total_discovered += discovered_count
        coverage = (discovered_count / TOTAL_NEURONS) * 100
        
        report_content.append("-" * 50)
        report_content.append(f"Layer {str(layer).zfill(2)} Starmap Coverage")
        report_content.append("-" * 50)
        report_content.append(f"  - Unique Neurons Visited: {discovered_count} / {TOTAL_NEURONS}")
        report_content.append(f"  - Coverage: {coverage:.4f}%\n")
        
    report_content.append("=" * 50)
    report_content.append("End of Report")
    report_content.append("=" * 50)
    
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    print(f"\nSuccess! Final report saved to: {report_filepath}")
    return total_discovered

def main_loop():
    """The main execution function for the discovery engine."""
    start_time = time.time()
    last_save_time = start_time
    
    state = {}
    discovered_sets = {layer: set() for layer in TARGET_LAYERS}

    if STATE_FILE_PATH.exists():
        print(f"Loading existing state from '{STATE_FILE_PATH}'...")
        with open(STATE_FILE_PATH, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        for layer in TARGET_LAYERS:
            starmap_key = f"starmap_L{layer}"
            if starmap_key in state:
                discovered_sets[layer] = set(map(int, state[starmap_key].keys()))
        print("State loaded. Resuming discovery.")
    else:
        print("No existing state file found. Starting a new discovery session.")
        source_file = input("Please provide the full path to your large text file (e.g., a book, dictionary): ")
        if not Path(source_file).exists():
            print(f"Error: File not found at '{source_file}'")
            return
        state = {
            "source_filepath": source_file,
            "last_processed_offset": 0,
            **{f"starmap_L{layer}": {} for layer in TARGET_LAYERS}
        }

    source_filepath = Path(state["source_filepath"])
    start_offset = state["last_processed_offset"]

    # --- PROGRESS BAR SETUP ---
    total_file_size = source_filepath.stat().st_size if source_filepath.exists() else 0 # <-- NEW
    
    processed_prompts_count = 0
    total_new_discoveries = 0

    try:
        with open(source_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_offset)
            
            print("\nStarting the discovery engine. Press Ctrl+C to pause, save, and generate a report.")
            
            leftover_text = ""
            while True:
                chunk = f.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    print("\nEnd of file reached.")
                    break
                
                full_text = leftover_text + chunk
                sentences = sent_tokenize(full_text)
                
                if chunk.endswith(tuple('.!?')):
                    leftover_text = ""
                else:
                    leftover_text = sentences.pop()

                for sentence in sentences:
                    prompt = sentence.strip().replace('\n', ' ')
                    if not prompt: continue
                    
                    processed_prompts_count += 1
                    peak_neurons = ping_model_for_peak_neurons(prompt)
                    
                    for layer, neuron_id in peak_neurons.items():
                        if neuron_id is not None and neuron_id not in discovered_sets[layer]:
                            total_new_discoveries += 1
                            discovered_sets[layer].add(neuron_id)
                            starmap_key = f"starmap_L{layer}"
                            state[starmap_key][str(neuron_id)] = prompt
                            
                            coverage = (len(discovered_sets[layer]) / TOTAL_NEURONS) * 100
                            print(f"\n>> DISCOVERY L{str(layer).zfill(2)}! Neuron #{neuron_id} found with prompt: \"{prompt[:80]}...\"")
                            print(f"   (L{str(layer).zfill(2)} Coverage: {len(discovered_sets[layer])}/{TOTAL_NEURONS} | {coverage:.4f}%)")

                state["last_processed_offset"] = f.tell()

                # --- UPDATE PROGRESS BAR ---
                current_offset = state["last_processed_offset"] # <-- NEW
                progress_percent = (current_offset / total_file_size) * 100 if total_file_size > 0 else 0 # <-- NEW
                progress_message = ( # <-- NEW
                    f"\rProcessed: {processed_prompts_count} prompts... "
                    f"[{progress_percent:.2f}% of {source_filepath.name}]"
                )
                sys.stdout.write(progress_message) # <-- MODIFIED
                sys.stdout.flush()

                current_time = time.time()
                if current_time - last_save_time > SAVE_INTERVAL_SECONDS:
                    print(f"\n--- Auto-saving progress at {datetime.now().strftime('%H:%M:%S')} ---")
                    with open(STATE_FILE_PATH, 'w', encoding='utf-8') as f_state:
                        json.dump(state, f_state, indent=2)
                    last_save_time = current_time

    except KeyboardInterrupt:
        print("\n\nInterruption detected. Pausing discovery...")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Saving final state and generating report...")
        end_time = time.time()
        
        with open(STATE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        print(f"Final progress saved to '{STATE_FILE_PATH}'.")
        
        discoveries_this_session = generate_report(state, end_time - start_time)
        print(f"Found {total_new_discoveries} new neurons this session.")
        print("You can restart the script at any time to resume from where you left off.")


if __name__ == "__main__":
    main_loop()