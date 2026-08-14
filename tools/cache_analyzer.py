# cache_analyzer.py
#
# The "Quality Assurance Lab" for the Chimera Directive's data pipeline.
#
# This script reads the output of a data_pipeline.py run and provides a
# detailed analysis in two stages:
#   1. A full statistical report on the dataset's character.
#   2. An interactive deep-dive mode to investigate specific neurons.
#
# USAGE:
#   - Interactive Mode: python cache_analyzer.py
#   - Headless Mode:  python cache_analyzer.py <path_to_run_directory>
#

import json
import sys
import argparse
import time
from pathlib import Path
import numpy as np

# --- SciPy for Correlation (Optional but Recommended) ---
try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- Local Project Imports ---
try:
    import config
except ImportError:
    print("FATAL ERROR: Could not import project modules. Run from the project's root directory.")
    sys.exit(1)

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================
def get_relative_time(ts: float) -> str:
    """Converts a timestamp into a human-readable relative time string."""
    delta = time.time() - ts
    if delta < 60: return f"{int(delta)} seconds ago"
    if delta < 3600: return f"{int(delta / 60)} minutes ago"
    if delta < 86400: return f"{int(delta / 3600)} hours ago"
    if delta < 604800: return f"{int(delta / 86400)} days ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))

def prompt_for_selection(options: list, title: str) -> int:
    """Displays a numbered list of options and prompts the user for a valid choice."""
    print(f"\n{title}")
    for i, option_text in enumerate(options):
        print(f"  [{i+1}] {option_text}")
    while True:
        try:
            choice = input("Please select an option: ")
            choice_int = int(choice)
            if 1 <= choice_int <= len(options): return choice_int
            else: print(f"Invalid selection. Please enter a number between 1 and {len(options)}.")
        except ValueError: print("Invalid input. Please enter a number.")

# ==============================================================================
# --- DATA LOADING (NOW MORE ROBUST) ---
# ==============================================================================
def load_run_data(run_dir: Path) -> tuple[dict[str, np.ndarray] | None, Path | None]:
    """Loads binary hit-counts and the JSONL log from a run directory."""
    print(f"Analyzing run directory: {run_dir}")
    
    # The directory name IS the run name. This is the robust fix.
    run_name = run_dir.name
    print(f"Inferred Run Name: {run_name}")
    
    hit_counts = {}
    quadrant_keys = ["exp_r", "exp_i", "imp_r", "imp_i"]
    
    # Load binary files
    all_bins_found = True
    for key in quadrant_keys:
        bin_path = run_dir / f"{run_name}_{key}.bin"
        if not bin_path.is_file():
            print(f"Error: Missing binary cache file: {bin_path}")
            all_bins_found = False
            continue
        with open(bin_path, 'rb') as f:
            hit_counts[key] = np.frombuffer(f.read(), dtype=np.uint32)
    
    # Find JSONL file
    jsonl_path = run_dir / f"{run_name}.jsonl"
    if not jsonl_path.is_file():
        print(f"Error: Missing research log file: {jsonl_path}")
        return None, None
        
    if not all_bins_found:
        return None, None

    return hit_counts, jsonl_path

# ==============================================================================
# --- LAYER 2: STATISTICAL ANALYSIS ---
# ==============================================================================
def calculate_statistics(hit_counts: dict) -> dict:
    stats = {}
    for key, arr in hit_counts.items():
        total_hits = np.sum(arr)
        top_10_indices = np.argsort(arr)[-10:][::-1]
        stats[key] = {
            "total_hits": total_hits, "max_hit": np.max(arr), "mean_hit": np.mean(arr),
            "median_hit": np.median(arr), "sparsity": np.count_nonzero(arr == 0) / len(arr),
            "top_10_neurons": top_10_indices, "top_10_hits": arr[top_10_indices]
        }
    return stats

def calculate_comparisons(hit_counts: dict, stats: dict) -> dict:
    comparisons = {}
    pairs = [("exp_r", "imp_r"), ("exp_i", "imp_i"), ("exp_r", "exp_i")]
    for q1, q2 in pairs:
        key = f"{q1}_vs_{q2}"
        set1, set2 = set(stats[q1]["top_10_neurons"]), set(stats[q2]["top_10_neurons"])
        correlation = "N/A (SciPy not installed)"
        if SCIPY_AVAILABLE:
            corr_val, _ = pearsonr(hit_counts[q1], hit_counts[q2])
            correlation = f"{corr_val:.4f}"
        comparisons[key] = {"top_10_overlap": len(set1.intersection(set2)), "correlation": correlation}
    return comparisons

def display_statistical_report(stats: dict, comparisons: dict):
    print("\n" + "="*80); print("--- STATISTICAL ANALYSIS REPORT ---".center(80)); print("="*80)
    for key, data in stats.items():
        print(f"\n--- Quadrant: {key.replace('_', ' ').title()} ---")
        print(f"  Total Hits:         {data['total_hits']:>15,}")
        print(f"  Sparsity (Zeroes):  {data['sparsity']:>15.2%}")
        print(f"  Max Hit (Neuron):   {data['max_hit']:>15,}")
        print(f"  Mean Hit / Neuron:  {data['mean_hit']:>15.2f}")
        print("\n  Top 10 Hotspot Neurons (ID: Hits)")
        for neuron, hits in zip(data['top_10_neurons'], data['top_10_hits']):
            print(f"    - {neuron:<5} : {hits:,}")
    print("\n" + "-"*80); print("--- Cross-Quadrant Comparisons ---".center(80)); print("-"*80)
    for key, data in comparisons.items():
        print(f"\n- Comparison: {key.replace('_', ' ').title()}")
        print(f"  Top 10 Overlap:   {data['top_10_overlap']} neurons")
        print(f"  Correlation:      {data['correlation']}")
    print("\n" + "="*80)

# ==============================================================================
# --- LAYER 3: INTERACTIVE DEEP DIVE ---
# ==============================================================================
def find_sentences_for_neuron(neuron_id: int, jsonl_path: Path, cache: dict) -> dict:
    if neuron_id in cache: return cache[neuron_id]
    print(f"\nScanning research log for Neuron {neuron_id}... (This may take a moment on first lookup)")
    found = {k: [] for k in ["exp_r", "exp_i", "imp_r", "imp_i"]}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                for key in found:
                    if data.get(key) == neuron_id: found[key].append(data.get("s", "Missing sentence"))
            except json.JSONDecodeError: continue
    cache[neuron_id] = found
    return found

def run_interactive_deep_dive(jsonl_path: Path, hit_counts: dict):
    sentence_cache = {}
    while True:
        try:
            prompt = "\nEnter a Neuron ID to investigate (e.g., 1790), or 'q' to quit: "
            user_input = input(prompt).strip().lower()
            if user_input in ['q', 'quit']: print("Exiting deep dive mode."); break
            neuron_id = int(user_input)
            if not (0 <= neuron_id < config.DIMENSION):
                print(f"Error: Neuron ID must be between 0 and {config.DIMENSION - 1}."); continue
            sentences_by_quadrant = find_sentences_for_neuron(neuron_id, jsonl_path, sentence_cache)
            print("\n" + "-"*50); print(f"--- Deep Dive: Neuron {neuron_id} ---".center(50)); print("-"*50)
            print("\n--- Hit Counts ---")
            for key, arr in hit_counts.items(): print(f"  {key.replace('_', ' ').title():<25}: {arr[neuron_id]:,} hits")
            for key, sentences in sentences_by_quadrant.items():
                print(f"\n--- Sample sentences for {key.replace('_', ' ').title()} ---")
                if not sentences: print("  None found in this sample.")
                else:
                    for s in sentences[:5]: print(f"  - \"{s}\"")
                    if len(sentences) > 5: print(f"  ... and {len(sentences) - 5} more.")
        except ValueError: print("Invalid input. Please enter a number or 'q'.")
        except KeyboardInterrupt: print("\nExiting deep dive mode."); break

# ==============================================================================
# --- MAIN ORCHESTRATION ---
# ==============================================================================
def run_analysis(run_dir: Path):
    """The core analysis engine, takes a directory path and runs all stages."""
    if not run_dir.is_dir():
        print(f"FATAL ERROR: Directory not found: {run_dir}"); sys.exit(1)
    
    hit_counts, jsonl_path = load_run_data(run_dir)
    if hit_counts is None or jsonl_path is None:
        print("Aborting due to missing data files."); sys.exit(1)
        
    stats = calculate_statistics(hit_counts)
    comparisons = calculate_comparisons(hit_counts, stats)
    display_statistical_report(stats, comparisons)
    run_interactive_deep_dive(jsonl_path, hit_counts)

def run_interactive_setup():
    """Guides the user to select a run directory."""
    print("--- Chimera Directive Cache Analyzer (Interactive Mode) ---")
    output_dir = Path("../output")
    print(f"\nScanning for completed pipeline runs in '{output_dir}'...")
    
    run_dirs = [p for p in output_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        print(f"FATAL ERROR: No run directories found in '{output_dir}'."); sys.exit(1)
        
    # Sort by most recently modified
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    dir_options = [f"{p.name} (Created: {get_relative_time(p.stat().st_mtime)})" for p in run_dirs]
    dir_choice_idx = prompt_for_selection(dir_options, "Please select a run directory to analyze:") - 1
    selected_dir = run_dirs[dir_choice_idx]
    
    return selected_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzes the output of a data_pipeline.py run.")
    # nargs='?' makes the argument optional for interactive mode
    parser.add_argument("run_directory", type=Path, nargs='?', default=None, help="Optional: Path to the output directory of a pipeline run.")
    args = parser.parse_args()

    if args.run_directory:
        # --- Headless Mode ---
        run_analysis(args.run_directory)
    else:
        # --- Interactive Mode ---
        selected_run_dir = run_interactive_setup()
        run_analysis(selected_run_dir)