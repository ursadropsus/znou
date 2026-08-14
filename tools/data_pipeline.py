# data_pipeline.py
#
# The "God Script" for the Chimera Directive's data pipeline.
#
# This single, comprehensive tool handles the entire workflow from a raw text corpus
# to final game-ready binary assets and detailed research logs.
#
# USAGE:
#   - Interactive Mode: python data_pipeline.py
#     (Launches a guided, menu-driven setup for ease of use.)
#
#   - Headless Mode: python data_pipeline.py --input <path> --name <name> [--sample <rate>]
#     (Runs directly with command-line arguments for automation.)
#

import json
import sys
import time
import argparse
import datetime
import random
from pathlib import Path

import torch
import numpy as np
import nltk
import io

# --- Local Project Imports ---
try:
    import experiment_runner
    import config
except ImportError:
    print("FATAL ERROR: Could not import project modules. Run from the project's root directory.")
    sys.exit(1)

# --- Matplotlib for Visual Reporting (Optional but Recommended) ---
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not found. Visual reports will not be generated.")
    print("Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

# --- NLTK Dependency Check ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK sentence tokenizer (punkt)...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")


# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
class PipelineConfig:
    """A container for all pipeline settings."""
    TARGET_LAYER = 5
    DATA_DIR = Path("../data")      # Use ../ to look one level up from the script's location
    OUTPUT_DIR = Path("../output")  # Use ../ to keep output dir at the project root
    LOG_INTERVAL = 5000
    CHECKPOINT_INTERVAL = 20000

PIPELINE_CONFIG = PipelineConfig()


# ==============================================================================
# --- HELPER FUNCTIONS for INTERACTIVE MODE ---
# ==============================================================================
def get_human_readable_size(size_bytes: int) -> str:
    """Converts a file size in bytes to a human-readable string."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def prompt_for_selection(options: list, title: str) -> int:
    """Displays a numbered list of options and prompts the user for a valid choice."""
    print(f"\n{title}")
    for i, option_text in enumerate(options):
        print(f"  [{i+1}] {option_text}")

    while True:
        try:
            choice = input("Please select an option: ")
            choice_int = int(choice)
            if 1 <= choice_int <= len(options):
                return choice_int
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# ==============================================================================
# --- CORE LOGIC & STATE MANAGEMENT ---
# ==============================================================================
class PipelineState:
    """Manages the state of a pipeline run, including checkpointing."""
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.checkpoint_path = run_dir / "checkpoint.json"
        self.line_index: int = 0
        self.total_sentences: int = 0
        self.start_time: float = time.time()
        self.hit_counts: dict[str, np.ndarray] = {
            "exp_r": np.zeros(config.DIMENSION, dtype=np.uint32),
            "exp_i": np.zeros(config.DIMENSION, dtype=np.uint32),
            "imp_r": np.zeros(config.DIMENSION, dtype=np.uint32),
            "imp_i": np.zeros(config.DIMENSION, dtype=np.uint32),
        }
        self.coverage_log: list[dict] = []

    def load_checkpoint(self):
        if not self.checkpoint_path.is_file():
            print("Checkpoint not found. Starting a fresh run.")
            return
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                state_json = json.load(f)
            self.line_index = state_json['line_index']
            self.total_sentences = state_json['total_sentences']
            self.coverage_log = state_json['coverage_log']
            for key in self.hit_counts:
                bin_path = self.run_dir / f"checkpoint_hits_{key}.bin"
                with open(bin_path, 'rb') as f_bin:
                    self.hit_counts[key] = np.frombuffer(f_bin.read(), dtype=np.uint32)
            print(f"Resumed from line index: {self.line_index} ({self.total_sentences} sentences processed).")
        except Exception as e:
            print(f"Warning: Could not read checkpoint. Starting fresh. Error: {e}")
            self.__init__(self.run_dir)

    def save_checkpoint(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for key, arr in self.hit_counts.items():
            bin_path = self.run_dir / f"checkpoint_hits_{key}.bin"
            with open(bin_path, 'wb') as f_bin:
                f_bin.write(arr.tobytes())
        state_json = { "line_index": self.line_index, "total_sentences": self.total_sentences, "coverage_log": self.coverage_log, "timestamp": time.time() }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(state_json, f, indent=4)

    def log_coverage(self):
        coverage = { "total_sentences": self.total_sentences, "timestamp": time.time() }
        for key, arr in self.hit_counts.items():
            coverage[f"coverage_{key}"] = int(np.count_nonzero(arr))
        self.coverage_log.append(coverage)

def get_all_quadrant_ids_for_sentence(prompt_text: str) -> dict[str, int] | None:
    model, tokenizer = experiment_runner.model, experiment_runner.tokenizer
    if model is None or tokenizer is None: raise RuntimeError("Model not loaded.")
    clean_prompt = prompt_text.strip()
    if not clean_prompt or len(clean_prompt.split()) < 3: return None
    try:
        input_ids_exp = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].to(config.DEVICE)
        if input_ids_exp.shape[1] == 0: return None
        explicit_activations = None
        def hook_fn_exp(act, hook): nonlocal explicit_activations; explicit_activations = act.detach().cpu()
        hook_name = f"blocks.{PIPELINE_CONFIG.TARGET_LAYER}.mlp.hook_post"
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, hook_fn_exp)]): model(input_ids_exp)
        bos_tensor = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=config.DEVICE)
        input_ids_imp = torch.cat([bos_tensor, input_ids_exp], dim=1)
        implicit_activations = None
        def hook_fn_imp(act, hook): nonlocal implicit_activations; implicit_activations = act.detach().cpu()
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, hook_fn_imp)]): model(input_ids_imp)
        if explicit_activations is None or implicit_activations is None: return None
        exp_acts, imp_acts = explicit_activations[0], implicit_activations[0]
        return { "exp_r": exp_acts.max(dim=0).values.argmax().item(), "exp_i": exp_acts[-1].argmax().item(), "imp_r": imp_acts.max(dim=0).values.argmax().item(), "imp_i": imp_acts[-1].argmax().item() }
    except Exception: return None

# ==============================================================================
# --- REPORTING ---
# ==============================================================================
def generate_final_report(state: PipelineState, args: argparse.Namespace, run_dir: Path, total_lines: int):
    print("Generating final report...")
    end_time = time.time()
    elapsed = end_time - state.start_time
    # Generate Plot
    if MATPLOTLIB_AVAILABLE and state.coverage_log:
        try:
            plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(12, 7))
            sentences = [log['total_sentences'] for log in state.coverage_log]
            for key in state.hit_counts:
                ax.plot(sentences, [log[f"coverage_{key}"] for log in state.coverage_log], label=f"{key.replace('_', ' ').title()} Coverage")
            ax.set_title(f"Neuron Coverage Over Time\nDataset: {args.input} (Run: {args.name})"); ax.set_xlabel("Sentences Processed"); ax.set_ylabel(f"Unique Neurons Discovered (out of {config.DIMENSION})")
            ax.grid(True, linestyle='--', alpha=0.3); ax.legend(); ax.set_ylim(bottom=0); ax.set_xlim(left=0)
            plot_path = run_dir / "report_coverage_over_time.png"
            plt.savefig(plot_path, dpi=150); print(f"Saved coverage plot to: {plot_path}")
        except Exception as e: print(f"Could not generate plot: {e}")
    # Generate Markdown Report
    report_path = run_dir / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Data Pipeline Run Report: `{args.name}`\n\n- **Run Start Time:** `{datetime.datetime.fromtimestamp(state.start_time).isoformat()}`\n- **Run End Time:** `{datetime.datetime.fromtimestamp(end_time).isoformat()}`\n- **Total Duration:** `{str(datetime.timedelta(seconds=elapsed))}`\n\n")
        f.write("## Processing Summary\n\n"); f.write(f"- **Source Dataset:** `{args.input}`\n- **Sampling Rate:** `{args.sample * 100 if args.sample else 100}%`\n- **Lines Processed:** `{state.line_index + 1:,} / {total_lines:,}`\n- **Sentences Processed:** `{state.total_sentences:,}`\n")
        sps = state.total_sentences / elapsed if elapsed > 0 else 0; f.write(f"- **Average Speed:** `{sps:.2f} sentences/sec`\n\n")
        f.write("## Final Neuron Coverage\n\n| Quadrant | Unique Neurons | Coverage |\n|:---|---:|---:|\n")
        for key, arr in state.hit_counts.items():
            count = int(np.count_nonzero(arr)); percent = (count / config.DIMENSION) * 100
            f.write(f"| {key.replace('_', ' ').title()} | `{count:,}` | `{percent:.2f}%` |\n")
        f.write("\n## Output Files\n\n"); f.write(f"- **Research Log (JSONL):** `output/{args.name}/{args.name}.jsonl`\n")
        for key in state.hit_counts: f.write(f"- **Game Cache (Binary):** `output/{args.name}/{args.name}_{key}.bin`\n")
        f.write(f"- **This Report:** `output/{args.name}/report.md`\n")
        if MATPLOTLIB_AVAILABLE and state.coverage_log: f.write(f"- **Coverage Plot:** `output/{args.name}/report_coverage_over_time.png`\n\n## Coverage Visualization\n\n![Neuron Coverage Over Time](report_coverage_over_time.png)\n")
    print(f"Saved detailed report to: {report_path}")

# ==============================================================================
# --- PIPELINE EXECUTION ENGINE ---
# ==============================================================================
def run_pipeline(args):
    """The core processing engine. Takes a configuration object and runs the pipeline."""
    run_dir = PIPELINE_CONFIG.OUTPUT_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Chimera Directive Data Pipeline ---")
    print(f"Run Name: {args.name}")
    print(f"Source File: {args.input}")
    print(f"Output Directory: {run_dir}")
    print(f"Sampling: {'{:.1f}%'.format(args.sample * 100) if args.sample else 'Full (100%)'}")

    # --- Load Model ---
    print("\nLoading GPT-2 model and tokenizer...")
    experiment_runner.load_model_and_tokenizer(); print("Model loaded successfully.")
    
    state = PipelineState(run_dir)
    jsonl_path = run_dir / f"{args.name}.jsonl"
    lines_to_process = []
    total_lines_in_file = 0

    # --- CHOOSE PROCESSING STRATEGY: FAST SAMPLING vs RESILIENT FULL RUN ---
    if args.sample:
        # Fast in-memory method for samples
        print("\nStrategy: Fast in-memory sampling.")
        print("Reading source file into memory...")
        with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
        total_lines_in_file = len(all_lines)
        num_to_sample = int(total_lines_in_file * args.sample)
        print(f"Source contains {total_lines_in_file:,} lines. Randomly sampling {num_to_sample:,} lines.")
        lines_to_process = random.sample(all_lines, num_to_sample)
        # For samples, we always start fresh, so no checkpoint loading.
        # jsonl is opened in 'w' mode to clear previous sample runs.
        jsonl_file_handle = open(jsonl_path, 'w', encoding='utf-8') 
    else:
        # Slower, line-by-line method for full runs to support robust checkpointing
        print("\nStrategy: Resilient disk-based processing for full run.")
        with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines_in_file = sum(1 for _ in f)
        print(f"Source contains {total_lines_in_file:,} lines.")
        state.load_checkpoint() # Only load checkpoint for full runs
        # jsonl is opened in 'a' mode to resume where we left off.
        jsonl_file_handle = open(jsonl_path, 'a', encoding='utf-8')
        lines_to_process = open(args.input, 'r', encoding='utf-8', errors='ignore')

    try:
        print(f"\nStarting main processing loop...")
        for i, line in enumerate(lines_to_process):
            if not args.sample and i < state.line_index: # Skip already processed lines in full runs
                continue
            state.line_index = i

            for sentence in nltk.sent_tokenize(line):
                neuron_ids = get_all_quadrant_ids_for_sentence(sentence)
                if neuron_ids:
                    jsonl_file_handle.write(json.dumps({"s": sentence.strip(), **neuron_ids}) + '\n')
                    for key, nid in neuron_ids.items(): state.hit_counts[key][nid] += 1
                state.total_sentences += 1
                
                # --- Logging and Checkpointing ---
                if state.total_sentences % PIPELINE_CONFIG.LOG_INTERVAL == 0:
                    elapsed = time.time() - state.start_time
                    sps = state.total_sentences / elapsed if elapsed > 0 else 0
                    print(f"\n> Progress: {state.total_sentences:,} sentences processed (line {i:,}) | Speed: {sps:.2f} sent/sec")
                    state.log_coverage()
                
                if not args.sample and state.total_sentences % PIPELINE_CONFIG.CHECKPOINT_INTERVAL == 0:
                    state.save_checkpoint(); jsonl_file_handle.flush()
                    print(f"  ... Checkpoint saved at sentence {state.total_sentences:,}")

        print("\n--- Finished processing. Finalizing run. ---")
    except KeyboardInterrupt: print("\n\n--- Interruption detected. Shutting down gracefully. ---")
    except Exception as e: print(f"\n\n--- An unexpected error occurred: {e} ---", file=sys.stderr); import traceback; traceback.print_exc()
    finally:
        print("\nFinalizing...")
        if 'jsonl_file_handle' in locals() and jsonl_file_handle: jsonl_file_handle.close()
        # For samples, lines_to_process is a list. For full runs, it's a file handle.
        if not args.sample and isinstance(lines_to_process, io.TextIOWrapper): lines_to_process.close()
        
        # Save final state, especially important for samples that don't checkpoint during run
        state.save_checkpoint(); print("Final checkpoint saved.")
        
        for key, arr in state.hit_counts.items():
            final_path = run_dir / f"{args.name}_{key}.bin"
            with open(final_path, 'wb') as f: f.write(arr.tobytes())
            print(f"Baked final game cache: {final_path}")
        
        generate_final_report(state, args, run_dir, total_lines_in_file)
        print("--- Pipeline shutdown complete. ---")


# ==============================================================================
# --- INTERACTIVE & HEADLESS MODE DISPATCHER ---
# ==============================================================================
def run_interactive_setup():
    """Guides the user through setting up a pipeline run interactively."""
    print("--- Chimera Directive Data Factory (Interactive Mode) ---")
    
    # --- 1. File Selection ---
    print(f"\nScanning '{PIPELINE_CONFIG.DATA_DIR}' for source files...")
    txt_files = sorted(list(PIPELINE_CONFIG.DATA_DIR.glob("*.txt")))
    if not txt_files:
        print(f"FATAL ERROR: No .txt files found in the '{PIPELINE_CONFIG.DATA_DIR}' directory.")
        sys.exit(1)
        
    file_options = [f"{p.name} ({get_human_readable_size(p.stat().st_size)})" for p in txt_files]
    file_choice_idx = prompt_for_selection(file_options, "Please select a source file to process:") - 1
    selected_file = txt_files[file_choice_idx]

    # --- 2. Sample Selection ---
    sample_options = ["Full Run (100%)", "Large Sample (10%)", "Quick Test (1%)", "Custom Percentage..."]
    sample_choice = prompt_for_selection(sample_options, f"You have selected '{selected_file.name}'. How much of it should be processed?")
    
    sample_rate = None
    if sample_choice == 1: sample_rate = None
    elif sample_choice == 2: sample_rate = 0.10
    elif sample_choice == 3: sample_rate = 0.01
    elif sample_choice == 4:
        while True:
            try:
                custom_perc = float(input("Enter custom percentage (e.g., 25 for 25%): "))
                if 0 < custom_perc <= 100: sample_rate = custom_perc / 100.0; break
                else: print("Please enter a percentage between 0 and 100.")
            except ValueError: print("Invalid input. Please enter a number.")

    # --- 3. Naming the Run ---
    print("\nPlease provide a unique name for this run.")
    print("This will be used for the output directory and filenames.")
    sample_str = "full" if sample_rate is None else f"{int(sample_rate*100)}perc"
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    suggested_name = f"{selected_file.stem}_{sample_str}_{date_str}"
    
    run_name_input = input(f"Suggested name: [{suggested_name}]\nPress ENTER to accept, or type a new name: ")
    run_name = run_name_input.strip() if run_name_input.strip() else suggested_name

    # --- 4. Final Confirmation ---
    print("\n================================================"); print("  READY TO LAUNCH"); print("------------------------------------------------")
    print(f"  Source File:   {selected_file.name}"); print(f"  Processing:    {'Full Run (100%)' if sample_rate is None else '{:.1f}%'.format(sample_rate * 100)}")
    print(f"  Run Name:      {run_name}"); print(f"  Output will be saved to: {PIPELINE_CONFIG.OUTPUT_DIR / run_name}/"); print("================================================")

    while True:
        confirm = input("Proceed with this configuration? (y/n): ").lower().strip()
        if confirm in ['y', 'yes']:
            class Args: pass
            args = Args(); args.input = str(selected_file); args.name = run_name; args.sample = sample_rate
            return args
        elif confirm in ['n', 'no']: print("Launch aborted by user."); sys.exit(0)
        else: print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # --- Headless Mode ---
        parser = argparse.ArgumentParser(description="The Chimera Directive's unified data processing pipeline.", formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument("-i", "--input", type=str, required=True, help="Path to the source text file (e.g., ../data/wiki103.txt)")
        parser.add_argument("-n", "--name", type=str, required=True, help="A unique name for this run.")
        parser.add_argument("-s", "--sample", type=float, default=None, help="Optional: A float between 0.0 and 1.0 for sampling.\nExample: 0.1 for 10%.")
        args = parser.parse_args()
        if args.sample and not (0.0 < args.sample <= 1.0):
            parser.error("--sample must be between 0.0 and 1.0.")
        run_pipeline(args)
    else:
        # --- Interactive Mode ---
        interactive_args = run_interactive_setup()
        run_pipeline(interactive_args)