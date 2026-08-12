import torch
import numpy as np
import experiment_runner
import config
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- The Core Analysis Logic ---

ANALYSIS_TARGET_LAYER = 5
TOP_K_VALUE = 30

def map_constellation(
    concept_name: str,
    target_neuron: int,
    probes: dict,
    ablation_targets: dict[int, str],
    output_dir: str,
    protocol: str = 'implicit',
    mode: str = 'resonance',
    weight_factor: float = 1.0
):
    """
    Performs a full, rigorous constellation analysis for a given concept,
    generating a multi-part report with supporting visual evidence.
    """
    model = experiment_runner.model
    tokenizer = experiment_runner.tokenizer
    if not model or not tokenizer:
        raise RuntimeError("Model or tokenizer not loaded.")

    print(f"\n--- Mapping Constellation for '{concept_name.upper()}' ---")
    print(f"--- Analysis Protocol: [{protocol.upper()} / {mode.upper()}] ---")

    # --- Helper function ---
    def get_activation_vector(prompt: str) -> tuple[np.ndarray, int]:
        if protocol == 'implicit':
            input_ids = model.to_tokens(prompt, prepend_bos=True).to(config.DEVICE)
        else:
            input_ids = model.to_tokens(prompt, prepend_bos=False).to(config.DEVICE)
        captured_activations = {}
        def capture_hook_fn(activation_tensor, hook):
            captured_activations['mlp_post'] = activation_tensor.detach().cpu().numpy()
        hook_name = f"blocks.{ANALYSIS_TARGET_LAYER}.mlp.hook_post"
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, capture_hook_fn)]): model(input_ids)
        full_activation_tensor = captured_activations['mlp_post'][0, :, :]
        if mode == 'resonance':
            max_activations_per_neuron = np.max(full_activation_tensor, axis=0)
            peak_neuron_id = np.argmax(max_activations_per_neuron)
            token_index_of_peak = np.argmax(full_activation_tensor[:, peak_neuron_id])
            return full_activation_tensor[token_index_of_peak, :], peak_neuron_id
        else:
            vector = full_activation_tensor[-1, :]
            peak_id = np.argmax(vector)
            return vector, peak_id

    # --- 1. Capture landmark vectors ---
    print("\n  > Capturing landmark vectors for ablation...")
    landmark_vectors = {}
    for neuron_id, canonical_prompt in ablation_targets.items():
        landmark_vectors[neuron_id], _ = get_activation_vector(canonical_prompt)

    # --- 2. Process all probes to gather raw data ---
    all_probes = probes['positive'] + probes['structural'] + probes['negative'] + probes['multilingual']
    probe_data = {}
    all_ablated_top_k = []

    print(f"\n  > Processing {len(all_probes)} total probes...")
    for prompt in all_probes:
        v_probe, original_peak = get_activation_vector(prompt)
        v_ablated = v_probe.copy()
        for landmark_vec in landmark_vectors.values():
            v_ablated -= (weight_factor * landmark_vec)
        
        top_k_indices = np.argsort(v_ablated)[-TOP_K_VALUE:][::-1]
        
        # We only count positive probes towards the main constellation fingerprint
        if prompt in probes['positive']:
            all_ablated_top_k.extend(top_k_indices)
        
        probe_data[prompt] = {
            "v_probe": v_probe,
            "original_peak": original_peak,
            "v_ablated": v_ablated,
            "ablated_top_k": top_k_indices
        }
    
    # --- 3. Generate Reports ---
    generate_reports(concept_name, target_neuron, probes, probe_data, all_ablated_top_k, output_dir)

def generate_reports(concept_name, target_neuron, probes, probe_data, all_ablated_top_k, output_dir):
    print("\n  > Generating reports and visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{concept_name}_report.txt")

    with open(report_path, "w") as f:
        f.write(f"--- Constellation Case File: {concept_name.upper()} ---\n\n")
        f.write("This report details a multi-faceted analysis of the model's internal representation of a concept.\n")
        f.write("Analysis includes Ablation Spectrometry and Cosine Similarity measurements.\n\n")

        # --- Ablation Fingerprint Analysis ---
        constellation_counts = Counter(all_ablated_top_k)
        most_common_neurons = constellation_counts.most_common(15)
        
        f.write("--- I. Ablation Fingerprint Summary ---\n")
        f.write(f"The following neurons appeared most frequently in the Top-{TOP_K_VALUE} across all {len(probes['positive'])} positive probes after ablating landmarks.\n\n")
        f.write("Rank | Neuron ID | Frequency\n")
        f.write("---- | --------- | ---------\n")
        for i, (neuron_id, count) in enumerate(most_common_neurons):
            highlight = "<-- TARGET" if neuron_id == target_neuron else ""
            f.write(f"#{i+1:<3} | J5-{neuron_id:<5} | {count:<2} {highlight}\n")

        # Create Fingerprint Chart
        neuron_labels = [f"J5-{nid}" for nid, count in most_common_neurons]
        frequencies = [count for nid, count in most_common_neurons]
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(neuron_labels, frequencies, color='#29eaff'); ax.invert_yaxis()
        ax.set_xlabel('Frequency (Number of Top-30 Appearances in Positive Probes)'); ax.set_title(f"Ablation Fingerprint for: '{concept_name.upper()}'")
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        for bar in bars:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2., f'{bar.get_width()}', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{concept_name}_fingerprint.png"))
        plt.close()

        # --- Cosine Similarity Analysis ---
        all_probes = probes['positive'] + probes['structural'] + probes['negative'] + probes['multilingual']
        probe_vectors = [probe_data[p]["v_probe"] for p in all_probes]
        similarity_matrix = cosine_similarity(probe_vectors)

        f.write("\n\n--- II. Cosine Similarity Analysis ---\n")
        f.write("This measures the directional similarity of the original, un-ablated activation vectors.\n")
        f.write("A score of 1.0 (bright yellow) means vectors are nearly identical; 0.0 (dark purple) means they are unrelated.\n")
        f.write("Key Insight: Note the bright squares forming between synonyms ('cold', 'freezing', 'icy') even when their ablated peaks differ.\n\n")
        
        # Create Heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(similarity_matrix, xticklabels=all_probes, yticklabels=all_probes, cmap="viridis", annot=False)
        plt.title(f"Activation Vector Cosine Similarity for: '{concept_name.upper()}'"); plt.xticks(rotation=90); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{concept_name}_heatmap.png"))
        plt.close()

        # --- Detailed Probe-by-Probe Breakdown ---
        f.write("\n\n--- III. Detailed Probe-by-Probe Ablation Results ---\n")
        for probe_type, prompt_list in probes.items():
            f.write(f"\n-- {probe_type.upper()} PROBES --\n")
            for prompt in prompt_list:
                data = probe_data[prompt]
                f.write(f"\n> PROMPT: \"{prompt}\"\n")
                f.write(f"  Original Peak: J5-{data['original_peak']}\n")
                f.write(f"  Ablated Top-{TOP_K_VALUE} Results:\n")
                peak_ablated_value = np.max(data['v_ablated'])
                for i, nid in enumerate(data['ablated_top_k']):
                    strength = (data['v_ablated'][nid] / peak_ablated_value) * 100
                    highlight = "  <-- TARGET" if nid == target_neuron else ""
                    f.write(f"    #{i+1:<2}: J5-{nid:<4} (Strength: {strength:6.2f}%) {highlight}\n")

    print(f"\nSUCCESS: Full report and graphs saved to '{output_dir}'")

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- The Case File ---
    # This dictionary now defines the entire experiment.
    CASE_FILE = {
        "name": "Physical_Cold",
        "target_neuron": 38,
        "probes": {
            "positive": [
                "It was cold", "cold", "it was freezing", "freezing",
                "it was icy", "icy", "it was frigid", "frigid",
                "it was chilly", "chilly", "winter is coming", "the arctic tundra",
                "a blast of cold air", "absolute zero",
            ],
            "structural": [
                "The day was cold", "I am cold", "She felt cold",
            ],
            "negative": [
                "It was hot", "hot", "a warm day", "the desert sun",
                "a happy dog", "the price of steel",
            ],
            "multilingual": [
                "Il faisait froid", # French: "It was cold"
                "hacía frío",      # Spanish: "it was cold"
                "寒い",            # Japanese: "samui" (cold)
            ]
        },
        "landmarks": { 1888: ".", 1790: "?" }
    }

    # --- Setup and Execution ---
    output_directory = os.path.join("constellations", CASE_FILE["name"])
    
    map_constellation(
        concept_name=CASE_FILE["name"],
        target_neuron=CASE_FILE["target_neuron"],
        probes=CASE_FILE["probes"],
        ablation_targets=CASE_FILE["landmarks"],
        output_dir=output_directory
    )