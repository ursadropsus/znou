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

def analyze_conceptual_axis(
    case_file: dict,
    protocol: str = 'implicit',
    mode: str = 'resonance',
    weight_factor: float = 1.0
):
    """
    Performs a full, rigorous dimensional analysis for a conceptual axis,
    generating a multi-part report with supporting visual evidence.
    """
    model = experiment_runner.model
    tokenizer = experiment_runner.tokenizer
    if not model or not tokenizer:
        raise RuntimeError("Model or tokenizer not loaded.")
        
    concept_name = case_file["name"]
    output_dir = os.path.join("constellations", concept_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Analyzing Conceptual Axis: '{concept_name.upper()}' ---")
    print(f"--- Protocol: [{protocol.upper()} / {mode.upper()}] ---")

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
    for neuron_id, canonical_prompt in case_file["landmarks"].items():
        landmark_vectors[neuron_id], _ = get_activation_vector(canonical_prompt)

    # --- 2. Process all probes to gather raw data ---
    all_probes = []
    for concept in case_file["axis_concepts"]:
        all_probes.extend(case_file["axis_concepts"][concept]["probes"])
        
    probe_data = {}
    print(f"\n  > Processing {len(all_probes)} total probes...")
    for prompt in all_probes:
        v_probe, original_peak = get_activation_vector(prompt)
        v_ablated = v_probe.copy()
        for landmark_vec in landmark_vectors.values():
            v_ablated -= (weight_factor * landmark_vec)
        top_k_indices = np.argsort(v_ablated)[-TOP_K_VALUE:][::-1]
        probe_data[prompt] = { "v_probe": v_probe, "original_peak": original_peak, "v_ablated": v_ablated, "ablated_top_k": top_k_indices }
    
    # --- 3. Generate Reports ---
    generate_dimensional_report(case_file, probe_data, all_probes, output_dir)

def generate_dimensional_report(case_file, probe_data, all_probes, output_dir):
    print("\n  > Generating reports and visualizations...")
    report_path = os.path.join(output_dir, f"{case_file['name']}_report.txt")

    with open(report_path, "w") as f:
        f.write(f"--- Dimensional Analysis Report: {case_file['name'].upper()} ---\n\n")
        f.write("This report details a multi-faceted analysis of a conceptual axis within the model's latent space.\n")

        # --- I. Ablation Fingerprint Analysis ---
        f.write("\n--- I. Ablation Fingerprint Analysis ---\n")
        f.write(f"This section identifies the core neurons for each facet after ablating landmarks.\n")
        
        fingerprints = {}
        for concept, data in case_file["axis_concepts"].items():
            f.write(f"\n-- Fingerprint for: {concept.upper()} --\n")
            all_top_k_for_concept = []
            for prompt in data["probes"]:
                all_top_k_for_concept.extend(probe_data[prompt]["ablated_top_k"])
            
            counts = Counter(all_top_k_for_concept)
            most_common = counts.most_common(15)
            fingerprints[concept] = set(nid for nid, count in most_common)

            f.write("Rank | Neuron ID | Frequency\n")
            f.write("---- | --------- | ---------\n")
            for i, (neuron_id, count) in enumerate(most_common):
                highlight = "<-- TARGET" if neuron_id == case_file.get("target_neuron") else ""
                f.write(f"#{i+1:<3} | J5-{neuron_id:<5} | {count:<2} {highlight}\n")

            # Generate and save individual fingerprint chart
            neuron_labels = [f"J5-{nid}" for nid, count in most_common]
            frequencies = [count for nid, count in most_common]
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(neuron_labels, frequencies, color=data["color"])
            ax.invert_yaxis()
            ax.set_xlabel('Frequency in Top-30')
            ax.set_title(f"Ablation Fingerprint for: '{concept.upper()}'")
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            for bar in bars:
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2., 
                       f'{int(bar.get_width())}', va='center')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{case_file['name']}_{concept}_fingerprint.png"))
            plt.close()

        # --- II. Shared Circuitry Analysis ---
        f.write("\n\n--- II. Shared Circuitry Analysis ---\n")
        f.write("This section identifies neurons shared across multiple facets of the concept.\n\n")
        
        # For Tsalal, look for overlap across the void facets
        void_facets = ['epistemic_void', 'ontological_void', 'sensory_void', 'cognitive_limit']
        existing_void_facets = [vf for vf in void_facets if vf in fingerprints]
        
        if len(existing_void_facets) >= 2:
            # Find intersection of all void facets
            shared_across_voids = fingerprints[existing_void_facets[0]].copy()
            for facet in existing_void_facets[1:]:
                shared_across_voids &= fingerprints[facet]
            
            if shared_across_voids:
                f.write(f"Core Tsalal Circuit (shared across {len(existing_void_facets)} void facets):\n")
                f.write(", ".join(f"J5-{nid}" for nid in sorted(list(shared_across_voids))) + "\n\n")
            else:
                f.write("No neurons shared across all void facets.\n\n")
            
            # Also check pairwise overlaps
            f.write("Pairwise overlaps:\n")
            for i, facet1 in enumerate(existing_void_facets):
                for facet2 in existing_void_facets[i+1:]:
                    overlap = fingerprints[facet1] & fingerprints[facet2]
                    f.write(f"  {facet1} ∩ {facet2}: {len(overlap)} neurons\n")

        # --- III. Cosine Similarity Analysis ---
        probe_vectors = [probe_data[p]["v_probe"] for p in all_probes]
        similarity_matrix = cosine_similarity(probe_vectors)
        f.write("\n\n--- III. Cosine Similarity Analysis (Un-Ablated Vectors) ---\n")
        
        # Save visual heatmap
        plt.figure(figsize=(20, 18))
        sns.heatmap(similarity_matrix, xticklabels=all_probes, yticklabels=all_probes, 
                   cmap="viridis", annot=False)
        plt.title(f"Activation Vector Cosine Similarity for: '{case_file['name'].upper()}'")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{case_file['name']}_heatmap.png"))
        plt.close()
        
        # Save text-based heatmap
        with open(os.path.join(output_dir, f"{case_file['name']}_similarity_matrix.txt"), "w") as hf:
            header = "," + ",".join(f'"{p}"' for p in all_probes) + "\n"
            hf.write(header)
            for i, prompt in enumerate(all_probes):
                row = f'"{prompt}",' + ",".join(f"{score:.4f}" for score in similarity_matrix[i]) + "\n"
                hf.write(row)
        f.write("Visual heatmap and machine-readable similarity_matrix.txt have been generated.\n")

        # --- IV. Detailed Probe-by-Probe Breakdown ---
        f.write("\n\n--- IV. Detailed Probe-by-Probe Ablation Results ---\n")
        for concept, data in case_file["axis_concepts"].items():
            f.write(f"\n-- {concept.upper()} PROBES --\n")
            for prompt in data["probes"]:
                p_data = probe_data[prompt]
                f.write(f"\n> PROMPT: \"{prompt}\"\n")
                f.write(f"  Original Peak: J5-{p_data['original_peak']}\n")
                f.write("  Ablated Top-K Results:\n")
                peak_ablated = np.max(p_data['v_ablated'])
                for i, nid in enumerate(p_data['ablated_top_k']):
                    strength = (p_data['v_ablated'][nid] / peak_ablated) * 100
                    highlight = "  <-- TARGET" if nid == case_file.get("target_neuron") else ""
                    f.write(f"    #{i+1:<2}: J5-{nid:<4} (Strength: {strength:6.2f}%) {highlight}\n")

    print(f"\nSUCCESS: Full report and graphs saved to '{output_dir}'")

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- The Tsalal Case File ---
    CASE_FILE = {
        "name": "AXIS_OF_TSALAL",
        "target_neuron": 938,  # unimaginable/unnameable specialist
        "axis_concepts": {
            "epistemic_void": {
                "name": "Unknowable", 
                "color": "#1a0033",
                "probes": [
                    "unknowable", "unthinkable", "inconceivable", "unfathomable",
                    "unimaginable", "unnameable", "indescribable", "imperceptible",
                    "incogitable", "mind-boggling", "beyond wildest dreams",
                    "not understandable", "there is nothing to understand",
                    "impossible to comprehend", "cannot be grasped"
                ]
            },
            "ontological_void": {
                "name": "Nothingness",
                "color": "#000000",
                "probes": [
                    "nothing", "void", "nothingness", "absence", "non-existence",
                    "there is nothing", "absolute nothingness", "total void",
                    "nihility", "nihil", "annihilation", "oblivion",
                    "null", "nullity", "vacancy", "vacuity"
                ]
            },
            "sensory_void": {
                "name": "Darkness/Abyss",
                "color": "#0d0d0d",
                "probes": [
                    "darkness", "dark", "shadow", "shadowy", "abyss", "abyssal",
                    "night", "nocturnal", "blackness", "dim", "desolation", "desolate",
                    "hollow", "emptiness", "silence", "vacuum"
                ]
            },
            "cognitive_limit": {
                "name": "Beyond Understanding",
                "color": "#330066",
                "probes": [
                    "strange", "unheard-of", "impossible", "improbable",
                    "singular", "unmaking", "beyond comprehension",
                    "transcendent horror", "cosmic dread", "maddening",
                    "defies explanation", "surpasses understanding"
                ]
            },
            "affirmation": {
                "name": "Presence/Being",
                "color": "#ffffff",
                "probes": [
                    "everything", "all", "totality", "presence", "being",
                    "there is everything", "absolute presence", "fullness",
                    "existence", "reality", "substance", "abundance"
                ]
            },
            "knowable": {
                "name": "Comprehensible",
                "color": "#6699cc",
                "probes": [
                    "understandable", "knowable", "conceivable", "thinkable",
                    "clear", "obvious", "transparent", "comprehensible",
                    "graspable", "explicable", "rational", "logical"
                ]
            },
            "light": {
                "name": "Illumination",
                "color": "#ffff99",
                "probes": [
                    "light", "bright", "glowing", "shining", "incandescent",
                    "luminous", "radiant", "brilliant", "dazzling", "blazing"
                ]
            }
        },
        "landmarks": { 1888: ".", 1790: "?" }  # Update these after running verify_landmarks.py
    }

    analyze_conceptual_axis(CASE_FILE)