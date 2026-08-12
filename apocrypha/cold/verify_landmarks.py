import torch
import numpy as np
import experiment_runner
import config

# --- The Verification Logic ---

VERIFICATION_LAYER = 5

def verify_landmarks(
    target_neurons: list[int],
    candidate_prompts: list[str],
    protocol: str = 'implicit',
    mode: str = 'resonance'
):
    """
    Tests a suite of candidate prompts to find the best activator for each target landmark neuron.
    """
    model = experiment_runner.model
    if not model:
        raise RuntimeError("Model not loaded.")

    print(f"\n--- Running Landmark Verification Protocol ---")
    print(f"--- Protocol: [{protocol.upper()} / {mode.upper()}] ---")

    # --- Helper function ---
    def get_peak_activation(prompt: str) -> int:
        # This is a simplified version of our analysis helper
        if protocol == 'implicit':
            input_ids = model.to_tokens(prompt, prepend_bos=True).to(config.DEVICE)
        else:
            input_ids = model.to_tokens(prompt, prepend_bos=False).to(config.DEVICE)
        
        captured_activations = {}
        def capture_hook_fn(activation_tensor, hook):
            captured_activations['mlp_post'] = activation_tensor.detach().cpu().numpy()
        hook_name = f"blocks.{VERIFICATION_LAYER}.mlp.hook_post"
        
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, capture_hook_fn)]):
            model(input_ids)
        
        full_activation_tensor = captured_activations['mlp_post'][0, :, :]

        if mode == 'resonance':
            max_activations_per_neuron = np.max(full_activation_tensor, axis=0)
            return np.argmax(max_activations_per_neuron)
        else: # inference
            vector = full_activation_tensor[-1, :]
            return np.argmax(vector)

    # --- Main Verification Loop ---
    verified_prompts = {}

    for neuron_id in target_neurons:
        print(f"\n> Verifying best prompt for landmark: J5-{neuron_id}")
        best_prompt = None
        for prompt in candidate_prompts:
            peak_id = get_peak_activation(prompt)
            status = "MATCH" if peak_id == neuron_id else "MISS"
            print(f"  - Prompt: '{prompt}' -> Peak: J5-{peak_id}  ({status})")
            if peak_id == neuron_id:
                best_prompt = prompt
                break # Found a working prompt, move to the next neuron
        
        if best_prompt:
            verified_prompts[neuron_id] = best_prompt
        else:
            verified_prompts[neuron_id] = "!! No simple prompt found !!"

    print("\n\n--- VERIFICATION COMPLETE ---")
    print("Use this dictionary for your ablation experiments:")
    print("\nlandmarks_to_ablate = {")
    for nid, prompt in verified_prompts.items():
        print(f"    {nid}: \"{prompt}\",")
    print("}")

# --- Main Execution Block ---

if __name__ == "__main__":
    # The landmarks we need to find canonical prompts for
    LANDMARKS_TO_VERIFY = [1888, 1790, 1821]

    # A suite of simple, structural prompts to test
    CANDIDATE_PROMPTS = [
        ".", "?", "!", "\n\n", "the", "a", "is", "to",
        "The", "A", "Is", "To", " 's", " of", " in",
        "<|endoftext|>"
    ]

    verify_landmarks(LANDMARKS_TO_VERIFY, CANDIDATE_PROMPTS)