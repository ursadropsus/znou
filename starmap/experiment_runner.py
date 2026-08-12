# experiment_runner.py
# --- MODIFIED: Refactored for 4-quadrant (Implicit/Explicit) analysis ---

import torch
import numpy as np
import time
import traceback
import datetime
from collections import defaultdict
from pathlib import Path
from transformer_lens import HookedTransformer, utils as tl_utils

import config
import utils_game

# --- Global Storage ---
model = None; tokenizer = None; basis_u1 = None; basis_u2 = None; current_basis_path: Path | None = None; basis_load_error: str | None = None; active_intervention_rules: dict = {}; interventions_enabled: bool = False

# --- Model & Basis Loading (Unchanged) ---
def load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        print(f"[Runner] Loading model '{config.MODEL_NAME}' onto device '{config.DEVICE}'...")
        try:
            model = HookedTransformer.from_pretrained(config.MODEL_NAME); model.to(config.DEVICE); model.eval()
            tokenizer = model.tokenizer
            if tokenizer.pad_token is None:
                print("[Runner] Tokenizer pad token is None, setting to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            if hasattr(model.cfg, 'pad_token_id') and tokenizer.pad_token_id is not None:
                 model.cfg.pad_token_id = tokenizer.pad_token_id
            print("[Runner] Model and tokenizer loaded successfully.")
        except ImportError: print("\n\nERROR: `transformer_lens` or `torch` missing.\nInstall: pip install transformer_lens torch\n\n"); raise
        except Exception as e: print(f"[Runner] Error loading model: {e}"); traceback.print_exc(); raise RuntimeError(f"Failed to load model '{config.MODEL_NAME}'.") from e

# --- All other experimental functions (load_basis_vectors, interventions, etc.) remain unchanged ---
# ... (functions from load_basis_vectors down to get_activations_for_prompt_aggregated are identical) ...
def load_basis_vectors(new_basis_path: Path | None = None) -> bool:
    global basis_u1, basis_u2, current_basis_path, basis_load_error
    path_to_load = new_basis_path
    if path_to_load is None:
        if current_basis_path is not None or basis_load_error is not None:
            return basis_u1 is not None and basis_u2 is not None
        else:
            basis_load_error = "No basis file specified or previously loaded."
            return False
    print(f"[Runner] Attempting to load basis vectors from: {path_to_load}")
    basis_u1 = None; basis_u2 = None; current_basis_path = None; basis_load_error = None
    if not isinstance(path_to_load, Path):
         try: path_to_load = Path(str(path_to_load))
         except TypeError: basis_load_error = "Invalid path type for basis file."; print(f"[Runner] Error: {basis_load_error}"); return False
    if not path_to_load.is_file():
        basis_load_error = f"Basis file not found: {path_to_load}"
        print(f"[Runner] Error: {basis_load_error}"); return False
    try:
        with np.load(path_to_load, allow_pickle=True) as data:
            if 'basis_1' not in data or 'basis_2' not in data:
                basis_load_error = "Keys 'basis_1' and/or 'basis_2' missing in basis file."
            else:
                b1r, b2r = data['basis_1'], data['basis_2']
                if b1r.shape != (config.DIMENSION,) or b2r.shape != (config.DIMENSION,):
                    basis_load_error = f"Basis dimension mismatch. Expected ({config.DIMENSION},), got {b1r.shape} and {b2r.shape}."
                else:
                     u1_cand, u2_cand = utils_game.create_orthonormal_basis(b1r, b2r)
                     if u1_cand is not None and u2_cand is not None:
                         basis_u1 = u1_cand; basis_u2 = u2_cand; current_basis_path = path_to_load
                         print("[Runner] Orthonormal basis prepared successfully.")
                         return True
                     else:
                         basis_load_error = "Failed to create orthonormal basis (vectors might be collinear or zero)."
    except Exception as e:
        basis_load_error = f"Failed to load/process basis file: {e}"
        traceback.print_exc()
    print(f"[Runner] Error loading basis: {basis_load_error}"); return False
def set_active_interventions(parsed_rules: dict, is_enabled: bool):
    global active_intervention_rules, interventions_enabled
    if isinstance(parsed_rules, dict):
        active_intervention_rules = parsed_rules
        interventions_enabled = is_enabled
        if interventions_enabled and active_intervention_rules:
            print(f"[Runner] Interventions ENABLED ({len(active_intervention_rules)} layers with rules).")
        elif interventions_enabled and not active_intervention_rules:
            print("[Runner] Interventions enabled by toggle, but NO valid rules are currently defined.")
        else:
            print("[Runner] Interventions DISABLED.")
    else:
        print("[Runner Error] Invalid intervention rules type received. Disabling interventions.")
        active_intervention_rules = {}
        interventions_enabled = False
def get_intervention_status() -> tuple[bool, dict]:
    return interventions_enabled, active_intervention_rules
def dynamic_intervention_hook(activation_tensor: torch.Tensor, hook):
    global active_intervention_rules, interventions_enabled
    if not interventions_enabled or not active_intervention_rules:
        return activation_tensor
    try:
        layer_index = int(hook.name.split('.')[1])
        if layer_index in active_intervention_rules:
            rules_for_layer = active_intervention_rules[layer_index]
            if not rules_for_layer: return activation_tensor
            last_token_index = activation_tensor.shape[1] - 1
            if last_token_index < 0:
                return activation_tensor
            for rule in rules_for_layer:
                neuron_idx = rule['neuron']
                mode = rule['mode']
                threshold = rule['threshold']
                target_val = rule['target']
                if 0 <= neuron_idx < activation_tensor.shape[2]:
                    current_val = activation_tensor[0, last_token_index, neuron_idx].item()
                    apply_intervention = False
                    if mode == 'override':
                        apply_intervention = True
                    elif mode == 'neg_floor' and current_val < threshold:
                        apply_intervention = True
                    elif mode == 'pos_ceil' and current_val > threshold:
                        apply_intervention = True
                    if apply_intervention:
                        target_tensor = torch.tensor(target_val, dtype=activation_tensor.dtype, device=activation_tensor.device)
                        activation_tensor[0, last_token_index, neuron_idx] = target_tensor
    except Exception as e:
        print(f"\n!! Error in dynamic_intervention_hook for {hook.name}: {e}")
        traceback.print_exc()
    return activation_tensor
def capture_activations(model_input, target_layers, existing_captured_data_dict=None):
    if model is None: raise RuntimeError("Model not loaded for activation capture.")
    captured_data = existing_captured_data_dict if existing_captured_data_dict is not None else {}
    hooks = []
    def capture_hook_fn(activation_tensor, hook):
        try:
            layer_idx = int(hook.name.split('.')[1])
            captured_data[layer_idx] = activation_tensor.detach().cpu()
        except Exception as e_hook_cap:
            print(f"\nError in capture_hook_fn for {hook.name}: {e_hook_cap}")
            traceback.print_exc()
    for layer_idx_cap in target_layers:
        hooks.append((tl_utils.get_act_name("post", layer_idx_cap), capture_hook_fn))
    if not hooks:
        print("[Runner] Warning: No capture hooks prepared for this capture_activations call.")
        return captured_data
    try:
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            _ = model(model_input, return_type=None)
    except Exception as e_fwd_pass:
        print(f"\nError during activation capture forward pass: {e_fwd_pass}")
        traceback.print_exc()
        return captured_data
    return captured_data
def summarize_activations(captured_activations_dict):
    summary = {};
    if not captured_activations_dict: return summary
    q_tensor_last = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    for layer_idx, activation_tensor in captured_activations_dict.items():
        if not isinstance(activation_tensor, torch.Tensor):
            summary[layer_idx] = {"error": "Invalid tensor type"}; continue
        acts_float = activation_tensor.float()
        if acts_float.ndim != 3:
            if acts_float.numel() == 0 and acts_float.ndim == 1:
                summary[layer_idx] = {"error": "Invalid or empty activation tensor"}
            else:
                summary[layer_idx] = {"error": f"Unexpected tensor ndim {acts_float.ndim}"}
            continue
        if acts_float.shape[0] != 1:
             acts_float = acts_float[0:1, :, :]
        seq_len = acts_float.shape[1]
        if seq_len == 0:
            summary[layer_idx] = {"error": "Zero sequence length"}; continue
        if acts_float.numel() == 0:
            summary[layer_idx] = {"error": "Invalid or empty activation tensor (zero elements after initial checks)"}; continue
        try:
            d_feature = acts_float.shape[2]
            layer_mean = acts_float.mean().item()
            layer_max = acts_float.max().item()
            max_val_per_neuron_across_tokens, _ = acts_float[0].max(dim=0)
            max_activating_neuron_idx = max_val_per_neuron_across_tokens.argmax().item()
            last_token_activations = acts_float[0, -1, :]
            last_token_mean = last_token_activations.mean().item()
            last_token_stddev = last_token_activations.std().item()
            last_token_quantiles = torch.quantile(last_token_activations, q=q_tensor_last).tolist()
            peak_neuron_last_token_idx = -1
            if last_token_activations.numel() > 0:
                peak_neuron_last_token_idx_tensor = last_token_activations.argmax(dim=0)
                peak_neuron_last_token_idx = peak_neuron_last_token_idx_tensor.item()
            stats = {
                "mean_activation_all_tokens": layer_mean,
                "max_activation_all_tokens": layer_max,
                "max_activating_neuron_idx": max_activating_neuron_idx,
                "peak_neuron_last_token_idx": peak_neuron_last_token_idx,
                "last_token_stats": {
                    "mean": last_token_mean, "stddev": last_token_stddev,
                    "median": last_token_quantiles[1],
                    "percentile_10th": last_token_quantiles[0],
                    "percentile_90th": last_token_quantiles[2]
                }
            }
            summary[layer_idx] = stats
        except Exception as e_layer:
            print(f"!! Error summarizing L{layer_idx}: {e_layer}"); traceback.print_exc()
            summary[layer_idx] = {"error": str(e_layer)}
    return summary

# --- NEW/REFACTORED QUADRANT ANALYSIS FUNCTIONS ---

def _get_all_neuron_ids_for_prompt(prompt_text: str, layer_idx: int) -> dict | None:
    """
    Internal master function to get all four neuron IDs in two efficient passes.
    Returns a dictionary like: {'imp_res': 1, 'imp_inf': 2, 'exp_res': 3, 'exp_inf': 4}
    """
    if model is None or tokenizer is None:
        print("[Runner Error] Master function: Model or tokenizer not loaded.", file=sys.stderr)
        return None

    results = {}

    try:
        # --- Pass 1: Implicit Protocol (String Pass) ---
        captured_acts_implicit = capture_activations(prompt_text, [layer_idx])
        summary_implicit = summarize_activations(captured_acts_implicit)
        if summary_implicit and layer_idx in summary_implicit and "error" not in summary_implicit[layer_idx]:
            results['imp_res'] = summary_implicit[layer_idx].get("max_activating_neuron_idx")
            results['imp_inf'] = summary_implicit[layer_idx].get("peak_neuron_last_token_idx")
        else:
            results['imp_res'], results['imp_inf'] = None, None

        # --- Pass 2: Explicit Protocol (Tensor Pass) ---
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(config.DEVICE)
        if input_ids.shape[1] > 0:
            captured_acts_explicit = capture_activations(input_ids, [layer_idx])
            summary_explicit = summarize_activations(captured_acts_explicit)
            if summary_explicit and layer_idx in summary_explicit and "error" not in summary_explicit[layer_idx]:
                results['exp_res'] = summary_explicit[layer_idx].get("max_activating_neuron_idx")
                results['exp_inf'] = summary_explicit[layer_idx].get("peak_neuron_last_token_idx")
            else:
                results['exp_res'], results['exp_inf'] = None, None
        else: # Handle empty prompt case
            results['exp_res'], results['exp_inf'] = None, None
            
        return results

    except Exception as e:
        print(f"\n[Runner Error] An exception occurred in _get_all_neuron_ids_for_prompt: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

def get_peak_neuron_for_prompt(prompt_text: str, layer_idx: int, protocol: str, mode: str) -> int | None:
    """
    Public-facing wrapper. Gets the single requested neuron ID by calling the master function.
    This is the primary entry point for the API server.
    """
    if protocol not in ['implicit', 'explicit'] or mode not in ['resonance', 'inference']:
        print(f"[Runner Error] Invalid protocol '{protocol}' or mode '{mode}'.", file=sys.stderr)
        return None

    all_ids = _get_all_neuron_ids_for_prompt(prompt_text, layer_idx)

    if all_ids is None:
        return None

    # Construct the key, e.g., 'imp_res'
    protocol_prefix = 'imp' if protocol == 'implicit' else 'exp'
    mode_suffix = 'res' if mode == 'resonance' else 'inf'
    key = f"{protocol_prefix}_{mode_suffix}"
    
    return all_ids.get(key)


# --- Remaining experimental functions are unchanged ---
def get_mean_activation_for_prompt(prompt_text: str, layer_idx: int) -> np.ndarray | None:
    if model is None or tokenizer is None:
        print("[Runner Error] Model or tokenizer not loaded. Cannot get activations."); return None
    if not (0 <= layer_idx < model.cfg.n_layers):
        print(f"[Runner Error] Invalid layer_idx {layer_idx} for model with {model.cfg.n_layers} layers."); return None
    try:
        input_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=model.cfg.n_ctx)["input_ids"].to(config.DEVICE)
        if input_ids.shape[1] == 0:
            print(f"[Runner Error] Prompt '{prompt_text[:30]}...' tokenized to empty sequence."); return None
    except Exception as e_tok:
        print(f"[Runner Error] Tokenization failed for prompt '{prompt_text[:30]}...': {e_tok}"); return None
    hook_name = tl_utils.get_act_name("post", layer_idx)
    captured_act_container = {'activation': None}
    def single_capture_hook_fn(activation_tensor, hook):
        captured_act_container['activation'] = activation_tensor.detach().cpu()
    print(f"[Runner Activation] Getting mean activation for layer {layer_idx} from prompt: '{prompt_text[:30]}...'")
    try:
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, single_capture_hook_fn)]):
            model(input_ids)
    except Exception as e_fwd:
        print(f"[Runner Error] Forward pass failed for activation capture: {e_fwd}"); traceback.print_exc(); return None
    captured_act = captured_act_container['activation']
    if captured_act is None:
        print(f"[Runner Error] Failed to capture activation for layer {layer_idx}."); return None
    if not (captured_act.ndim == 3 and captured_act.shape[0] == 1 and captured_act.shape[2] == config.DIMENSION):
        print(f"[Runner Error] Unexpected activation shape. Expected [1, seq, {config.DIMENSION}], got {captured_act.shape}"); return None
    mean_vec = captured_act.float().mean(dim=1)[0].numpy()
    if mean_vec.shape == (config.DIMENSION,):
        return mean_vec.astype(np.float32)
    else:
        print(f"[Runner Error] Mean vector shape mismatch. Expected ({config.DIMENSION},), got {mean_vec.shape}"); return None
def get_activations_for_prompt_aggregated(prompt_text: str, layer_idx: int, aggregation_methods: list[str]) -> dict[str, np.ndarray | None]:
    if model is None or tokenizer is None:
        print("[Runner Error] Model or tokenizer not loaded. Cannot get aggregated activations."); return {method: None for method in aggregation_methods}
    if not (0 <= layer_idx < model.cfg.n_layers):
        print(f"[Runner Error] Invalid layer_idx {layer_idx} for model with {model.cfg.n_layers} layers."); return {method: None for method in aggregation_methods}
    try:
        input_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=model.cfg.n_ctx)["input_ids"].to(config.DEVICE)
        if input_ids.shape[1] == 0:
            print(f"[Runner Error] Prompt '{prompt_text[:30]}...' tokenized to empty sequence."); return {method: None for method in aggregation_methods}
    except Exception as e_tok:
        print(f"[Runner Error] Tokenization failed for prompt '{prompt_text[:30]}...': {e_tok}"); return {method: None for method in aggregation_methods}
    hook_name = tl_utils.get_act_name("post", layer_idx)
    captured_act_container = {'activation': None}
    def single_capture_hook_fn(activation_tensor, hook):
        captured_act_container['activation'] = activation_tensor.detach().cpu()
    try:
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, single_capture_hook_fn)]):
            model(input_ids)
    except Exception as e_fwd:
        print(f"[Runner Error] Forward pass failed for aggregated activation capture: {e_fwd}"); traceback.print_exc(); return {method: None for method in aggregation_methods}
    raw_activation_tensor = captured_act_container['activation']
    aggregated_results = {}
    if raw_activation_tensor is None:
        print(f"[Runner Error] Failed to capture activation for layer {layer_idx} for aggregation.")
        for method in aggregation_methods: aggregated_results[method] = None
        return aggregated_results
    if not (raw_activation_tensor.ndim == 3 and raw_activation_tensor.shape[0] == 1 and raw_activation_tensor.shape[2] == config.DIMENSION):
        print(f"[Runner Error] Unexpected raw activation shape for aggregation. Expected [1, seq, {config.DIMENSION}], got {raw_activation_tensor.shape}")
        for method in aggregation_methods: aggregated_results[method] = None
        return aggregated_results
    raw_activation_tensor_float = raw_activation_tensor.float()
    if "mean" in aggregation_methods:
        mean_vec = raw_activation_tensor_float.mean(dim=1)[0].numpy()
        if mean_vec.shape == (config.DIMENSION,):
            aggregated_results["mean"] = mean_vec.astype(np.float32)
        else:
            print(f"[Runner Error] Mean vector shape mismatch for aggregation. Expected ({config.DIMENSION},), got {mean_vec.shape}"); aggregated_results["mean"] = None
    if "last_token" in aggregation_methods:
        sequence_length = raw_activation_tensor_float.shape[1]
        if sequence_length > 0:
            last_token_vec = raw_activation_tensor_float[0, -1, :].numpy()
            if last_token_vec.shape == (config.DIMENSION,):
                aggregated_results["last_token"] = last_token_vec.astype(np.float32)
            else:
                print(f"[Runner Error] Last token vector shape mismatch for aggregation. Expected ({config.DIMENSION},), got {last_token_vec.shape}"); aggregated_results["last_token"] = None
        else:
            print(f"[Runner Warn] Zero sequence length in raw_activation_tensor, cannot get 'last_token' for '{prompt_text[:30]}...'."); aggregated_results["last_token"] = None
    for method in aggregation_methods:
        if method not in aggregated_results:
            print(f"[Runner Warn] Aggregation method '{method}' not implemented or failed, returning None."); aggregated_results[method] = None
    return aggregated_results
def calculate_prompt_projection(activation_tensor_layer: torch.Tensor | None) -> dict | None:
    if basis_u1 is None or basis_u2 is None: return None
    if activation_tensor_layer is None or not isinstance(activation_tensor_layer, torch.Tensor) or activation_tensor_layer.numel() == 0:
        return None
    try:
        if activation_tensor_layer.ndim != 3 or activation_tensor_layer.shape[0] != 1: return None
        mean_vec = activation_tensor_layer.float().mean(dim=1)[0].numpy()
        if mean_vec.shape != (config.DIMENSION,): return None
        norm_mean_vec = utils_game.normalise(mean_vec)
        proj_x = np.dot(norm_mean_vec, basis_u1); proj_y = np.dot(norm_mean_vec, basis_u2)
        r = np.sqrt(proj_x**2 + proj_y**2); theta_rad = np.arctan2(proj_y, proj_x)
        theta_deg = (np.degrees(theta_rad) + 360) % 360
        return {"angle_deg": theta_deg, "r": r, "x": proj_x, "y": proj_y}
    except Exception as e:
        print(f"[Runner] Error calculating prompt projection: {e}"); traceback.print_exc(); return None
def calculate_final_state_alignment(activation_tensor_layer: torch.Tensor | None) -> dict | None:
    if basis_u1 is None or basis_u2 is None: return None
    if activation_tensor_layer is None or not isinstance(activation_tensor_layer, torch.Tensor) or activation_tensor_layer.numel() == 0:
        return None
    try:
        if activation_tensor_layer.ndim != 3 or activation_tensor_layer.shape[0] != 1: return None
        last_vec = activation_tensor_layer.float()[0, -1, :].numpy()
        if last_vec.shape != (config.DIMENSION,): return None
        norm_last_vec = utils_game.normalise(last_vec)
        sim_u1 = np.dot(norm_last_vec, basis_u1); sim_u2 = np.dot(norm_last_vec, basis_u2)
        return {"sim_basis1": sim_u1, "sim_basis2": sim_u2}
    except Exception as e:
        print(f"[Runner] Error calculating final state alignment: {e}"); traceback.print_exc(); return None
def run_experiment(prompt_text: str, generation_params: dict) -> dict:
    global active_intervention_rules, interventions_enabled
    if model is None or tokenizer is None: raise RuntimeError("Model not loaded.")
    print(f"[Runner] Received prompt: '{prompt_text[:50]}...'"); start_time = time.time()
    try:
        input_ids_prompt = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(config.DEVICE)
        input_len_prompt = input_ids_prompt.shape[1]
    except Exception as e: print(f"[Runner] Tokenization Error: {e}"); return {"error": f"Tokenization failed: {e}"}
    if input_len_prompt == 0: print("[Runner] Error: Empty tokenized input."); return {"error": "Input tokenized empty."}
    print("[Runner] Capturing BASELINE activations (prompt only)...")
    captured_acts_baseline = capture_activations(input_ids_prompt, config.TARGET_LAYERS)
    activation_summary_baseline = summarize_activations(captured_acts_baseline) if captured_acts_baseline else {}
    analysis_layer_acts_baseline = captured_acts_baseline.get(config.ANALYSIS_LAYER) if captured_acts_baseline else None
    prompt_projection_baseline = None; final_alignment_baseline = None
    if analysis_layer_acts_baseline is not None:
        prompt_projection_baseline = calculate_prompt_projection(analysis_layer_acts_baseline)
        final_alignment_baseline = calculate_final_state_alignment(analysis_layer_acts_baseline)
    gen_mode_desc = "Sampling" if generation_params.get("do_sample", False) else "Greedy"
    print(f"[Runner] Generating text ({gen_mode_desc})...");
    output_ids = input_ids_prompt; generation_fwd_hooks = []; intervention_layers_applied_during_gen = []
    if interventions_enabled and active_intervention_rules:
         print("[Runner] Applying interventions DURING generation...")
         for layer_idx in active_intervention_rules.keys():
              if active_intervention_rules[layer_idx]:
                   hook_point_name = tl_utils.get_act_name("post", layer_idx)
                   generation_fwd_hooks.append((hook_point_name, dynamic_intervention_hook))
                   intervention_layers_applied_during_gen.append(layer_idx)
         if generation_fwd_hooks: print(f"[Runner] Intervention hooks for generation: L{sorted(list(set(intervention_layers_applied_during_gen)))}")
    generated_text = ""
    try:
        with torch.no_grad(), model.hooks(fwd_hooks=generation_fwd_hooks):
             output_ids = model.generate(input_ids_prompt, max_new_tokens=config.GENERATION_LENGTH, eos_token_id=tokenizer.eos_token_id, **generation_params)
        generated_ids = output_ids[0, input_len_prompt:]; generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"[Runner] Generated text: '{generated_text[:100]}...'")
    except Exception as e: print(f"[Runner] Generation Error: {e}"); traceback.print_exc(); generated_text = f"[Generation Error: {e}]"
    activation_summary_intervened = {}; analysis_layer_acts_intervened = None
    prompt_projection_intervened = None; final_alignment_intervened = None
    if interventions_enabled and active_intervention_rules:
        print("[Runner] Capturing INTERVENED activations (full output, with intervention hooks re-applied)...")
        hooks_for_intervened_pass = []; captured_data_intervened_dict = {}
        def capture_hook_fn_intervened(activation_tensor, hook):
            try: layer_idx_int_cap = int(hook.name.split('.')[1]); captured_data_intervened_dict[layer_idx_int_cap] = activation_tensor.detach().cpu()
            except Exception as e_hook_int: print(f"\nError in capture_hook_fn_intervened for {hook.name}: {e_hook_int}")
        for layer_idx_cap_int in config.TARGET_LAYERS:
            hooks_for_intervened_pass.append((tl_utils.get_act_name("post", layer_idx_cap_int), capture_hook_fn_intervened))
        for layer_idx_int_rule in active_intervention_rules.keys():
            if active_intervention_rules[layer_idx_int_rule]:
                 hooks_for_intervened_pass.append((tl_utils.get_act_name("post", layer_idx_int_rule), dynamic_intervention_hook))
        if hooks_for_intervened_pass:
            try:
                with torch.no_grad(), model.hooks(fwd_hooks=hooks_for_intervened_pass):
                    _ = model(output_ids, return_type=None)
                activation_summary_intervened = summarize_activations(captured_data_intervened_dict)
                analysis_layer_acts_intervened = captured_data_intervened_dict.get(config.ANALYSIS_LAYER)
                if analysis_layer_acts_intervened is not None:
                    prompt_projection_intervened = calculate_prompt_projection(analysis_layer_acts_intervened)
                    final_alignment_intervened = calculate_final_state_alignment(analysis_layer_acts_intervened)
            except Exception as e_int_cap_pass:
                print(f"[Runner] Error during intervened activation capture pass: {e_int_cap_pass}"); traceback.print_exc()
        else: print("[Runner] No hooks prepared for intervened pass, skipping intervened stats.")
    else: print("[Runner] Interventions not active or no rules, skipping intervened activation capture.")
    collapse_detected = utils_game.is_output_collapsed(generated_text)
    print(f"[Runner] Collapse detected: {collapse_detected}")
    raw_results = {
        "run_timestamp": datetime.datetime.now().isoformat(), "input_prompt": prompt_text, "generated_text": generated_text,
        "activation_stats_baseline": activation_summary_baseline, "prompt_vector_projection_baseline": prompt_projection_baseline, "final_state_alignment_baseline": final_alignment_baseline,
        "activation_stats_intervened": activation_summary_intervened if (interventions_enabled and active_intervention_rules) else None,
        "prompt_vector_projection_intervened": prompt_projection_intervened if (interventions_enabled and active_intervention_rules) else None,
        "final_state_alignment_intervened": final_alignment_intervened if (interventions_enabled and active_intervention_rules) else None,
        "collapse_detected": collapse_detected, "basis_file_used": str(current_basis_path) if current_basis_path else None, "basis_load_error": basis_load_error,
        "interventions_enabled_during_run": interventions_enabled, "interventions_applied_rules": active_intervention_rules if interventions_enabled else None,
        "interventions_applied_layers_during_gen": sorted(list(set(intervention_layers_applied_during_gen))) if interventions_enabled else None,
        "generation_params_used": generation_params, "srm_peak": None, "text_fragment": generated_text[:50] + "..." if generated_text else "", "anomaly_flags": []
    }
    end_time = time.time(); print(f"[Runner] Experiment finished in {end_time - start_time:.2f} seconds.")
    return raw_results
def get_current_basis_info() -> tuple[Path | None, str | None]:
     return current_basis_path, basis_load_error
def get_model_n_layers() -> int | None:
    if model and hasattr(model, 'cfg') and hasattr(model.cfg, 'n_layers'):
        return model.cfg.n_layers
    return None

load_model_and_tokenizer()