# utils_game.py

import datetime
import json
import traceback
import uuid
from pathlib import Path
import numpy as np
import re 
from collections import Counter, defaultdict 
import math
import warnings # For PCA variance warnings

from sklearn.decomposition import PCA # For PCA method

# Assuming config.py is in the same directory or accessible in the Python path
# If config.py is in the parent directory, you might need:
# from .. import config # Or adjust based on your project structure
# For now, let's assume it's directly importable or these are defined if config can't be reached here
try:
    from . import config # If utils_game is part of a package
except ImportError:
    import config # If running as a script in the same directory as config.py


DIMENSION = config.DIMENSION
EPSILON = config.EPSILON
MAX_LAYER_INDEX = config.ANALYSIS_LAYER # Or model.cfg.n_layers - 1 if model is loaded
PCA_VARIANCE_EPSILON = config.EPSILON * 100 # More relaxed for variance checks, e.g., 1e-7 if EPSILON is 1e-9

# --- Timestamp & ID ---
def get_formatted_timestamp() -> str:
    """Returns timestamp like 22APR25_1610."""
    return datetime.datetime.now().strftime("%d%b%y_%H%M").upper()

def generate_artifact_id() -> str:
    """Generates a unique ID for artifacts."""
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    short_uuid = str(uuid.uuid4())[:4]
    return f"ART-{ts}-{short_uuid}"

# --- JSON Handling ---
def safe_save_json(data: dict, filepath: Path):
    """Saves a dictionary to JSON safely, handling numpy types."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        def default_serializer(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, Path): return str(obj)
            try: return json.JSONEncoder.default(None, obj) 
            except TypeError: return str(obj) 
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=default_serializer)
        return True
    except Exception as e:
        print(f"[Util ERROR] Saving JSON to {filepath}: {e}")
        traceback.print_exc()
        return False

def safe_load_json(filepath: Path) -> dict | None:
    """Loads JSON from a file safely."""
    if not filepath.is_file(): return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Util ERROR] Loading JSON from {filepath}: {e}")
        traceback.print_exc()
        return None

# --- Vector Math ---
def normalise(array: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector.
    Raises ValueError if the norm of the array is less than EPSILON.
    """
    array_float = np.asarray(array, dtype=np.float32)
    if array_float.shape != (DIMENSION,):
        raise ValueError(f"Input array for normalise must have shape ({DIMENSION},), got {array_float.shape}")
        
    norm = np.linalg.norm(array_float)
    if norm < EPSILON:
        raise ValueError(f"Cannot normalise a near-zero vector (norm: {norm} < EPSILON: {EPSILON}).")
    return array_float / norm

def get_difference_vector(vec_A: np.ndarray, vec_B: np.ndarray, normalize_output: bool = False) -> np.ndarray:
    """
    Computes vec_A - vec_B. Optionally normalizes the output.
    Raises ValueError on invalid input.
    """
    if vec_A is None or vec_B is None:
        raise ValueError("Input vectors vec_A and vec_B for get_difference_vector cannot be None.")
    
    vec_A_f = np.asarray(vec_A, dtype=np.float32)
    vec_B_f = np.asarray(vec_B, dtype=np.float32)

    if vec_A_f.shape != (DIMENSION,) or vec_B_f.shape != (DIMENSION,):
        raise ValueError(f"Input vectors for get_difference_vector must have shape ({DIMENSION},). "
                         f"Got {vec_A_f.shape} and {vec_B_f.shape}")
    
    diff = vec_A_f - vec_B_f
    if normalize_output:
        return normalise(diff) # normalise will raise error if diff is near-zero
    return diff

def get_mean_vector(vec_A: np.ndarray, vec_B: np.ndarray, normalize_output: bool = False) -> np.ndarray:
    """
    Computes (vec_A + vec_B) / 2.0. Optionally normalizes the output.
    Raises ValueError on invalid input.
    """
    if vec_A is None or vec_B is None:
        raise ValueError("Input vectors vec_A and vec_B for get_mean_vector cannot be None.")

    vec_A_f = np.asarray(vec_A, dtype=np.float32)
    vec_B_f = np.asarray(vec_B, dtype=np.float32)

    if vec_A_f.shape != (DIMENSION,) or vec_B_f.shape != (DIMENSION,):
        raise ValueError(f"Input vectors for get_mean_vector must have shape ({DIMENSION},). "
                         f"Got {vec_A_f.shape} and {vec_B_f.shape}")
        
    mean_v = (vec_A_f + vec_B_f) / 2.0
    if normalize_output:
        return normalise(mean_v) # normalise will raise error if mean_v is near-zero
    return mean_v

def get_random_orthogonal_vector(u1: np.ndarray, random_seed: int | None = None) -> np.ndarray:
    """
    Generates a random vector and makes it orthogonal to the provided (normalized) u1.
    Raises ValueError on invalid u1, RuntimeError if a non-collinear vector cannot be found.
    """
    if u1 is None:
        raise ValueError("Input u1 for get_random_orthogonal_vector cannot be None.")
    u1_f = np.asarray(u1, dtype=np.float32)
    if u1_f.shape != (DIMENSION,):
        raise ValueError(f"Input u1 for get_random_orthogonal_vector must have shape ({DIMENSION},), got {u1_f.shape}")
    
    norm_u1 = np.linalg.norm(u1_f)
    if not np.isclose(norm_u1, 1.0, atol=EPSILON):
        raise ValueError(f"Input u1 must be normalized (norm: {norm_u1}).")

    if random_seed is not None:
        np.random.seed(random_seed)

    max_retries = 10
    for attempt in range(max_retries):
        random_vec = np.random.randn(DIMENSION).astype(np.float32)
        proj_u1_random_vec = np.dot(random_vec, u1_f) * u1_f
        u2_prime = random_vec - proj_u1_random_vec
        u2_prime_norm = np.linalg.norm(u2_prime)

        if u2_prime_norm >= EPSILON: # Check if it's not a zero vector
            return normalise(u2_prime) # normalise will handle the norm check before division
    
    raise RuntimeError(f"Failed to generate non-collinear random orthogonal vector to u1 after {max_retries} retries.")


def create_u2_orthogonal_to_u1_from_candidate(u1: np.ndarray, vec_for_u2_candidate: np.ndarray, normalize_output: bool = True) -> np.ndarray:
    """
    Given a normalized u1, creates u2 orthogonal to u1, based on vec_for_u2_candidate.
    Optionally normalizes u2.
    Raises ValueError on invalid inputs or if candidate is collinear with u1.
    """
    if u1 is None or vec_for_u2_candidate is None:
        raise ValueError("Inputs u1 and vec_for_u2_candidate cannot be None.")

    u1_f = np.asarray(u1, dtype=np.float32)
    vec_cand_f = np.asarray(vec_for_u2_candidate, dtype=np.float32)

    if u1_f.shape != (DIMENSION,) or vec_cand_f.shape != (DIMENSION,):
        raise ValueError(f"Inputs u1 and vec_for_u2_candidate must have shape ({DIMENSION},). "
                         f"Got {u1_f.shape} and {vec_cand_f.shape}")

    norm_u1 = np.linalg.norm(u1_f)
    if not np.isclose(norm_u1, 1.0, atol=EPSILON):
        raise ValueError(f"Input u1 must be normalized (norm: {norm_u1}).")

    proj_u1_vec_candidate = np.dot(vec_cand_f, u1_f) * u1_f
    u2_prime = vec_cand_f - proj_u1_vec_candidate
    u2_prime_norm = np.linalg.norm(u2_prime)

    if u2_prime_norm < EPSILON:
        raise ValueError("Candidate vector for u2 is effectively collinear with u1, resulting in a near-zero orthogonal component.")
    
    if normalize_output:
        return normalise(u2_prime)
    return u2_prime


def perform_pca_on_activations(list_of_activation_vectors: list[np.ndarray], 
                               n_components: int = 2, 
                               random_state_pca: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Performs PCA on a list of activation vectors.
    Returns (u1, u2) where u1=PC1, u2=PC2 (if n_components >= 2).
    Raises ValueError on invalid input, RuntimeError on PCA failure.
    Issues warnings for low variance components.
    """
    if not list_of_activation_vectors or len(list_of_activation_vectors) < n_components:
        raise ValueError(f"Not enough activation vectors for PCA (need at least {n_components}, got {len(list_of_activation_vectors)}).")

    try:
        data_matrix = np.array(list_of_activation_vectors, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Could not convert list_of_activation_vectors to NumPy array: {e}")

    if data_matrix.ndim != 2 or data_matrix.shape[1] != DIMENSION:
        raise ValueError(f"PCA data matrix must be 2D with shape (n_samples, {DIMENSION}). Got {data_matrix.shape}")

    try:
        pca = PCA(n_components=n_components, random_state=random_state_pca) # Relies on default centering
        pca.fit(data_matrix)
        
        explained_variance = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_
        print(f"[Util PCA] Explained variance ratio for {n_components} components: {explained_variance_ratio}")

        if n_components >= 1 and np.isclose(explained_variance[0], 0.0, atol=PCA_VARIANCE_EPSILON):
            warnings.warn(f"PCA first component (PC1) has near-zero variance ({explained_variance[0]:.2e}). "
                          "Input vectors may lack variation or be problematic.", UserWarning)
        
        # --- MODIFIED SECTION ---
        if n_components >= 2 and len(explained_variance) >= 2 and \
           np.isclose(explained_variance[1], 0.0, atol=PCA_VARIANCE_EPSILON): # Check length!
        # --- END MODIFIED SECTION ---
            warnings.warn(f"PCA second component (PC2) has near-zero variance ({explained_variance[1]:.2e}). "
                          "Input vectors might be rank-deficient or highly collinear.", UserWarning)

        pc1_vec = pca.components_[0].astype(np.float32)
        u1 = normalise(pc1_vec) # PCA components are unit length, but normalise is robust

        u2 = None
        if n_components >= 2:
            if len(pca.components_) >= 2: 
                pc2_vec = pca.components_[1].astype(np.float32)
                u2 = normalise(pc2_vec)
            # else: u2 remains None, which is valid if only 1 component was found/requested effectively
        
        return u1, u2
    except Exception as e:
        print(f"[Util ERROR] PCA computation failed: {e}")
        traceback.print_exc()
        raise RuntimeError(f"PCA computation failed: {e}")


def create_orthonormal_basis(vec1: np.ndarray, vec2: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Creates an orthonormal basis (u1, u2) from two vectors using Gram-Schmidt.
    u1 is normalize(vec1). u2 is derived from vec2 and made orthogonal to u1.
    Primarily used for the 'MEAN_A_VS_B_GRAM_SCHMIDT' method.
    Raises ValueError on issues like non-normalizable vectors or collinearity.
    Returns (None, None) on fatal error to maintain compatibility with older call sites if any,
    but ideally should consistently raise. For now, it relies on normalise and
    create_u2_orthogonal_to_u1_from_candidate to raise errors.
    """
    if vec1 is None or vec2 is None:
        # Keep print for now, but raise ValueError if this function is made stricter
        print("[Util ERROR] Input vectors None for basis creation in create_orthonormal_basis.")
        return None, None 
        # raise ValueError("Input vectors for create_orthonormal_basis cannot be None.")

    vec1_f = np.asarray(vec1, dtype=np.float32)
    vec2_f = np.asarray(vec2, dtype=np.float32)

    if vec1_f.shape != (DIMENSION,) or vec2_f.shape != (DIMENSION,):
        print(f"[Util ERROR] Invalid shapes {vec1_f.shape}, {vec2_f.shape} for create_orthonormal_basis. Expected ({DIMENSION},).")
        return None, None
        # raise ValueError(f"Input vectors for create_orthonormal_basis must have shape ({DIMENSION},).")
    
    try:
        u1 = normalise(vec1_f)
        u2 = create_u2_orthogonal_to_u1_from_candidate(u1, vec2_f, normalize_output=True)
        if u2 is None: # Should not happen if create_u2 raises error for collinearity
             print("[Util WARN] create_orthonormal_basis: u2 became None, likely due to collinearity handled by create_u2_orthogonal_to_u1_from_candidate returning None (should raise error).")
             return u1, None # Or None, None
        return u1, u2
    except ValueError as ve:
        print(f"[Util ERROR] in create_orthonormal_basis: {ve}")
        # traceback.print_exc() # Potentially too verbose for common failures like near-zero vecs
        return None, None # Propagate failure as None tuple for now

# --- Collapse Detection ---
# ... (is_output_collapsed function remains unchanged from your provided code) ...
def is_output_collapsed(
    text: str, min_len_threshold: int = 5, rep_char_threshold: float = 0.9,
    min_rep_pattern_len: int = 2, max_rep_pattern_len: int = 7, rep_pattern_threshold: float = 0.7
    ) -> bool:
    if not text or not isinstance(text, str): return True 
    text = text.strip(); n = len(text)
    if n < min_len_threshold: return True 
    
    if n > 0: 
        char_counts = Counter(text);
        if not char_counts: return True 
        most_common_char_count = char_counts.most_common(1)[0][1]
        if most_common_char_count / n >= rep_char_threshold: return True
        
    if n > min_rep_pattern_len * 2: 
        for pattern_len in range(min_rep_pattern_len, max_rep_pattern_len + 1):
            if pattern_len * 2 > n: continue 
            substrings = [text[i:i+pattern_len] for i in range(n - pattern_len + 1)]
            if not substrings: continue
            substring_counts = Counter(substrings);
            if not substring_counts: continue 
            most_common_substring, most_common_count = substring_counts.most_common(1)[0]
            if most_common_count > 1 and (most_common_count * pattern_len / n >= rep_pattern_threshold): 
                return True
    return False

# --- Dynamic Intervention Parser ---
# ... (parse_dynamic_interventions function remains unchanged) ...
def parse_dynamic_interventions(intervention_str: str | None) -> tuple[dict, list[str]]:
    rules = defaultdict(list); errors = []
    if not intervention_str: return {}, []
    
    parts = intervention_str.split(); valid_modes = ['neg_floor', 'pos_ceil', 'override'] 

    for i, part in enumerate(parts):
        part = part.strip();
        if not part: continue
        try:
            components = part.split(':');
            if len(components) != 5: raise ValueError(f"Expected 5 components, got {len(components)}")
            layer_s, neuron_s, mode, thresh_s, target_s = components
            
            layer = int(layer_s); neuron = int(neuron_s); mode = mode.lower()
            threshold = float(thresh_s) 
            target = float(target_s)

            # Using MAX_LAYER_INDEX from config
            if not (0 <= layer <= MAX_LAYER_INDEX): raise ValueError(f"Layer {layer} out of range (0-{MAX_LAYER_INDEX})")
            if not (0 <= neuron < DIMENSION): raise ValueError(f"Neuron {neuron} out of range (0-{DIMENSION-1})")
            if mode not in valid_modes: raise ValueError(f"Invalid mode '{mode}'. Use 'neg_floor', 'pos_ceil', or 'override'.")
            
            rules[layer].append({"neuron": neuron, "mode": mode, "threshold": threshold, "target": target})
        except ValueError as ve: err_msg = f"Rule '{part}': Invalid format/value ({ve})"; errors.append(err_msg)
        except Exception as e: err_msg = f"Rule '{part}': Unexpected error ({e})"; errors.append(err_msg)
    
    if not rules and not errors and intervention_str.strip(): errors.append("Input provided but no valid rules parsed.")
    return dict(rules), errors