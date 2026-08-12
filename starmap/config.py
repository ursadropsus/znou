# config.py
from pathlib import Path
import torch

# --- Core Settings ---
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LAYERS = [0, 5, 11] # Layers for activation capture during general experiments
ANALYSIS_LAYER = 11 # Default layer for basis projection and specific analysis
DIMENSION = 3072 # Default for GPT-2's MLP layer, will be updated if model changes
EPSILON = 1e-9

# --- File Paths ---
ARTIFACTS_DIR = Path("./game_data/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_BASIS_SEARCH_DIR = Path("./game_data/vectorbases")
DEFAULT_BASIS_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_OUTPUT_ROOT_DIR = Path("./analysis_results")
# Companion metadata for bases will be stored in DEFAULT_BASIS_SEARCH_DIR / "basesdata"

# --- Script Versioning for Inter-script Compatibility ---
# This suffix should match the output of analyze_artifacts.py that you want evaluate_basis_quality.py (etc.) to process.
# Example: "v2.5.1_clusteredHR"
# ** IMPORTANT: UPDATE THIS VALUE to match the suffix of your latest analyze_artifacts.py output files **
SCRIPT_VERSION_SUFFIX_FOR_ANALYSIS_FILES = "v2.5.1_clusteredHR"

# --- Game Settings (UI & Experiment Runner) ---
MAX_OUTPUT_LINES = 100 # For the UI output panel
GENERATION_LENGTH = 30 # Default number of new tokens to generate

# --- Default Generation Parameters (for experiment_runner & UI) ---
DEFAULT_DO_SAMPLE = False # Start with Greedy by default for reproducibility
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
# DEFAULT_TOP_P = 1.0 # Can add top_p later if desired

# --- Basis Generation Settings ---
class BasisGenerationMethods:
    """String constants for different basis generation methods."""
    # Original method: u1 from vec_A, u2 from vec_B via Gram-Schmidt
    MEAN_A_VS_B_GRAM_SCHMIDT = "mean_A_vs_B_gram_schmidt"
    
    # u1 from (vec_A - vec_B), u2 is a random vector made orthogonal
    DIFF_AB_RAND_ORTHO_U2 = "diff_AB_rand_ortho_u2"
    
    # u1 from (vec_A - vec_B), u2 from (vec_A + vec_B) made orthogonal
    DIFF_AB_MEAN_AB_ORTHO_U2 = "diff_AB_mean_AB_ortho_u2"
    
    # u1=PC1, u2=PC2 from PCA on a cluster of A and B type prompt activations
    PCA_A_B_CLUSTER = "pca_A_B_cluster"
    
    # Example for a future method, e.g., if you implement centering
    # MEAN_A_CENTERED_VS_B_CENTERED_GRAM_SCHMIDT = "mean_A_centered_vs_B_centered_gram_schmidt"

# Default number of sample prompts per pole when using PCA_A_B_CLUSTER method
# This is a guideline; scripts/UI might allow overriding or adapt if fewer are provided.
PCA_NUM_SAMPLES_PER_POLE = 10
# --- END Basis Generation Settings ---

# Note: DIMENSION might be dynamically updated by experiment_runner.py
# after the model is loaded if a different model is used.
# However, for scripts that might run *before* the model is loaded by the runner
# (like some basis creation scripts if they don't init the full runner),
# this default is important.