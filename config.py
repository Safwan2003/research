"""
Central configuration. Edit these paths/settings for your environment
before running run_pipeline.py.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# --- Data paths (edit these once you have access to the datasets) ---
OPENI_REPORTS_DIR = PROJECT_ROOT / "data" / "openi" / "reports"
OPENI_IMAGES_DIR = PROJECT_ROOT / "data" / "openi" / "images"
CHEXPERT_CSV_PATH = PROJECT_ROOT / "data" / "chexpert" / "train.csv"
CHEXPERT_IMAGES_ROOT = PROJECT_ROOT / "data" / "chexpert"

# --- Vocabulary ---
VOCABULARY_PATH = PROJECT_ROOT / "vocabulary_list.txt"

# --- Results persistence (append-only experiment log, never overwritten) ---
RESULTS_LOG_PATH = PROJECT_ROOT / "results" / "experiment_log.json"

# --- VLM settings (Section 4.3) ---
VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # paper's default backbone
VLM_DEVICE = "cuda"  # use "cpu" if no GPU, but expect it to be very slow

# --- Experiment sizes (Section 4) ---
N_STUDIES_ABLATION = 1000  # paper uses 1,000 studies for the AUC ablation
N_STUDIES_AGENTIC = 50     # paper uses 50 studies for stepwise agentic analysis

RANDOM_SEED = 42
