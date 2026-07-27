"""
Central configuration. Edit these paths/settings for your environment
before running run_pipeline.py.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# --- Data paths (edit these once you have access to the datasets) ---
OPENI_REPORTS_DIR = PROJECT_ROOT / "data" / "openi" / "reports"
OPENI_IMAGES_DIR = PROJECT_ROOT / "data" / "openi" / "images"

# CheXpert: train.csv's first 1000 rows now have real downloaded images
# (scripts/download_chexpert_subset.py --n-studies 1000, completed), matching
# N_STUDIES_ABLATION below at the paper's actual scale. valid.csv (234 real
# studies, all from the small "batch 1" blob) is still there as a smaller/
# faster fallback if you ever want to sanity-check something quickly.
CHEXPERT_CSV_PATH = PROJECT_ROOT / "data" / "chexpert" / "train.csv"
CHEXPERT_IMAGES_ROOT = PROJECT_ROOT / "data" / "chexpert"

# Which single pathology column the CheXpert ablation/AUC target and
# fine-tuning label are computed from. Must NOT be inferred from the pseudo-
# report text (dataset.py excludes exactly this column when building that
# text) -- using the "No Finding" column as a global abnormal/normal label
# instead is tempting but leaks badly, since a positive "No Finding" flag
# deterministically implies all 13 other columns are negative, which is most
# of what the pseudo-report text is built from. See dataset.py's
# _build_chexpert_pseudo_report docstring.
CHEXPERT_TARGET_FINDING = "Cardiomegaly"

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
