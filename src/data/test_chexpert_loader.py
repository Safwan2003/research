import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import load_chexpert_dataset

csv_path = "data/chexpert/CheXpert-v1.0 batch 1 (validate & csv)/valid.csv"
images_root = "data/chexpert/CheXpert-v1.0 batch 1 (validate & csv)"

print(f"Loading CheXpert validation dataset from {csv_path}...")
studies = load_chexpert_dataset(csv_path, images_root, limit=5)

print(f"Successfully loaded {len(studies)} validation studies:\n")
for s in studies:
    print(f"Study ID: {s.study_id}")
    print(f"  Ground-Truth Label (0=normal, 1=abnormal): {s.label}")
    print(f"  Generated Pseudo-Report: {s.report_text}")
    print(f"  Frontal Image Path: {s.frontal_image_path}")
    print(f"  File Exists: {Path(s.frontal_image_path).exists()}\n")
