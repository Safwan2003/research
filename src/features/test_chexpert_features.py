import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vlm"))

from radiomics import extract_radiomics
from xai_gradcam import extract_xai_features
from vocabulary import extract_vocabulary_features
from feature_card import build_feature_card, feature_card_to_prompt_text
from src.data.dataset import load_chexpert_dataset

csv_path = "data/chexpert/CheXpert-v1.0 batch 1 (validate & csv)/valid.csv"
images_root = "data/chexpert/CheXpert-v1.0 batch 1 (validate & csv)"

studies = load_chexpert_dataset(csv_path, images_root, limit=1)
study = studies[0]

print(f"--- Processing CheXpert Study {study.study_id} ---")
print(f"Image path: {study.frontal_image_path}")

img = np.array(Image.open(study.frontal_image_path).convert("L"))

print("\n1. Extracting F_rad (radiomics)...")
rad = extract_radiomics(img)
print("  -> F_rad:", rad)

print("\n2. Extracting F_xai (Grad-CAM spatial statistics)...")
xai = extract_xai_features(img)
print("  -> F_xai:", xai)

print("\n3. Extracting F_voc (vocabulary matching)...")
voc = extract_vocabulary_features(study.report_text)
print("  -> F_voc:", voc)

print("\n4. Building Feature Card (F_tool = Serialize(F_rad, F_xai, F_voc))...")
card = build_feature_card(rad, xai, voc)
print(feature_card_to_prompt_text(card))
