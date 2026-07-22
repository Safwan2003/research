"""
Dataset loading utilities for OpenI (Indiana University Chest X-ray Collection)
and CheXpert, matching Section 4.1 of the paper.

This is a SKELETON -- the exact loading code depends on how your professor's
lab has the data organized (raw XML reports vs a pre-processed CSV, image
folder layout, etc). Fill in `_parse_openi_xml` / `_load_chexpert_csv` to
match whatever you actually have access to. The important contract is the
`Study` dataclass below: everything downstream (feature extraction, prompts,
evaluation) just needs a list of Study objects.

Where to get the data:
  - OpenI: https://openi.nlm.nih.gov/faq (public, has paired frontal/lateral
    X-rays + free-text radiology reports in XML)
  - CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/ or
    https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
    (requires a data use agreement / registration -- ask your professor if
    the lab already has a copy on a shared drive/server)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


@dataclass
class Study:
    """One medical study: matches Eq. (1), S = {I1, I2, R}."""
    study_id: str
    frontal_image_path: Optional[str] = None
    lateral_image_path: Optional[str] = None
    report_text: str = ""
    label: Optional[int] = None  # e.g. 0=normal, 1=abnormal, for AUC evaluation
    metadata: dict = field(default_factory=dict)


def _parse_openi_xml(xml_path: Path) -> Study:
    """
    Parse a single OpenI XML report file into a Study object.

    OpenI XML reports contain <AbstractText Label="..."> sections (FINDINGS, IMPRESSION)
    and <parentImage id="..."> elements. Also extracts binary ground-truth label:
      0 = normal (Normal tag present / no major abnormalities)
      1 = abnormal (abnormality tags or findings present)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    findings, impression = "", ""
    for abstract in root.iter("AbstractText"):
        label = (abstract.get("Label") or abstract.get("label") or "").upper()
        text = (abstract.text or "").strip()
        if "FINDING" in label:
            findings = text
        elif "IMPRESSION" in label:
            impression = text

    report_text = f"{findings} {impression}".strip()

    # Image IDs
    image_ids = []
    for img in root.iter("parentImage"):
        img_id = img.get("id")
        if img_id:
            image_ids.append(img_id)

    # Determine ground-truth label (0 = Normal, 1 = Abnormal)
    major_tags = [elem.text.lower() for elem in root.iter() if elem.tag in ("major", "MeSH", "term") and elem.text]
    all_text = (report_text + " " + " ".join(major_tags)).lower()

    if "normal" in major_tags or "no acute cardiopulmonary process" in all_text or "unremarkable" in all_text:
        study_label = 0
    elif any(term in all_text for term in ["cardiomegaly", "effusion", "pneumothorax", "opacity", "consolidation", "atelectasis", "edema", "granuloma", "infiltrate", "calcinosis"]):
        study_label = 1
    else:
        study_label = 0 if "normal" in all_text else 1

    return Study(
        study_id=xml_path.stem,
        report_text=report_text,
        label=study_label,
        metadata={"image_ids": image_ids, "major_tags": major_tags},
    )


def load_openi_dataset(reports_dir: str, images_dir: str, limit: Optional[int] = None) -> list:
    """
    Load OpenI studies from a directory of XML report files + a directory of
    associated PNG images.
    """
    reports_dir = Path(reports_dir)
    images_dir = Path(images_dir)

    studies = []
    xml_files = sorted(reports_dir.glob("*.xml"))
    if limit:
        xml_files = xml_files[:limit]

    for xml_path in xml_files:
        study = _parse_openi_xml(xml_path)

        image_ids = study.metadata.get("image_ids", [])
        # Resolve image paths if files exist
        valid_img_paths = []
        for img_id in image_ids:
            # Check with or without .png extension
            cand = images_dir / f"{img_id}.png" if not img_id.endswith(".png") else images_dir / img_id
            if cand.exists():
                valid_img_paths.append(str(cand))

        if valid_img_paths:
            study.frontal_image_path = valid_img_paths[0]
            if len(valid_img_paths) > 1:
                study.lateral_image_path = valid_img_paths[1]

        studies.append(study)

    return studies


_CHEXPERT_FINDING_COLUMNS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]


def _build_chexpert_pseudo_report(row: dict, uncertain_policy: str = "u_zeros") -> str:
    """
    CheXpert ships only structured 0/1/-1/blank finding labels, not free-text
    reports -- Section 4.1 of the paper describes this as CheXpert's content
    being "shorter and more label-centric" than OpenI's free-text reports.
    """
    if row.get("No Finding") == 1 or row.get("No Finding") == 1.0:
        return "No Finding. No acute cardiopulmonary abnormality."

    parts = []
    for col in _CHEXPERT_FINDING_COLUMNS:
        val = row.get(col)
        if val == 1.0 or val == 1:
            parts.append(f"{col}: positive;")
        elif val == -1.0 or val == -1:
            if uncertain_policy == "u-ones" or uncertain_policy == "u_ones":
                parts.append(f"{col}: positive;")
            else:
                parts.append(f"{col}: negative;")
        elif val == 0.0 or val == 0:
            parts.append(f"{col}: negative;")

    return " ".join(parts) if parts else "No Finding."


def load_chexpert_dataset(
    csv_path: str,
    images_root: str,
    limit: Optional[int] = None,
    uncertain_policy: str = "u-zeros",
) -> list:
    """
    Load CheXpert studies from a CSV file (e.g., train.csv / valid.csv).

    Per Section 4.1 & 5.4 of the paper:
      1. Generates a pseudo-report string from finding columns if free-text report is missing
         (e.g., "Cardiomegaly: positive; Edema: negative; ...").
      2. Assigns binary ground truth label (0 = Normal, 1 = Abnormal) based on 'No Finding'
         column and pathology columns under the specified uncertain_policy ('u-zeros' or 'u-ones').
      3. Resolves image file paths dynamically relative to images_root.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    studies = []
    images_root_path = Path(images_root)

    for i, row in df.iterrows():
        row_dict = row.to_dict()
        
        # 1. Report text resolution
        if "Report" in row and pd.notna(row["Report"]) and str(row["Report"]).strip():
            report_text = str(row["Report"]).strip()
        elif "findings" in row and pd.notna(row["findings"]) and str(row["findings"]).strip():
            report_text = str(row["findings"]).strip()
        else:
            report_text = _build_chexpert_pseudo_report(row_dict, uncertain_policy)

        # 2. Binary Ground Truth Label (0 = Normal, 1 = Abnormal)
        no_finding = row_dict.get("No Finding")
        if no_finding == 1.0 or no_finding == 1:
            label = 0
        else:
            has_pathology = False
            for col in _CHEXPERT_FINDING_COLUMNS:
                val = row_dict.get(col)
                if val == 1.0 or val == 1 or (val == -1 and (uncertain_policy == "u-ones" or uncertain_policy == "u_ones")):
                    has_pathology = True
                    break
            label = 1 if has_pathology else 0

        # 3. Path resolution
        rel_path_str = str(row["Path"]).lstrip("/")
        cand_path = images_root_path / rel_path_str
        if not cand_path.exists():
            parts = Path(rel_path_str).parts
            if len(parts) > 1:
                alt_path = images_root_path.joinpath(*parts[1:])
                if alt_path.exists():
                    cand_path = alt_path

        study = Study(
            study_id=f"chexpert_{i}",
            frontal_image_path=str(cand_path),
            report_text=report_text,
            label=label,
            metadata={
                "raw_row": row_dict,
                "uncertain_policy": uncertain_policy,
                "dataset": "chexpert",
            },
        )
        studies.append(study)

    return studies


if __name__ == "__main__":
    print("CheXpert dataset module ready.")
