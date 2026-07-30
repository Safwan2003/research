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

DEFAULT_CHEXPERT_TARGET_FINDING = "Cardiomegaly"


def _compute_chexpert_label(row: dict, target_finding: str, uncertain_policy: str = "u_zeros") -> int:
    """
    Binary label for the `target_finding` pathology column (1=positive/present,
    0=negative/absent), used as the AUC target for the Table-1-style ablation.
    """
    val = row.get(target_finding)
    if val == 1.0 or val == 1:
        return 1
    if val == -1.0 or val == -1:
        return 1 if uncertain_policy == "u_ones" else 0
    return 0  # blank/NaN/0.0 -> negative


def _build_chexpert_pseudo_report(row: dict, target_finding: str, uncertain_policy: str = "u_zeros") -> str:
    """
    CheXpert ships only structured 0/1/-1/blank finding labels, not free-text
    reports -- Section 4.1 of the paper describes this as CheXpert's content
    being "shorter and more label-centric" than OpenI's free-text reports.
    There is nothing for F_voc / text-embedding extraction to operate on
    unless we synthesize a short label-derived pseudo-report, so that's what
    this does.

    IMPORTANT: `target_finding` (the exact column `_compute_chexpert_label`
    reads) is deliberately excluded here, and there is NO blanket "No
    Finding" shortcut either. An earlier version of this function echoed
    EVERY finding column -- including the one being predicted -- into this
    text (and short-circuited on "No Finding", which is just as informative
    about the target since it implies every pathology column is negative),
    so the "text" feature used by the ablation study's logistic regression
    was trivially reconstructing its own label (Text-only AUC came out ~1.0
    on every real run, and CheXpert's expected radiomics-beats-text
    reversal, Section 5.4, never showed up). Excluding the target column
    means any AUC the text embedding achieves now reflects genuine
    comorbidity signal in the OTHER finding columns, not a shortcut.

    Args:
        row: one row of the CheXpert CSV as a dict (column name -> value).
        target_finding: the column NOT to mention (must match the label's
            target_finding, see load_chexpert_dataset / _compute_chexpert_label).
        uncertain_policy: how to render a -1 ("uncertain") label --
            "u_zeros" (default) renders it as "<finding> uncertain."; "u_ones"
            renders it as "<finding> present." (the two standard conventions
            for handling CheXpert's uncertain label in the literature).
    """
    parts = []
    for col in _CHEXPERT_FINDING_COLUMNS:
        if col == target_finding:
            continue
        val = row.get(col)
        if val == 1.0 or val == 1:
            parts.append(f"{col} present.")
        elif val == -1.0 or val == -1:
            if uncertain_policy == "u_ones":
                parts.append(f"{col} present.")
            else:
                parts.append(f"{col} uncertain.")
        # blank/NaN/0.0 -> not mentioned in this study, matches CheXpert's own
        # sparse-label convention (absence of a positive mention, not a
        # confirmed negative).

    return " ".join(parts) if parts else "No other findings noted."


def load_chexpert_dataset(
    csv_path: str, images_root: str, limit: Optional[int] = None,
    uncertain_policy: str = "u_zeros",
    target_finding: str = DEFAULT_CHEXPERT_TARGET_FINDING,
    offset: int = 0,
) -> list:
    """
    Load CheXpert studies from a CSV file (e.g., train.csv / valid.csv).

    CheXpert's CSV has one row per image with columns like "Path", "No
    Finding", and 13 other disease-finding columns as 0/1/-1/blank.

    label: binary presence/absence of `target_finding` (default
    "Cardiomegaly", one of CheXpert's standard single-pathology benchmark
    targets), from `_compute_chexpert_label`. This is what feeds the Table
    1-style ablation's logistic regression. Deliberately NOT the global
    "No Finding" abnormal/normal flag used in an earlier version -- see
    _compute_chexpert_label's docstring for why that leaks against
    report_text below.

    report_text: prefers a real free-text "Report"/"findings" column if the
    CSV actually has one (some CheXpert exports do); otherwise falls back to
    a label-derived pseudo-report (see _build_chexpert_pseudo_report) so
    F_voc and text-embedding extraction have real input instead of an empty
    string. `target_finding` itself is always excluded from the pseudo-report
    fallback -- see that function's docstring for why (avoids the text
    feature trivially leaking the label). The real-report case doesn't have
    this concern: real radiology text legitimately correlating with the
    diagnosis is genuine signal, not an engineered shortcut.

    offset: skip this many rows before taking `limit`. Used to carve out a
    slice that's disjoint from whatever rows src/vlm/finetune_qwen.py trained
    on (which always starts at row 0) -- evaluating a fine-tuned model on
    studies it was also trained on inflates every downstream quality metric.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.head(limit)

    images_root_path = Path(images_root)
    studies = []
    images_root_path = Path(images_root)

    for i, row in df.iterrows():
        row_dict = row.to_dict()
        label = _compute_chexpert_label(row_dict, target_finding, uncertain_policy)

        # Prefer a real free-text report column if the CSV actually has one
        # (some CheXpert exports do) -- only fall back to the synthetic,
        # leakage-free pseudo-report when there's no real text to use.
        if "Report" in row and pd.notna(row["Report"]) and str(row["Report"]).strip():
            report_text = str(row["Report"]).strip()
        elif "findings" in row and pd.notna(row["findings"]) and str(row["findings"]).strip():
            report_text = str(row["findings"]).strip()
        else:
            report_text = _build_chexpert_pseudo_report(row_dict, target_finding, uncertain_policy)

        # Path resolution: try the CSV's path as-is, then with a leading
        # path component stripped (handles both "CheXpert-v1.0/train/..."
        # and images_root already pointing at the "CheXpert-v1.0" dir).
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
                "raw_row": row_dict, "uncertain_policy": uncertain_policy,
                "target_finding": target_finding, "dataset": "chexpert",
            },
        )
        studies.append(study)

    return studies



def _self_test_chexpert_no_leakage():
    """
    Synthetic self-test (no CSV/GPU needed): confirms the pseudo-report text
    never mentions `target_finding`'s own value, so a text embedding of it
    cannot trivially reconstruct the label it's meant to predict.
    """
    positive_row = {
        "No Finding": 0, "Cardiomegaly": 1, "Edema": 1, "Pneumonia": 0,
        "Atelectasis": -1, "Pleural Effusion": 0,
    }
    negative_row = {
        "No Finding": 0, "Cardiomegaly": 0, "Edema": 1, "Pneumonia": 0,
        "Atelectasis": -1, "Pleural Effusion": 0,
    }

    pos_label = _compute_chexpert_label(positive_row, "Cardiomegaly")
    neg_label = _compute_chexpert_label(negative_row, "Cardiomegaly")
    assert pos_label == 1 and neg_label == 0, "label computation broken"

    pos_text = _build_chexpert_pseudo_report(positive_row, "Cardiomegaly")
    neg_text = _build_chexpert_pseudo_report(negative_row, "Cardiomegaly")
    assert "Cardiomegaly" not in pos_text, f"leak: {pos_text!r} mentions the target finding"
    assert "Cardiomegaly" not in neg_text, f"leak: {neg_text!r} mentions the target finding"
    assert "Edema present." in pos_text and "Edema present." in neg_text
    assert "Atelectasis uncertain." in pos_text  # default u_zeros policy

    print("OK: pseudo-report text never mentions the target_finding column.")
    print(f"  positive-label row's text: {pos_text!r}")
    print(f"  negative-label row's text: {neg_text!r}")


if __name__ == "__main__":
    _self_test_chexpert_no_leakage()

    print(
        "\nThis module is a skeleton -- point load_openi_dataset()/load_chexpert_dataset() "
        "at your actual data directories once your professor gives you access.\n"
        "Example:\n\n"
        "    studies = load_openi_dataset(\n"
        "        reports_dir='data/openi/reports',\n"
        "        images_dir='data/openi/images',\n"
        "        limit=10,  # start small while debugging\n"
        "    )\n"
        "    print(f'Loaded {len(studies)} studies')\n"
        "    print(studies[0])\n"
    )
