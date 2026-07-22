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

    OpenI XML reports typically contain <AbstractText Label="FINDINGS"> and
    <AbstractText Label="IMPRESSION"> sections, plus <parentImage id="..."/>
    references to the associated image files. Adjust the tag names below if
    your copy of OpenI is structured differently (versions vary).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    findings, impression = "", ""
    for abstract in root.iter("AbstractText"):
        label = abstract.get("Label", "").upper()
        text = (abstract.text or "").strip()
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text

    report_text = f"{findings} {impression}".strip()

    image_ids = [img.get("id") for img in root.iter("parentImage")]

    return Study(
        study_id=xml_path.stem,
        report_text=report_text,
        metadata={"image_ids": image_ids},
    )


def load_openi_dataset(reports_dir: str, images_dir: str, limit: Optional[int] = None) -> list:
    """
    Load OpenI studies from a directory of XML report files + a directory of
    associated PNG/JPG images.

    Args:
        reports_dir: path to folder containing per-study .xml report files.
        images_dir: path to folder containing the corresponding image files.
        limit: optionally cap the number of studies loaded (useful while
               developing/debugging, before running on the full 1,000-study set).

    Returns:
        list of Study objects.
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
        if len(image_ids) >= 1:
            study.frontal_image_path = str(images_dir / f"{image_ids[0]}.png")
        if len(image_ids) >= 2:
            study.lateral_image_path = str(images_dir / f"{image_ids[1]}.png")

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
    There is nothing for F_voc / text-embedding extraction to operate on
    unless we synthesize a short label-derived pseudo-report, so that's what
    this does.

    Args:
        row: one row of the CheXpert CSV as a dict (column name -> value).
        uncertain_policy: how to render a -1 ("uncertain") label --
            "u_zeros" (default) renders it as "<finding> uncertain."; "u_ones"
            renders it as "<finding> present." (the two standard conventions
            for handling CheXpert's uncertain label in the literature).
    """
    if row.get("No Finding") == 1:
        return "No Finding. No acute cardiopulmonary abnormality."

    parts = []
    for col in _CHEXPERT_FINDING_COLUMNS:
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

    return " ".join(parts) if parts else "No Finding."


def load_chexpert_dataset(
    csv_path: str, images_root: str, limit: Optional[int] = None,
    uncertain_policy: str = "u_zeros",
) -> list:
    """
    Load CheXpert studies from the standard train.csv/valid.csv layout.

    CheXpert's CSV has one row per image with columns like "Path", "No
    Finding", and 13 other disease-finding columns as 0/1/-1/blank.

    label: derived as the binary "abnormal" target used elsewhere in this
    project (0=normal, 1=abnormal), from the "No Finding" column --
    label = 0 if "No Finding" == 1, else 1. This is what feeds the Table
    1-style ablation's logistic regression.

    report_text: since CheXpert has no free-text reports, this is a
    label-derived pseudo-report (see _build_chexpert_pseudo_report) so F_voc
    and text-embedding extraction have real input instead of an empty string.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    studies = []
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        no_finding = row_dict.get("No Finding")
        label = 0 if (no_finding == 1 or no_finding == 1.0) else 1

        study = Study(
            study_id=str(i),
            frontal_image_path=str(Path(images_root) / row["Path"]),
            report_text=_build_chexpert_pseudo_report(row_dict, uncertain_policy),
            label=label,
            metadata={"raw_row": row_dict, "uncertain_policy": uncertain_policy},
        )
        studies.append(study)

    return studies


if __name__ == "__main__":
    print(
        "This module is a skeleton -- point load_openi_dataset()/load_chexpert_dataset() "
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
