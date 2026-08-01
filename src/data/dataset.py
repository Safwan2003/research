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

import re
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

_PATIENT_ID_RE = re.compile(r"(patient\d+)")


def extract_patient_id(path_str: str) -> str:
    """
    Pull 'patientXXXXX' out of a CheXpert Path column value, e.g.
    'CheXpert-v1.0/train/patient12345/study2/view1_frontal.jpg' ->
    'patient12345'. Raises on no match -- a silently-None patient id would
    corrupt grouping/stratification without any visible symptom.
    """
    m = _PATIENT_ID_RE.search(str(path_str))
    if not m:
        raise ValueError(f"No patientNNNNN token found in CheXpert Path {path_str!r}")
    return m.group(1)


def _patient_id_column(df):
    """
    Vectorized extract_patient_id() over df['Path']. Rows whose Path doesn't
    parse are dropped (with a warning) rather than raising -- one malformed
    row shouldn't crash sampling over the other ~223k good ones.
    """
    extracted = df["Path"].astype(str).str.extract(_PATIENT_ID_RE)[0]
    n_bad = int(extracted.isna().sum())
    if n_bad:
        print(f"WARNING: {n_bad} row(s) had no parseable patient id in their Path column; "
              f"excluding them from patient-level sampling.")
    return extracted


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


def sample_chexpert_patients(
    csv_path: str,
    seed: int = 42,
    stratify_by: Optional[str] = None,
    uncertain_policy: str = "u_zeros",
) -> list:
    """
    Read csv_path ONCE and return every unique patient id it contains, in a
    single deterministic shuffled order. Two calls with identical
    (csv_path, seed, stratify_by, uncertain_policy) always return the
    identical list -- that determinism (not shared in-memory state) is what
    lets two separate process invocations (e.g. src/vlm/finetune_qwen.py
    and run_chexpert_eval.py) agree on the same canonical order and
    therefore compute disjoint prefix/suffix slices of it without talking
    to each other.

    This exists because train.csv has multiple rows (images) per patient --
    naively sampling/slicing ROWS (what the old "first N rows" code did,
    and what a row-level random sample would still do) can put the same
    patient's different images on both sides of a train/held-out split,
    leaking patient identity across it. Sampling PATIENTS instead and then
    taking every row for each selected patient avoids that by construction.

    If `stratify_by` is a finding column name (e.g. "Cardiomegaly"),
    patients are bucketed into "ever positive" / "never positive" by taking
    the max of _compute_chexpert_label over each patient's rows (one
    positive study makes the whole patient positive -- the natural
    patient-level analogue of the per-row binary label), each bucket is
    shuffled independently, then interleaved proportionally so that ANY
    prefix of the returned list -- not just the full list -- has
    approximately the population's positive:negative ratio. This is what
    makes a small sample's label prevalence track the full dataset's,
    instead of the "first N rows" bug where a narrow, non-random slice
    could have wildly different prevalence than the population.

    If `stratify_by` is None, patients are plain-shuffled (uniform
    patient-level random sampling, no stratification).
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)
    patient_col = _patient_id_column(df)
    df = df[patient_col.notna()].copy()
    df["_patient_id"] = patient_col[patient_col.notna()]

    rng = np.random.default_rng(seed)

    if stratify_by:
        if stratify_by not in df.columns:
            print(f"WARNING: stratify_by={stratify_by!r} not found in {csv_path} columns; "
                  f"every patient will be treated as negative for stratification purposes.")
        row_labels = df.apply(
            lambda r: _compute_chexpert_label(r.to_dict(), stratify_by, uncertain_policy), axis=1,
        )
        patient_label = row_labels.groupby(df["_patient_id"]).max()
        pos = patient_label[patient_label == 1].index.to_numpy()
        neg = patient_label[patient_label == 0].index.to_numpy()
        rng.shuffle(pos)
        rng.shuffle(neg)

        total = len(pos) + len(neg)
        pos_fraction = len(pos) / total if total else 0.0
        merged, pi, ni = [], 0, 0
        for i in range(total):
            target_pos_count = round((i + 1) * pos_fraction)
            if pi < target_pos_count and pi < len(pos):
                merged.append(pos[pi]); pi += 1
            elif ni < len(neg):
                merged.append(neg[ni]); ni += 1
            else:
                merged.append(pos[pi]); pi += 1
        return merged
    else:
        all_patients = df["_patient_id"].unique()
        rng.shuffle(all_patients)
        return all_patients.tolist()


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


def _build_chexpert_full_report(row: dict, uncertain_policy: str = "u_zeros") -> str:
    """
    Ground-truth report text for SCORING (hallucination rate / ROUGE /
    BERTScore) only -- NOT what gets fed to the model. _build_chexpert_pseudo_
    report (the model's INPUT / the ablation study's text feature) must hide
    target_finding to avoid leaking the label it's predicting. But the text a
    hallucination check compares generated output against is supposed to be
    the actual truth about the study, target finding included -- a real
    radiology report states the diagnosis. Reusing the leakage-free pseudo-
    report as "ground truth" here made every model that correctly and
    confidently names target_finding (e.g. a LoRA adapter deliberately
    trained to do so in finetune_qwen.py's build_target_completion) score as
    having "hallucinated" its own correct answer, since that word could never
    appear in ground truth by construction. See run_chexpert_eval.py's
    run_agentic_on_chexpert, which uses this for `ground_truths` while still
    passing the leakage-free text as the model's input.

    Implemented as _build_chexpert_pseudo_report with target_finding=None so
    no column name can match it and get skipped -- every finding column,
    including the target, is included.
    """
    return _build_chexpert_pseudo_report(row, target_finding=None, uncertain_policy=uncertain_policy)


def _select_patients_by_row_budget(order: list, row_counts: dict, offset: int, limit: Optional[int]) -> list:
    """
    Walk `order` (a list of patient ids), accumulating each patient's row
    count, and select whole patients whose row-span starts at or after
    `offset` and before `offset + limit` (or to the end of `order` if
    `limit` is None). A patient's row-span is considered to "start" at the
    cumulative row count BEFORE that patient is added -- this is what
    guarantees a second call with offset=<first call's offset + limit>
    picks up exactly where the first call left off, with no patient
    double-counted or dropped at the boundary, even though patients
    contribute different numbers of rows (so a plain row-count offset would
    otherwise risk landing mid-patient).
    """
    selected = []
    cum = 0
    stop_at = offset + limit if limit is not None else None
    for pid in order:
        cnt = row_counts.get(pid, 0)
        if cnt == 0:
            continue
        if stop_at is not None and cum >= stop_at:
            break
        if cum >= offset:
            selected.append(pid)
        cum += cnt
    return selected


def load_chexpert_dataset(
    csv_path: str, images_root: str, limit: Optional[int] = None,
    uncertain_policy: str = "u_zeros",
    target_finding: str = DEFAULT_CHEXPERT_TARGET_FINDING,
    offset: int = 0,
    sample_strategy: str = "sequential",
    seed: int = 42,
    stratify: bool = True,
    patient_ids: Optional[list] = None,
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

    sample_strategy: "sequential" (default, unchanged historical behavior)
    takes rows [offset, offset+limit) in CSV order -- train.csv is sorted by
    patient id, so this is a narrow, non-representative, single-patient-ID-
    range slice, and multi-row patients can straddle an offset boundary
    (leaking a patient's images across a train/held-out split). "random"
    fixes both problems: it samples whole PATIENTS (via
    sample_chexpert_patients(), optionally stratified by `target_finding`
    so a small sample's label prevalence still tracks the full dataset's),
    then includes every row for each selected patient. In this mode, `limit`
    is a target ROW count (not a patient count -- CheXpert averages ~3.4
    rows/patient), satisfied by greedily including whole patients in
    canonical shuffled order until the cumulative row count reaches `limit`
    (small overshoot from the last patient included is expected); `offset`
    is in the same row-count terms and must equal a previous call's own
    offset+limit to be exactly disjoint from it (see
    _select_patients_by_row_budget). For an EXACT (not offset-approximate)
    disjoint slice regardless of seed/offset arithmetic, pass `patient_ids`
    instead (see below) -- that's what run_chexpert_eval.py does when a
    src/vlm/finetune_qwen.py training run's train_patient_ids.json sidecar
    is available.

    seed, stratify: only used when sample_strategy="random" and
    `patient_ids` is not given -- forwarded to sample_chexpert_patients()
    (stratify=True stratifies by `target_finding`; False plain-shuffles).

    patient_ids: an explicit, pre-selected/pre-ordered list of patient ids
    to load from (e.g. the full canonical order with a training run's
    patients already excluded) -- takes precedence over the seed/stratify-
    based sample_chexpert_patients() call. Still subject to the same
    `limit`-as-row-budget walk (from the start of this list, i.e. as if
    offset=0); requires sample_strategy="random".

    offset (sequential mode): skip this many rows before taking `limit`.
    """
    import pandas as pd

    if sample_strategy not in ("sequential", "random"):
        raise ValueError(f"sample_strategy must be 'sequential' or 'random', got {sample_strategy!r}")
    if patient_ids is not None and sample_strategy != "random":
        raise ValueError("patient_ids requires sample_strategy='random'")

    df = pd.read_csv(csv_path)

    if sample_strategy == "sequential":
        if offset:
            df = df.iloc[offset:]
        if limit:
            df = df.head(limit)
    else:
        patient_col = _patient_id_column(df)
        df = df[patient_col.notna()].copy()
        df["_patient_id"] = patient_col[patient_col.notna()]
        row_counts = df["_patient_id"].value_counts().to_dict()

        if patient_ids is not None:
            order, walk_offset = list(patient_ids), 0
        else:
            order = sample_chexpert_patients(
                csv_path=csv_path, seed=seed,
                stratify_by=target_finding if stratify else None,
                uncertain_policy=uncertain_policy,
            )
            walk_offset = offset

        selected_patients = set(_select_patients_by_row_budget(order, row_counts, walk_offset, limit))
        df = df[df["_patient_id"].isin(selected_patients)]

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
            ground_truth_report = report_text  # real text, no target_finding scrubbing to undo
        elif "findings" in row and pd.notna(row["findings"]) and str(row["findings"]).strip():
            report_text = str(row["findings"]).strip()
            ground_truth_report = report_text
        else:
            report_text = _build_chexpert_pseudo_report(row_dict, target_finding, uncertain_policy)
            ground_truth_report = _build_chexpert_full_report(row_dict, uncertain_policy)

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
                "patient_id": extract_patient_id(str(row["Path"])),
                "ground_truth_report": ground_truth_report,
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

    # The scoring-only counterpart must do the OPPOSITE: include target_finding,
    # or a model that correctly names it gets scored as hallucinating its own
    # correct answer (see _build_chexpert_full_report's docstring).
    pos_full = _build_chexpert_full_report(positive_row)
    neg_full = _build_chexpert_full_report(negative_row)
    assert "Cardiomegaly present." in pos_full, f"ground truth should state the positive finding: {pos_full!r}"
    assert "Cardiomegaly" not in neg_full, f"ground truth should not assert a finding that's absent: {neg_full!r}"
    assert "Edema present." in pos_full and "Edema present." in neg_full

    print("OK: full report text (used as scoring ground truth, never as model input) does mention target_finding.")
    print(f"  positive-label row's full text: {pos_full!r}")
    print(f"  negative-label row's full text: {neg_full!r}")


def _self_test_patient_level_sampling():
    """
    Synthetic self-test (no real CheXpert data/GPU needed): builds a small
    CSV with multiple rows per patient and confirms
    sample_chexpert_patients() / load_chexpert_dataset(sample_strategy=
    "random") sample whole PATIENTS (not rows) deterministically, and that
    two calls chained via offset=<first call's limit> -- the same pattern
    run_finetune_pipeline.sh uses for its train vs. held-out slices --
    share zero patients, even though patients contribute different numbers
    of rows. Also confirms sample_strategy="sequential" (the default)
    behaves exactly as it always has.
    """
    import csv
    import tempfile
    from collections import Counter

    row_counts_per_patient = [1, 2, 3, 4, 2, 1, 3, 2, 1, 4] * 5  # 50 patients, 115 rows total
    rows = []
    for i, n_images in enumerate(row_counts_per_patient):
        patient_id = f"patient{i:05d}"
        cardiomegaly = 1 if i % 3 == 0 else 0  # deterministic-but-mixed label distribution
        for v in range(n_images):
            rows.append({
                "Path": f"CheXpert-v1.0/train/{patient_id}/study1/view{v + 1}_frontal.jpg",
                "Cardiomegaly": cardiomegaly,
            })

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Path", "Cardiomegaly"])
        writer.writeheader()
        writer.writerows(rows)
        csv_path = f.name

    try:
        order_a = sample_chexpert_patients(csv_path, seed=7, stratify_by="Cardiomegaly")
        order_b = sample_chexpert_patients(csv_path, seed=7, stratify_by="Cardiomegaly")
        assert order_a == order_b, "sample_chexpert_patients isn't deterministic for a fixed seed"
        assert len(order_a) == 50, f"expected 50 unique patients, got {len(order_a)}"

        train_studies = load_chexpert_dataset(
            csv_path, images_root="/nonexistent", limit=100,
            target_finding="Cardiomegaly", sample_strategy="random", seed=7,
        )
        held_out_studies = load_chexpert_dataset(
            csv_path, images_root="/nonexistent", limit=100, offset=100,
            target_finding="Cardiomegaly", sample_strategy="random", seed=7,
        )
        train_patients = {s.metadata["patient_id"] for s in train_studies}
        held_out_patients = {s.metadata["patient_id"] for s in held_out_studies}
        assert train_patients, "train slice selected zero patients"
        assert held_out_patients, "held-out slice selected zero patients"
        overlap = train_patients & held_out_patients
        assert not overlap, f"train/held-out slices share patients: {overlap}"

        sequential_studies = load_chexpert_dataset(csv_path, images_root="/nonexistent", limit=5)
        counts = Counter(s.metadata["patient_id"] for s in sequential_studies)
        assert counts == Counter({"patient00000": 1, "patient00001": 2, "patient00002": 2}), (
            f"sample_strategy='sequential' (the default) changed behavior: got {counts}"
        )
    finally:
        Path(csv_path).unlink(missing_ok=True)

    print(f"OK: sample_chexpert_patients() is deterministic; random-mode train "
          f"({len(train_patients)} patients) and held-out ({len(held_out_patients)} patients) "
          f"slices share zero patients; sequential mode unchanged.")


if __name__ == "__main__":
    _self_test_chexpert_no_leakage()
    _self_test_patient_level_sampling()

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
