"""
analyze_embeddings.py
=====================

FER Representation Reliability Benchmark
-----------------------------------------

Analyzes DINOv2 ViT-B/14 embeddings for the controlled 215-viewpoint dataset.

IMPORTANT:
- This script does NOT use the traditional 6 emotion classes.
- Each complete dataset folder is treated as one expression sequence / identity
  for the controlled experiment.
- A = Representation Drift
- B = Expression Consistency (437-way / complete-sequence prototype retrieval)
- C = Expression Separability (own-prototype margin against the closest rival)

Core methodological idea
-------------------------
We do not claim cosine distance, nearest-prototype classification, or integration
to be mathematically new. The project-specific contribution is the CONTROLLED
VIEWPOINT TRAJECTORY analysis:

    fixed expression
        |
        +--> representation trajectory across 215 viewpoints
        |
        +--> prediction consistency trajectory
        |
        +--> semantic-separability margin trajectory
        |
        +--> critical viewpoint boundaries
        |
        +--> integrated trajectory quantities
        |
        +--> lead/lag analysis:
             does representation instability appear BEFORE prediction failure?

This script is intentionally transparent and reproducible.

Expected input
--------------
data/metadata.csv
data/embeddings/embeddings_00000.npy
data/embeddings/metadata_00000.csv
...

The shard metadata must contain:
folder, gender, expression, viewpoint, angle, image_path

Expected embedding shape:
(number_of_images_in_shard, 768)

The script automatically uses only folders with all 215 viewpoints.

Outputs
-------
analysis/
    per_view_metrics.csv
    per_expression_summary.csv
    population_profile.csv
    directional_summary.csv
    bootstrap_population_ci.csv
    analysis_report.json
    A_representation_drift.png
    B_expression_consistency.png
    C_expression_separability.png
    README_analysis.md

Dependencies
------------
pip install numpy pandas matplotlib

Run from repository root:
    python scripts/analyze_embeddings.py

Optional:
    python scripts/analyze_embeddings.py --no-plots
    python scripts/analyze_embeddings.py --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPECTED_VIEWPOINTS = 215
MIN_VIEWPOINT = 0
MAX_VIEWPOINT = 214
FRONTAL_VIEWPOINT = 107

# Near-frontal window used to estimate "normal" representation / margin
# variability before substantial viewpoint change.
BASELINE_DEG = 5

# A threshold crossing is considered a boundary only if it persists for
# several consecutive viewpoints. This avoids calling a single noisy point
# a failure.
SUSTAINED_FAILURE = 3

# B/C prototype:
# Use a small near-frontal window, excluding the exact frontal image V107.
# This avoids making B/C at V107 trivially correct by comparing an image to
# itself.
PROTOTYPE_WINDOW_DEG = 5

# Quantiles for robust thresholds.
A_DRIFT_QUANTILE = 0.95
C_MARGIN_QUANTILE = 0.05

# Bootstrap repetitions. 500 is a good default; 1000 is preferable for
# final paper figures if runtime is acceptable.
DEFAULT_BOOTSTRAP = 500

EPS = 1e-12


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[0]


def natural_key(path: Path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else -1


def trapz_mean(y: np.ndarray, x: np.ndarray) -> float:
    """Integral divided by total x-range."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if len(y) < 2:
        return float(np.nan)

    width = x[-1] - x[0]
    if width <= 0:
        return float(np.nan)

    if hasattr(np, "trapezoid"):
        area = np.trapezoid(y, x)
    else:
        area = np.trapz(y, x)

    return float(area / width)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    return x / max(float(n), EPS)


def angular_distance_deg_from_cos(cosine: np.ndarray) -> np.ndarray:
    c = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(c)).astype(np.float32)


def sustained_first_index(mask: np.ndarray, run: int = SUSTAINED_FAILURE):
    """
    Return the first index at which `mask` is true for `run` consecutive
    positions. Returns None if no sustained run exists.
    """
    mask = np.asarray(mask, dtype=bool)

    if len(mask) < run:
        return None

    count = 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        if count >= run:
            return i - run + 1

    return None


def signed_direction(angle: float) -> str:
    if angle < 0:
        return "left"
    if angle > 0:
        return "right"
    return "frontal"


# ---------------------------------------------------------------------------
# Metadata / shard loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "folder",
    "gender",
    "expression",
    "viewpoint",
    "angle",
    "image_path",
}


def find_embedding_dir(root: Path) -> Path:
    candidates = [
        root / "data" / "embeddings",
        root / "embeddings",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find embeddings directory. Expected:\n"
        f"  {root / 'data' / 'embeddings'}"
    )


def load_metadata(root: Path) -> pd.DataFrame:
    """
    Prefer the shard metadata because it is guaranteed to be aligned with
    each embedding shard. Fall back to data/metadata.csv if necessary.
    """
    embedding_dir = find_embedding_dir(root)

    shard_meta = sorted(
        embedding_dir.glob("metadata_*.csv"),
        key=natural_key,
    )

    if shard_meta:
        frames = []
        for path in shard_meta:
            df = pd.read_csv(path)

            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise RuntimeError(
                    f"{path} is missing required columns: {sorted(missing)}"
                )

            df["_source_metadata_shard"] = path.name
            frames.append(df)

        metadata = pd.concat(frames, ignore_index=True)

    else:
        path = root / "data" / "metadata.csv"
        if not path.exists():
            raise FileNotFoundError(
                "No shard metadata or data/metadata.csv was found."
            )

        metadata = pd.read_csv(path)

        missing = REQUIRED_COLUMNS - set(metadata.columns)
        if missing:
            raise RuntimeError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

    # ---------------------------------------------------------
    # Normalize viewpoint
    # ---------------------------------------------------------
    # The metadata pipeline may store viewpoint either as:
    #     0, 1, 2, ..., 214
    # or as:
    #     V000, V001, V002, ..., V214
    #
    # Internally we always use integer indices 0..214.
    raw_viewpoint = metadata["viewpoint"].astype(str).str.strip()

    normalized_viewpoint = raw_viewpoint.str.replace(
        r"^[Vv]", "", regex=True
    )

    metadata["viewpoint"] = pd.to_numeric(
        normalized_viewpoint, errors="coerce"
    )

    # ---------------------------------------------------------
    # Normalize angle
    # ---------------------------------------------------------
    # Accept numeric angles as well as strings such as:
    #     0, +12, -12, +12.0, -12.0, 12 degrees, 12°
    raw_angle = metadata["angle"].astype(str).str.strip()

    normalized_angle = (
        raw_angle
        .str.replace("°", "", regex=False)
        .str.replace("degrees", "", regex=False, case=False)
        .str.replace("degree", "", regex=False, case=False)
        .str.strip()
    )

    metadata["angle"] = pd.to_numeric(
        normalized_angle, errors="coerce"
    )

    # ---------------------------------------------------------
    # Validate conversion
    # ---------------------------------------------------------
    if metadata["viewpoint"].isna().any():
        bad = (
            raw_viewpoint[metadata["viewpoint"].isna()]
            .drop_duplicates()
            .tolist()
        )
        raise RuntimeError(
            "Metadata contains invalid viewpoint values.\n"
            "Expected values such as 0..214 or V000..V214.\n"
            f"Examples: {bad[:20]}"
        )

    if metadata["angle"].isna().any():
        bad = (
            raw_angle[metadata["angle"].isna()]
            .drop_duplicates()
            .tolist()
        )
        raise RuntimeError(
            "Metadata contains invalid angle values.\n"
            "Expected numeric angles such as -107..107.\n"
            f"Examples: {bad[:20]}"
        )

    metadata["viewpoint"] = metadata["viewpoint"].astype(int)
    metadata["angle"] = metadata["angle"].astype(float)

    # ---------------------------------------------------------
    # Validate viewpoint range
    # ---------------------------------------------------------
    invalid_viewpoints = metadata[
        (metadata["viewpoint"] < MIN_VIEWPOINT)
        | (metadata["viewpoint"] > MAX_VIEWPOINT)
    ]

    if not invalid_viewpoints.empty:
        examples = (
            invalid_viewpoints["viewpoint"]
            .drop_duplicates()
            .tolist()
        )
        raise RuntimeError(
            f"Metadata contains viewpoint values outside "
            f"{MIN_VIEWPOINT}-{MAX_VIEWPOINT}.\n"
            f"Examples: {examples[:20]}"
        )

    return metadata


def load_shards(
    embedding_dir: Path,
) -> List[Tuple[Path, Path]]:
    embedding_files = sorted(
        embedding_dir.glob("embeddings_*.npy"),
        key=natural_key,
    )

    pairs = []

    for emb_path in embedding_files:
        match = re.search(r"(\d+)$", emb_path.stem)
        if not match:
            continue

        idx = match.group(1)
        meta_path = embedding_dir / f"metadata_{idx}.csv"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing metadata shard for {emb_path.name}: "
                f"{meta_path.name}"
            )

        pairs.append((emb_path, meta_path))

    if not pairs:
        raise FileNotFoundError(
            f"No embeddings_*.npy files found in {embedding_dir}"
        )

    return pairs


def validate_shards(
    shards: List[Tuple[Path, Path]],
    global_metadata: pd.DataFrame,
):
    total = 0

    for emb_path, meta_path in shards:
        emb = np.load(emb_path, mmap_mode="r")
        meta = pd.read_csv(meta_path)

        if len(emb) != len(meta):
            raise RuntimeError(
                f"Shard mismatch:\n"
                f"  {emb_path.name}: {len(emb)} embeddings\n"
                f"  {meta_path.name}: {len(meta)} metadata rows"
            )

        if emb.ndim != 2:
            raise RuntimeError(
                f"{emb_path.name} is not a 2D embedding matrix: {emb.shape}"
            )

        if emb.shape[1] != 768:
            raise RuntimeError(
                f"{emb_path.name} has embedding dimension {emb.shape[1]}, "
                f"but this pipeline expects DINOv2 ViT-B/14 = 768."
            )

        total += len(emb)

    if total != len(global_metadata):
        raise RuntimeError(
            f"Total shard rows ({total}) != global metadata rows "
            f"({len(global_metadata)})."
        )

    print(f"Validated {len(shards)} embedding shards.")
    print(f"Total embedding rows: {total:,}")


# ---------------------------------------------------------------------------
# Complete sequence selection
# ---------------------------------------------------------------------------

def find_complete_folders(metadata: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    counts = (
        metadata.groupby("folder")["viewpoint"]
        .nunique()
        .sort_index()
    )

    complete = counts[counts == EXPECTED_VIEWPOINTS].index.tolist()

    incomplete = counts[counts != EXPECTED_VIEWPOINTS].reset_index()
    incomplete.columns = ["folder", "viewpoint_count"]

    print()
    print("SEQUENCE COVERAGE")
    print("-" * 72)
    print(f"Total folders:              {len(counts):,}")
    print(f"Complete 215-view folders:  {len(complete):,}")
    print(f"Excluded incomplete:         {len(incomplete):,}")

    if len(incomplete):
        print()
        print("Excluded folders:")
        for _, row in incomplete.iterrows():
            print(
                f"  {row['folder']}: "
                f"{int(row['viewpoint_count'])} viewpoints"
            )

    return complete, incomplete


# ---------------------------------------------------------------------------
# Robust metadata normalization for embedding shards
# ---------------------------------------------------------------------------

def normalize_viewpoint_series(series: pd.Series) -> pd.Series:
    """Normalize viewpoint values such as 107 or V107 to integer 0..214."""
    raw = series.astype(str).str.strip()
    normalized = raw.str.replace(r"^[Vv]", "", regex=True)
    values = pd.to_numeric(normalized, errors="coerce")

    if values.isna().any():
        bad = raw[values.isna()].drop_duplicates().tolist()
        raise RuntimeError(
            "Embedding-shard metadata contains invalid viewpoint values.\n"
            "Expected values such as 0..214 or V000..V214.\n"
            f"Examples: {bad[:20]}"
        )

    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Embedding-shard metadata contains non-finite viewpoint values.")

    if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        bad = values[values % 1 != 0].drop_duplicates().tolist()
        raise RuntimeError(
            "Embedding-shard metadata contains non-integer viewpoint values.\n"
            f"Examples: {bad[:20]}"
        )

    return values.astype(int)


# ---------------------------------------------------------------------------
# Build reference representations
# ---------------------------------------------------------------------------

def build_references(
    shards: List[Tuple[Path, Path]],
    complete_folders: set,
):
    """
    Builds two references per expression sequence:

    1) frontal_ref:
       exact V107 embedding, used for A.

    2) prototype:
       mean of normalized embeddings from |angle| in [1, 5] degrees,
       excluding V107, used for B/C.

    This avoids trivial self-matching at V107 for B/C.
    """
    frontal: Dict[str, np.ndarray] = {}
    proto_sum: Dict[str, np.ndarray] = {}
    proto_count: Dict[str, int] = {}

    for emb_path, meta_path in shards:
        emb = np.load(emb_path).astype(np.float32)
        meta = pd.read_csv(meta_path)

        meta["viewpoint"] = normalize_viewpoint_series(meta["viewpoint"])
        meta["angle"] = pd.to_numeric(
            meta["angle"], errors="coerce"
        ).astype(float)

        keep = meta["folder"].isin(complete_folders)

        if not keep.any():
            continue

        indices = np.flatnonzero(keep.to_numpy())
        emb_norm = normalize_rows(emb)

        for idx in indices:
            row = meta.iloc[idx]
            folder = row["folder"]
            viewpoint = int(row["viewpoint"])
            angle = float(row["angle"])

            vec = emb_norm[idx]

            if viewpoint == FRONTAL_VIEWPOINT:
                if folder in frontal:
                    raise RuntimeError(
                        f"Duplicate frontal viewpoint for folder: {folder}"
                    )
                frontal[folder] = vec.copy()

            if (
                viewpoint != FRONTAL_VIEWPOINT
                and abs(angle) <= PROTOTYPE_WINDOW_DEG
                and abs(angle) >= 1
            ):
                if folder not in proto_sum:
                    proto_sum[folder] = np.zeros(
                        emb_norm.shape[1],
                        dtype=np.float32,
                    )
                    proto_count[folder] = 0

                proto_sum[folder] += vec
                proto_count[folder] += 1

    missing_frontal = complete_folders - set(frontal)
    if missing_frontal:
        raise RuntimeError(
            "Some complete folders are missing V107 frontal embeddings:\n"
            + "\n".join(sorted(missing_frontal)[:20])
        )

    prototypes = {}

    missing_proto = []

    for folder in complete_folders:
        if folder not in proto_sum or proto_count.get(folder, 0) == 0:
            missing_proto.append(folder)
            continue

        prototypes[folder] = normalize_vector(
            proto_sum[folder] / proto_count[folder]
        )

    if missing_proto:
        raise RuntimeError(
            "Could not build near-frontal prototypes for some folders:\n"
            + "\n".join(sorted(missing_proto)[:20])
        )

    folders = sorted(complete_folders)

    frontal_matrix = np.stack(
        [frontal[f] for f in folders],
        axis=0,
    ).astype(np.float32)

    prototype_matrix = np.stack(
        [prototypes[f] for f in folders],
        axis=0,
    ).astype(np.float32)

    folder_to_index = {
        folder: i for i, folder in enumerate(folders)
    }

    return (
        folders,
        folder_to_index,
        frontal_matrix,
        prototype_matrix,
    )


# ---------------------------------------------------------------------------
# Threshold estimation
# ---------------------------------------------------------------------------

def estimate_thresholds(
    per_view: pd.DataFrame,
) -> Dict[str, float]:
    """
    Robust thresholds are estimated from the near-frontal region.

    A:
      95th percentile of representation angular drift for |angle| <= 5.

    C:
      5th percentile of expression margin for |angle| <= 5.

    These are not universal constants. They are data-derived reference
    thresholds for this controlled benchmark.
    """
    baseline = per_view[
        per_view["angle"].abs() <= BASELINE_DEG
    ]

    if baseline.empty:
        raise RuntimeError("No near-frontal rows available for thresholds.")

    a_threshold = float(
        np.nanquantile(
            baseline["A_drift_angle_deg"].to_numpy(),
            A_DRIFT_QUANTILE,
        )
    )

    c_threshold = float(
        np.nanquantile(
            baseline["C_margin"].to_numpy(),
            C_MARGIN_QUANTILE,
        )
    )

    return {
        "A_drift_threshold_deg": a_threshold,
        "C_margin_threshold": c_threshold,
        "A_quantile": A_DRIFT_QUANTILE,
        "C_quantile": C_MARGIN_QUANTILE,
        "baseline_window_deg": BASELINE_DEG,
    }


# ---------------------------------------------------------------------------
# Main per-view analysis
# ---------------------------------------------------------------------------

def analyze_shards(
    shards: List[Tuple[Path, Path]],
    complete_folders: set,
    folder_to_index: Dict[str, int],
    frontal_matrix: np.ndarray,
    prototype_matrix: np.ndarray,
    folders: List[str],
) -> pd.DataFrame:
    """
    For every image:

    A:
        cosine similarity to its exact V107 reference
        -> angular representation drift

    B:
        nearest near-frontal prototype among all complete expression
        sequences
        -> predicted folder and correctness

    C:
        own prototype similarity - strongest rival similarity
        -> semantic separability margin
    """
    records = []

    prototype_matrix = normalize_rows(prototype_matrix)

    total_rows = sum(
        len(np.load(emb_path, mmap_mode="r"))
        for emb_path, _ in shards
    )

    processed = 0
    start = time.time()

    for emb_path, meta_path in shards:
        emb = np.load(emb_path).astype(np.float32)
        meta = pd.read_csv(meta_path)

        meta["viewpoint"] = normalize_viewpoint_series(meta["viewpoint"])
        meta["angle"] = pd.to_numeric(
            meta["angle"], errors="coerce"
        ).astype(float)

        keep = meta["folder"].isin(complete_folders)

        if not keep.any():
            processed += len(meta)
            continue

        keep_idx = np.flatnonzero(keep.to_numpy())

        for start_idx in range(0, len(keep_idx), 4096):
            idx = keep_idx[start_idx:start_idx + 4096]

            batch = normalize_rows(emb[idx])

            # --------------------------------------------------------------
            # A: Representation drift
            # --------------------------------------------------------------
            own_folder_indices = np.array(
                [folder_to_index[meta.iloc[i]["folder"]] for i in idx],
                dtype=np.int64,
            )

            refs = frontal_matrix[own_folder_indices]

            own_cos = np.sum(batch * refs, axis=1)
            own_cos = np.clip(own_cos, -1.0, 1.0)

            drift_1_minus_cos = 1.0 - own_cos
            drift_angle_deg = angular_distance_deg_from_cos(own_cos)

            # --------------------------------------------------------------
            # B + C: compare against all expression prototypes
            # --------------------------------------------------------------
            similarities = batch @ prototype_matrix.T

            best_idx = np.argmax(similarities, axis=1)
            best_similarity = similarities[
                np.arange(len(idx)),
                best_idx,
            ]

            # strongest rival after excluding the true sequence
            true_similarity = similarities[
                np.arange(len(idx)),
                own_folder_indices,
            ]

            similarities[
                np.arange(len(idx)),
                own_folder_indices,
            ] = -np.inf

            rival_idx = np.argmax(similarities, axis=1)
            rival_similarity = similarities[
                np.arange(len(idx)),
                rival_idx,
            ]

            margin = true_similarity - rival_similarity

            for j, row_index in enumerate(idx):
                row = meta.iloc[row_index]

                true_folder = row["folder"]
                predicted_folder = folders[int(best_idx[j])]

                records.append({
                    "folder": true_folder,
                    "gender": row["gender"],
                    "expression": row["expression"],
                    "viewpoint": int(row["viewpoint"]),
                    "angle": float(row["angle"]),
                    "image_path": row["image_path"],

                    # A
                    "A_cosine_to_V107": float(own_cos[j]),
                    "A_drift_1_minus_cos": float(
                        drift_1_minus_cos[j]
                    ),
                    "A_drift_angle_deg": float(
                        drift_angle_deg[j]
                    ),

                    # B
                    "B_predicted_folder": predicted_folder,
                    "B_correct": int(
                        predicted_folder == true_folder
                    ),
                    "B_top1_similarity": float(
                        best_similarity[j]
                    ),

                    # C
                    "C_rival_folder": folders[int(rival_idx[j])],
                    "C_own_similarity": float(
                        true_similarity[j]
                    ),
                    "C_rival_similarity": float(
                        rival_similarity[j]
                    ),
                    "C_margin": float(margin[j]),
                })

            processed += len(idx)

        elapsed = max(time.time() - start, 1e-9)
        speed = processed / elapsed

        print(
            f"Processed {processed:,}/{total_rows:,} rows "
            f"({speed:,.0f} rows/s)"
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Critical viewpoint analysis
# ---------------------------------------------------------------------------

def first_sustained_boundary(
    df: pd.DataFrame,
    value_column: str,
    condition,
) -> Tuple[float | None, float | None]:
    """
    Search from frontal viewpoint outward in one direction.

    Returns:
        signed angle at boundary,
        absolute angle
    """
    ordered = df.sort_values("angle")

    # Search negative side: -1, -2, ..., -107
    negative = ordered[
        (ordered["angle"] <= 0) &
        (ordered["angle"] >= -MAX_VIEWPOINT)
    ].copy()

    negative = negative.sort_values(
        "angle",
        ascending=False,
    )

    # Search positive side: +1, +2, ..., +107
    positive = ordered[
        (ordered["angle"] >= 0) &
        (ordered["angle"] <= MAX_VIEWPOINT)
    ].copy()

    positive = positive.sort_values(
        "angle",
        ascending=True,
    )

    results = []

    for direction_df in [negative, positive]:
        if direction_df.empty:
            results.append((None, None))
            continue

        values = direction_df[value_column].to_numpy()
        mask = condition(values)

        idx = sustained_first_index(mask)

        if idx is None:
            results.append((None, None))
        else:
            angle = float(direction_df.iloc[idx]["angle"])
            results.append((angle, abs(angle)))

    # [left, right]
    return results[0], results[1]


def expression_boundaries(
    per_view: pd.DataFrame,
    thresholds: Dict[str, float],
):
    rows = []

    for folder, group in per_view.groupby("folder", sort=True):
        group = group.sort_values("angle")

        # A: representation drift becomes unusually large
        a_left, a_right = first_sustained_boundary(
            group,
            "A_drift_angle_deg",
            lambda x: x > thresholds["A_drift_threshold_deg"],
        )

        # B: prediction is wrong for 3 consecutive viewpoints
        b_left, b_right = first_sustained_boundary(
            group,
            "B_correct",
            lambda x: x < 0.5,
        )

        # C: margin drops below robust near-frontal lower bound
        c_left, c_right = first_sustained_boundary(
            group,
            "C_margin",
            lambda x: x < thresholds["C_margin_threshold"],
        )

        # Hard C failure: rival is actually closer than own prototype
        c_hard_left, c_hard_right = first_sustained_boundary(
            group,
            "C_margin",
            lambda x: x < 0.0,
        )

        row = {
            "folder": folder,

            "A_left_angle": a_left[0],
            "A_left_abs_angle": a_left[1],
            "A_right_angle": a_right[0],
            "A_right_abs_angle": a_right[1],

            "B_left_angle": b_left[0],
            "B_left_abs_angle": b_left[1],
            "B_right_angle": b_right[0],
            "B_right_abs_angle": b_right[1],

            "C_left_angle": c_left[0],
            "C_left_abs_angle": c_left[1],
            "C_right_angle": c_right[0],
            "C_right_abs_angle": c_right[1],

            "C_hard_left_angle": c_hard_left[0],
            "C_hard_left_abs_angle": c_hard_left[1],
            "C_hard_right_angle": c_hard_right[0],
            "C_hard_right_abs_angle": c_hard_right[1],
        }

        # --------------------------------------------------------------
        # Integrated trajectory quantities
        # --------------------------------------------------------------
        left = group[
            (group["angle"] >= -MAX_VIEWPOINT) &
            (group["angle"] <= 0)
        ].sort_values("angle")

        right = group[
            (group["angle"] >= 0) &
            (group["angle"] <= MAX_VIEWPOINT)
        ].sort_values("angle")

        row["A_left_integrated_drift_deg"] = trapz_mean(
            left["A_drift_angle_deg"].to_numpy(),
            left["angle"].to_numpy(),
        )

        row["A_right_integrated_drift_deg"] = trapz_mean(
            right["A_drift_angle_deg"].to_numpy(),
            right["angle"].to_numpy(),
        )

        row["A_left_integrated_1_minus_cos"] = trapz_mean(
            left["A_drift_1_minus_cos"].to_numpy(),
            left["angle"].to_numpy(),
        )

        row["A_right_integrated_1_minus_cos"] = trapz_mean(
            right["A_drift_1_minus_cos"].to_numpy(),
            right["angle"].to_numpy(),
        )

        row["B_left_consistency_AUC"] = trapz_mean(
            left["B_correct"].to_numpy(),
            left["angle"].to_numpy(),
        )

        row["B_right_consistency_AUC"] = trapz_mean(
            right["B_correct"].to_numpy(),
            right["angle"].to_numpy(),
        )

        row["C_left_margin_AUC"] = trapz_mean(
            left["C_margin"].to_numpy(),
            left["angle"].to_numpy(),
        )

        row["C_right_margin_AUC"] = trapz_mean(
            right["C_margin"].to_numpy(),
            right["angle"].to_numpy(),
        )

        # --------------------------------------------------------------
        # Lead / lag:
        # Does A cross its instability threshold before B fails?
        # Does C margin collapse before B fails?
        # --------------------------------------------------------------
        for direction in ["left", "right"]:
            a = row[f"A_{direction}_abs_angle"]
            b = row[f"B_{direction}_abs_angle"]
            c = row[f"C_{direction}_abs_angle"]

            row[f"A_leads_B_{direction}"] = (
                int(a < b)
                if a is not None and b is not None
                else np.nan
            )

            row[f"C_leads_B_{direction}"] = (
                int(c < b)
                if c is not None and b is not None
                else np.nan
            )

            row[f"B_minus_A_{direction}"] = (
                b - a
                if a is not None and b is not None
                else np.nan
            )

            row[f"B_minus_C_{direction}"] = (
                b - c
                if c is not None and b is not None
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Population curves
# ---------------------------------------------------------------------------

def population_profile(per_view: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate both sides using absolute viewpoint angle.

    For each |angle|:
      A = mean representation drift
      B = mean expression consistency
      C = mean semantic margin

    This produces the benchmark-level reliability profile.
    """
    temp = per_view.copy()
    temp["abs_angle"] = temp["angle"].abs()

    profile = (
        temp.groupby("abs_angle")
        .agg(
            A_mean_drift_deg=("A_drift_angle_deg", "mean"),
            A_median_drift_deg=("A_drift_angle_deg", "median"),
            A_mean_1_minus_cos=("A_drift_1_minus_cos", "mean"),

            B_accuracy=("B_correct", "mean"),

            C_mean_margin=("C_margin", "mean"),
            C_median_margin=("C_margin", "median"),

            N=("folder", "nunique"),
        )
        .reset_index()
        .sort_values("abs_angle")
    )

    return profile


def directional_profile(per_view: pd.DataFrame) -> pd.DataFrame:
    temp = per_view.copy()

    temp["direction"] = np.where(
        temp["angle"] < 0,
        "left",
        np.where(temp["angle"] > 0, "right", "frontal"),
    )
    temp["abs_angle"] = temp["angle"].abs()

    profile = (
        temp.groupby(["direction", "abs_angle"])
        .agg(
            A_mean_drift_deg=("A_drift_angle_deg", "mean"),
            B_accuracy=("B_correct", "mean"),
            C_mean_margin=("C_margin", "mean"),
            N=("folder", "nunique"),
        )
        .reset_index()
    )

    return profile.sort_values(
        ["direction", "abs_angle"]
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_population_ci(
    per_view: pd.DataFrame,
    n_bootstrap: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap complete expression sequences, not individual images.

    This is important: the 215 viewpoints belonging to one expression are
    not independent observations. The unit of resampling is therefore the
    expression sequence / folder.
    """
    rng = np.random.default_rng(seed)

    folders = sorted(per_view["folder"].unique())
    angles = np.arange(0, MAX_VIEWPOINT + 1)

    # Build matrices [folders, abs_angle].
    A = np.full((len(folders), len(angles)), np.nan, dtype=np.float32)
    B = np.full_like(A, np.nan)
    C = np.full_like(A, np.nan)

    folder_index = {f: i for i, f in enumerate(folders)}

    grouped = (
        per_view.assign(abs_angle=per_view["angle"].abs())
        .groupby(["folder", "abs_angle"])
        .agg(
            A=("A_drift_angle_deg", "mean"),
            B=("B_correct", "mean"),
            C=("C_margin", "mean"),
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():
        i = folder_index[row["folder"]]
        j = int(row["abs_angle"])

        if 0 <= j <= MAX_VIEWPOINT:
            A[i, j] = row["A"]
            B[i, j] = row["B"]
            C[i, j] = row["C"]

    rows = []

    for angle in angles:
        a = A[:, angle]
        b = B[:, angle]
        c = C[:, angle]

        valid_a = np.isfinite(a)
        valid_b = np.isfinite(b)
        valid_c = np.isfinite(c)

        if not valid_a.any():
            continue

        boot_a = np.empty(n_bootstrap, dtype=np.float32)
        boot_b = np.empty(n_bootstrap, dtype=np.float32)
        boot_c = np.empty(n_bootstrap, dtype=np.float32)

        valid_indices = np.arange(len(folders))

        for k in range(n_bootstrap):
            sample = rng.choice(
                valid_indices,
                size=len(valid_indices),
                replace=True,
            )

            boot_a[k] = np.nanmean(a[sample])
            boot_b[k] = np.nanmean(b[sample])
            boot_c[k] = np.nanmean(c[sample])

        rows.append({
            "abs_angle": int(angle),

            "A_mean": float(np.nanmean(a)),
            "A_ci_low": float(np.nanpercentile(boot_a, 2.5)),
            "A_ci_high": float(np.nanpercentile(boot_a, 97.5)),

            "B_mean": float(np.nanmean(b)),
            "B_ci_low": float(np.nanpercentile(boot_b, 2.5)),
            "B_ci_high": float(np.nanpercentile(boot_b, 97.5)),

            "C_mean": float(np.nanmean(c)),
            "C_ci_low": float(np.nanpercentile(boot_c, 2.5)),
            "C_ci_high": float(np.nanpercentile(boot_c, 97.5)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_boundaries(
    summary: pd.DataFrame,
) -> Dict[str, float]:
    out = {}

    for metric in ["A", "B", "C"]:
        left = summary[f"{metric}_left_abs_angle"].dropna()
        right = summary[f"{metric}_right_abs_angle"].dropna()

        out[f"{metric}_left_boundary_median"] = (
            float(left.median()) if len(left) else np.nan
        )
        out[f"{metric}_right_boundary_median"] = (
            float(right.median()) if len(right) else np.nan
        )

        out[f"{metric}_left_boundary_detection_rate"] = (
            float(left.notna().mean())
            if len(summary)
            else np.nan
        )
        out[f"{metric}_right_boundary_detection_rate"] = (
            float(right.notna().mean())
            if len(summary)
            else np.nan
        )

    for col in [
        "A_leads_B_left",
        "A_leads_B_right",
        "C_leads_B_left",
        "C_leads_B_right",
    ]:
        if col in summary:
            values = summary[col].dropna()
            out[f"{col}_rate"] = (
                float(values.mean()) if len(values) else np.nan
            )

    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    profile: pd.DataFrame,
    output_dir: Path,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; skipping plots. "
            "Install with: pip install matplotlib"
        )
        return

    x = profile["abs_angle"].to_numpy()

    # A: one figure
    plt.figure(figsize=(9, 5))
    plt.plot(
        x,
        profile["A_mean_drift_deg"].to_numpy(),
        linewidth=2,
    )
    plt.xlabel("Absolute viewpoint angle from frontal (degrees)")
    plt.ylabel("Mean representation drift (angular degrees)")
    plt.title("A — Representation Drift")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "A_representation_drift.png",
        dpi=180,
    )
    plt.close()

    # B: one figure
    plt.figure(figsize=(9, 5))
    plt.plot(
        x,
        profile["B_accuracy"].to_numpy(),
        linewidth=2,
    )
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Absolute viewpoint angle from frontal (degrees)")
    plt.ylabel("Expression consistency / top-1 accuracy")
    plt.title("B — Expression Consistency")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "B_expression_consistency.png",
        dpi=180,
    )
    plt.close()

    # C: one figure
    plt.figure(figsize=(9, 5))
    plt.plot(
        x,
        profile["C_mean_margin"].to_numpy(),
        linewidth=2,
    )
    plt.axhline(
        0.0,
        linewidth=1,
        linestyle="--",
    )
    plt.xlabel("Absolute viewpoint angle from frontal (degrees)")
    plt.ylabel("Own-prototype minus rival similarity")
    plt.title("C — Expression Separability")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / "C_expression_separability.png",
        dpi=180,
    )
    plt.close()

    print("Plots saved.")


# ---------------------------------------------------------------------------
# README for the generated analysis
# ---------------------------------------------------------------------------

def write_analysis_readme(
    output_dir: Path,
    thresholds: Dict[str, float],
):
    text = f"""# Embedding Analysis

This directory contains the first representation-level analysis of the
FER Reliability Benchmark.

## Three measurements

### A — Representation Drift

For each expression sequence, the exact frontal embedding at V107 is the
reference.

For viewpoint v:

- cosine similarity to V107 is computed
- angular embedding distance is computed
- the full viewpoint trajectory is integrated with the trapezoidal rule

The A boundary is the first sustained (3-view) crossing of the robust
near-frontal drift threshold.

A threshold:
`{thresholds["A_drift_threshold_deg"]:.6f}` angular degrees

### B — Expression Consistency

There is NO six-class emotion classifier here.

Each complete dataset folder is treated as one controlled expression
sequence.

A near-frontal prototype is built from viewpoints within +/-5 degrees,
excluding V107.

Every test viewpoint is matched against all complete expression prototypes.

B asks:

> Does the viewpoint image still retrieve its own expression sequence?

A failure boundary is the first sustained 3-viewpoint run in which the
correct sequence is no longer top-1.

### C — Expression Separability

For each viewpoint:

`C_margin = similarity_to_own_prototype - similarity_to_best_rival`

Positive margin:
the correct expression prototype is closer.

Zero:
the correct and rival prototypes are tied.

Negative:
another expression prototype is closer.

The C boundary uses a robust lower threshold estimated from the near-frontal
region.

C also reports a hard boundary where the margin becomes negative.

## Important scientific point

The mathematical ingredients themselves are not claimed as new:

- cosine similarity
- nearest-prototype retrieval
- margin
- numerical integration
- bootstrap confidence intervals

The research contribution is the controlled combination:

1. fixed expression
2. viewpoint-only perturbation
3. continuous representation trajectory
4. prediction consistency trajectory
5. separability trajectory
6. critical viewpoint boundary
7. integrated trajectory measures
8. lead/lag analysis of representation change versus prediction failure

The most important question is whether A or C changes systematically BEFORE B
fails.

That result, rather than the choice of cosine itself, is the potential
scientific contribution.
"""
    (output_dir / "README_analysis.md").write_text(
        text,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DINOv2 embeddings for viewpoint reliability."
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help="Number of bootstrap repetitions (default: 500).",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate PNG plots.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap.",
    )

    args = parser.parse_args()

    root = project_root()
    embedding_dir = find_embedding_dir(root)

    output_dir = root / "analysis" / "analyze_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FER REPRESENTATION RELIABILITY BENCHMARK")
    print("A/B/C VIEWPOINT ANALYSIS")
    print("=" * 78)
    print(f"Project root:       {root}")
    print(f"Embedding directory:{embedding_dir}")
    print(f"Output directory:   {output_dir}")
    print()

    # ----------------------------------------------------------------------
    # 1. Load metadata
    # ----------------------------------------------------------------------
    metadata = load_metadata(root)

    print(f"Metadata rows: {len(metadata):,}")

    # ----------------------------------------------------------------------
    # 2. Find shards and validate them
    # ----------------------------------------------------------------------
    shards = load_shards(embedding_dir)
    validate_shards(shards, metadata)

    # ----------------------------------------------------------------------
    # 3. Select complete sequences
    # ----------------------------------------------------------------------
    complete_folders, incomplete = find_complete_folders(metadata)
    complete_set = set(complete_folders)

    if len(complete_folders) < 2:
        raise RuntimeError(
            "At least two complete expression sequences are required."
        )

    # ----------------------------------------------------------------------
    # 4. Build A reference and B/C prototypes
    # ----------------------------------------------------------------------
    print()
    print("BUILDING REFERENCES")
    print("-" * 78)

    (
        folders,
        folder_to_index,
        frontal_matrix,
        prototype_matrix,
    ) = build_references(
        shards,
        complete_set,
    )

    print(f"Complete expression sequences: {len(folders):,}")
    print(f"Frontal reference matrix:       {frontal_matrix.shape}")
    print(f"B/C prototype matrix:            {prototype_matrix.shape}")

    # ----------------------------------------------------------------------
    # 5. Per-view analysis
    # ----------------------------------------------------------------------
    print()
    print("ANALYZING A / B / C")
    print("-" * 78)

    per_view = analyze_shards(
        shards,
        complete_set,
        folder_to_index,
        frontal_matrix,
        prototype_matrix,
        folders,
    )

    per_view = per_view.sort_values(
        ["folder", "angle"]
    ).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 6. Thresholds
    # ----------------------------------------------------------------------
    thresholds = estimate_thresholds(per_view)

    print()
    print("ROBUST THRESHOLDS")
    print("-" * 78)
    print(
        f"A drift threshold: "
        f"{thresholds['A_drift_threshold_deg']:.6f} degrees"
    )
    print(
        f"C margin threshold: "
        f"{thresholds['C_margin_threshold']:.6f}"
    )
    print(
        f"Baseline region: +/-{BASELINE_DEG} degrees"
    )
    print(
        f"Sustained boundary: "
        f"{SUSTAINED_FAILURE} consecutive viewpoints"
    )

    # ----------------------------------------------------------------------
    # 7. Expression-level boundaries
    # ----------------------------------------------------------------------
    summary = expression_boundaries(
        per_view,
        thresholds,
    )

    # Add gender/expression metadata.
    folder_info = (
        metadata[metadata["folder"].isin(complete_set)]
        .groupby("folder")
        .first()
        .reset_index()[
            ["folder", "gender", "expression"]
        ]
    )

    summary = folder_info.merge(
        summary,
        on="folder",
        how="right",
    )

    summary = summary.sort_values("folder").reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 8. Population profile
    # ----------------------------------------------------------------------
    profile = population_profile(per_view)
    directional = directional_profile(per_view)

    # ----------------------------------------------------------------------
    # 9. Bootstrap CIs
    # ----------------------------------------------------------------------
    print()
    print(
        f"BOOTSTRAP ({args.bootstrap} repetitions; "
        f"resampling complete expression sequences)"
    )
    print("-" * 78)

    bootstrap_ci = bootstrap_population_ci(
        per_view,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    # ----------------------------------------------------------------------
    # 10. Global report
    # ----------------------------------------------------------------------
    boundary_stats = summarize_boundaries(summary)

    report = {
        "benchmark": "FER Representation Reliability Benchmark",
        "embedding_model": "DINOv2 ViT-B/14",
        "embedding_dimension": 768,

        "dataset_rows": int(len(metadata)),
        "total_folders": int(metadata["folder"].nunique()),
        "complete_215_view_folders": int(len(complete_folders)),
        "excluded_incomplete_folders": int(len(incomplete)),

        "viewpoints": {
            "count": EXPECTED_VIEWPOINTS,
            "min": MIN_VIEWPOINT,
            "max": MAX_VIEWPOINT,
            "frontal_viewpoint": FRONTAL_VIEWPOINT,
        },

        "prototype": {
            "window_degrees": PROTOTYPE_WINDOW_DEG,
            "excludes_frontal_view": True,
        },

        "boundary_method": {
            "sustained_consecutive_views": SUSTAINED_FAILURE,
            "baseline_window_degrees": BASELINE_DEG,
            "A_quantile": A_DRIFT_QUANTILE,
            "C_quantile": C_MARGIN_QUANTILE,
            "A_drift_threshold_deg": thresholds[
                "A_drift_threshold_deg"
            ],
            "C_margin_threshold": thresholds[
                "C_margin_threshold"
            ],
        },

        "boundary_statistics": boundary_stats,

        "scientific_interpretation": {
            "A": (
                "Representation drift relative to the exact frontal "
                "reference."
            ),
            "B": (
                "Top-1 retrieval consistency among complete expression "
                "sequences."
            ),
            "C": (
                "Own-expression prototype margin over the strongest rival."
            ),
            "key_test": (
                "Determine whether A/C boundaries systematically precede "
                "B prediction failure."
            ),
        },

        "caution": (
            "The individual mathematical operations are standard. "
            "Novelty should be claimed from the controlled benchmark, "
            "trajectory-level analysis, critical viewpoint boundaries, "
            "and empirical findings—not from cosine similarity or "
            "numerical integration alone."
        ),
    }

    # ----------------------------------------------------------------------
    # 11. Save everything
    # ----------------------------------------------------------------------
    per_view_path = output_dir / "per_view_metrics.csv"
    summary_path = output_dir / "per_expression_summary.csv"
    profile_path = output_dir / "population_profile.csv"
    directional_path = output_dir / "directional_summary.csv"
    bootstrap_path = output_dir / "bootstrap_population_ci.csv"
    report_path = output_dir / "analysis_report.json"

    per_view.to_csv(
        per_view_path,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    profile.to_csv(
        profile_path,
        index=False,
        encoding="utf-8",
    )

    directional.to_csv(
        directional_path,
        index=False,
        encoding="utf-8",
    )

    bootstrap_ci.to_csv(
        bootstrap_path,
        index=False,
        encoding="utf-8",
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=lambda x: None if (
                isinstance(x, float) and not np.isfinite(x)
            ) else x,
        ),
        encoding="utf-8",
    )

    write_analysis_readme(
        output_dir,
        thresholds,
    )

    # ----------------------------------------------------------------------
    # 12. Plots
    # ----------------------------------------------------------------------
    if not args.no_plots:
        make_plots(profile, output_dir)

    # ----------------------------------------------------------------------
    # 13. Console summary
    # ----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)

    print(f"Complete sequences analyzed : {len(complete_folders):,}")
    print(f"Images analyzed              : {len(per_view):,}")
    print()

    print("Median critical boundaries")
    print("-" * 78)

    for metric, name in [
        ("A", "Representation drift"),
        ("B", "Expression consistency failure"),
        ("C", "Expression separability collapse"),
    ]:
        left = summary[f"{metric}_left_abs_angle"].dropna()
        right = summary[f"{metric}_right_abs_angle"].dropna()

        left_text = (
            f"{left.median():.1f}°"
            if len(left)
            else "not detected"
        )

        right_text = (
            f"{right.median():.1f}°"
            if len(right)
            else "not detected"
        )

        print(f"{name:<38} left={left_text:<15} right={right_text}")

    print()
    print("Lead/lag")
    print("-" * 78)

    for col in [
        "A_leads_B_left",
        "A_leads_B_right",
        "C_leads_B_left",
        "C_leads_B_right",
    ]:
        if col in summary:
            values = summary[col].dropna()

            if len(values):
                print(
                    f"{col:<28}: "
                    f"{values.mean() * 100:.1f}%"
                )

    print()
    print("Output files:")
    print(f"  {per_view_path}")
    print(f"  {summary_path}")
    print(f"  {profile_path}")
    print(f"  {directional_path}")
    print(f"  {bootstrap_path}")
    print(f"  {report_path}")
    print(f"  {output_dir / 'README_analysis.md'}")

    if not args.no_plots:
        print(f"  {output_dir / 'A_representation_drift.png'}")
        print(f"  {output_dir / 'B_expression_consistency.png'}")
        print(f"  {output_dir / 'C_expression_separability.png'}")

    print()
    print(
        "IMPORTANT: The numbers printed above are empirical results, "
        "not predetermined hypotheses."
    )


if __name__ == "__main__":
    main()
