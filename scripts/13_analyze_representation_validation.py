# -*- coding: utf-8 -*-
"""
STAGE 9 — REPRESENTATION-LEVEL VALIDATION
FER Reliability Benchmark

This version is deliberately robust to:
1) embeddings_00000.npy ... embeddings_00009.npy
2) metadata_00000.csv ... metadata_00009.csv
3) metadata viewpoint values such as V000, V001, ...
4) missing/invalid angle columns: angle is reconstructed from viewpoint
5) JSON NaN/Infinity: converted to null before writing
6) missing BH helper
7) incomplete sequences
8) one-to-one alignment between embeddings and per-view metrics
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

ROOT = Path(r"D:\1405\FER-Reliability-Benchmark")

PER_VIEW = ROOT / "analysis" / "4_analyze_embeddings_trajectory" / "per_view_metrics_multimetric.csv"
EMB_DIR = ROOT / "data" / "embeddings"
OUT = ROOT / "analysis" / "13_analyze_representation_validation"

A_THRESHOLD = 13.43702602
C_THRESHOLD = 0.00237080

SUSTAINED = 3
FRONTAL = 107
MIN_VIEW = 0
MAX_VIEW = 214
NVIEW = 215

PROTO_DEG = 5
ALPHA = 0.05
SEED = 20260810

EMBED_DIM = 768


# ---------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------

def banner(text: str) -> None:
    print("\n" + "#" * 76)
    print(text)
    print("#" * 76)


def clean(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def natural_index(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else 10**12


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def l2_normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-12)


def cosine_to_angle(cosine: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def sustained_first(mask, run: int = SUSTAINED):
    """
    Return the first index at which `run` consecutive True values begin.
    """
    n = 0
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        n = n + 1 if value else 0
        if n >= run:
            return i - run + 1
    return None


# ---------------------------------------------------------------------
# VIEWPOINT / ANGLE RECOVERY
# ---------------------------------------------------------------------

def recover_viewpoint(series: pd.Series) -> pd.Series:
    """
    Recover viewpoint from values such as:
      0, 1, 107, 214
      V000, V001, V107, V214
      viewpoint_000
      "..._107.png"

    Priority:
      1) numeric viewpoint column
      2) V### pattern
      3) trailing 3-digit image filename index
    """
    out = pd.to_numeric(series, errors="coerce")

    text = series.astype(str)

    # V001 / v001
    extracted_v = pd.to_numeric(
        text.str.extract(r"(?i)\bV(\d{1,3})\b", expand=False),
        errors="coerce",
    )
    out = out.fillna(extracted_v)

    # Any final _000 / -000 / 000 before extension
    extracted_tail = pd.to_numeric(
        text.str.extract(r"(?<!\d)(\d{1,3})(?:\.png)?$", expand=False),
        errors="coerce",
    )
    out = out.fillna(extracted_tail)

    return out


def recover_angle(metadata: pd.DataFrame) -> pd.Series:
    """
    The benchmark uses V107 as frontal.

    Signed angle is reconstructed as:
        angle = viewpoint - 107

    Therefore:
        V000 -> -107
        V106 -> -1
        V107 -> 0
        V108 -> +1
        V214 -> +107

    If an existing angle column is valid and agrees with this convention,
    it is retained. Otherwise the deterministic viewpoint-derived angle
    is used.
    """
    viewpoint = metadata["viewpoint"].to_numpy(float)

    derived = viewpoint - FRONTAL

    if "angle" in metadata.columns:
        existing = pd.to_numeric(metadata["angle"], errors="coerce").to_numpy(float)

        # Accept existing values only where they are finite and consistent
        # with the benchmark's frontal-centered convention.
        valid = np.isfinite(existing) & np.isfinite(derived)
        consistent = valid & (np.abs(existing - derived) < 1e-6)

        result = derived.copy()
        result[consistent] = existing[consistent]
        return pd.Series(result, index=metadata.index)

    return pd.Series(derived, index=metadata.index)


# ---------------------------------------------------------------------
# SHARD DISCOVERY
# ---------------------------------------------------------------------

def load_shards():
    """
    Correctly pairs:

        embeddings_00000.npy <-> metadata_00000.csv
        embeddings_00001.npy <-> metadata_00001.csv
        ...

    Important:
    We NEVER assume metadata_0.csv when the actual file is
    metadata_00000.csv.
    """
    if not EMB_DIR.exists():
        raise FileNotFoundError(f"Embedding directory not found:\n{EMB_DIR}")

    embedding_files = sorted(
        EMB_DIR.glob("embeddings_*.npy"),
        key=natural_index,
    )

    if not embedding_files:
        raise FileNotFoundError(
            f"No embeddings_*.npy files found in:\n{EMB_DIR}"
        )

    pairs = []

    for emb_path in embedding_files:
        m = re.search(r"(\d+)$", emb_path.stem)
        if not m:
            continue

        shard_number = int(m.group(1))

        # First try exact zero-padded naming.
        candidates = [
            EMB_DIR / f"metadata_{shard_number:05d}.csv",
            EMB_DIR / f"metadata_{shard_number}.csv",
        ]

        meta_path = next((p for p in candidates if p.exists()), None)

        if meta_path is None:
            raise FileNotFoundError(
                "\nMissing metadata shard for:\n"
                f"  {emb_path.name}\n"
                "Expected one of:\n"
                + "\n".join(f"  {p.name}" for p in candidates)
            )

        pairs.append((emb_path, meta_path))

    if not pairs:
        raise FileNotFoundError("No valid embedding/metadata shard pairs found.")

    print(f"Embedding shards found: {len(pairs)}")
    for ep, mp in pairs:
        print(f"  {ep.name}  <->  {mp.name}")

    return pairs


# ---------------------------------------------------------------------
# LOAD EMBEDDINGS + METADATA
# ---------------------------------------------------------------------

def load_embeddings(pairs):
    metadata_frames = []
    embedding_arrays = []

    for emb_path, meta_path in pairs:
        print(f"\nLoading: {emb_path.name}")

        emb = np.load(emb_path, mmap_mode="r")
        meta = pd.read_csv(meta_path, low_memory=False)

        required = {
            "folder",
            "gender",
            "expression",
            "image_path",
        }

        missing = required - set(meta.columns)
        if missing:
            raise RuntimeError(
                f"{meta_path.name} is missing required columns: "
                f"{sorted(missing)}"
            )

        if emb.ndim != 2:
            raise RuntimeError(
                f"{emb_path.name}: expected 2-D embeddings, got {emb.shape}"
            )

        if emb.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"{emb_path.name}: expected embedding dimension "
                f"{EMBED_DIM}, got {emb.shape[1]}"
            )

        if len(emb) != len(meta):
            raise RuntimeError(
                f"Row mismatch:\n"
                f"  {emb_path.name}: {len(emb)} embeddings\n"
                f"  {meta_path.name}: {len(meta)} metadata rows"
            )

        # Copy because mmap may be read-only.
        meta = meta.copy()

        # Recover viewpoint robustly.
        if "viewpoint" in meta.columns:
            meta["viewpoint"] = recover_viewpoint(meta["viewpoint"])
        else:
            meta["viewpoint"] = recover_viewpoint(meta["image_path"])

        # If viewpoint is still missing, recover from image path.
        missing_viewpoint = meta["viewpoint"].isna()
        if missing_viewpoint.any():
            meta.loc[missing_viewpoint, "viewpoint"] = recover_viewpoint(
                meta.loc[missing_viewpoint, "image_path"]
            )

        # Validate.
        bad_view = (
            meta["viewpoint"].isna()
            | (meta["viewpoint"] < MIN_VIEW)
            | (meta["viewpoint"] > MAX_VIEW)
            | (meta["viewpoint"] % 1 != 0)
        )

        if bad_view.any():
            examples = meta.loc[
                bad_view,
                ["folder", "expression", "viewpoint", "image_path"],
            ].head(20)

            raise RuntimeError(
                "\nCould not recover valid viewpoint values.\n"
                f"Valid range is {MIN_VIEW}..{MAX_VIEW}.\n"
                "Examples:\n"
                f"{examples.to_string(index=False)}"
            )

        meta["viewpoint"] = meta["viewpoint"].astype(int)

        # Deterministic benchmark angle.
        meta["angle"] = recover_angle(meta).astype(float)

        meta["_embedding_shard"] = emb_path.name
        meta["_metadata_shard"] = meta_path.name

        metadata_frames.append(meta)
        embedding_arrays.append(np.asarray(emb, dtype=np.float32))

        print(
            f"  rows={len(meta):,} "
            f"viewpoint={meta.viewpoint.min()}..{meta.viewpoint.max()}"
        )

    metadata = pd.concat(metadata_frames, ignore_index=True)
    embeddings = np.concatenate(embedding_arrays, axis=0)

    if len(metadata) != len(embeddings):
        raise RuntimeError("Final metadata/embedding length mismatch.")

    # Final duplicate check for exact sequence/viewpoint rows.
    key = ["folder", "viewpoint"]
    dup = metadata.duplicated(key, keep=False)

    if dup.any():
        examples = metadata.loc[
            dup, ["folder", "viewpoint", "_embedding_shard"]
        ].head(30)

        raise RuntimeError(
            "Duplicate folder/viewpoint rows detected.\n"
            f"{examples.to_string(index=False)}"
        )

    print(f"\nTotal embedding rows: {len(embeddings):,}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Unique sequences: {metadata.folder.nunique():,}")

    return metadata, embeddings


# ---------------------------------------------------------------------
# COMPLETE SEQUENCES
# ---------------------------------------------------------------------

def complete_sequences(metadata: pd.DataFrame):
    counts = (
        metadata.groupby("folder")["viewpoint"]
        .nunique()
        .sort_values()
    )

    good = set(counts[counts == NVIEW].index)

    bad = counts[counts != NVIEW].reset_index()
    bad.columns = ["folder", "viewpoint_count"]

    return good, bad


# ---------------------------------------------------------------------
# FRONTAL REFERENCES + EXPRESSION PROTOTYPES
# ---------------------------------------------------------------------

def build_references(metadata, embeddings, good):
    keep = metadata.folder.isin(good)

    m = metadata.loc[keep].reset_index(drop=True)
    e = l2_normalize_rows(embeddings[keep.to_numpy()])

    frontal = {}
    prototype_sum = {}
    prototype_count = {}

    for i, row in m.iterrows():
        folder = clean(row["folder"])
        viewpoint = int(row["viewpoint"])
        angle = float(row["angle"])
        z = e[i]

        if viewpoint == FRONTAL:
            if folder in frontal:
                raise RuntimeError(
                    f"Duplicate frontal V{FRONTAL} in sequence: {folder}"
                )
            frontal[folder] = z

        # Near-frontal prototype: 1..5 degrees on either side.
        if viewpoint != FRONTAL and 1 <= abs(angle) <= PROTO_DEG:
            if folder not in prototype_sum:
                prototype_sum[folder] = np.zeros(EMBED_DIM, dtype=np.float32)
                prototype_count[folder] = 0

            prototype_sum[folder] += z
            prototype_count[folder] += 1

    missing_frontal = good - set(frontal)
    if missing_frontal:
        raise RuntimeError(
            "Missing frontal V107 for sequences:\n"
            + ", ".join(sorted(missing_frontal)[:30])
        )

    prototypes = {}

    for folder in good:
        if prototype_count.get(folder, 0) > 0:
            prototypes[folder] = l2_normalize_vector(
                prototype_sum[folder] / prototype_count[folder]
            )

    missing_proto = good - set(prototypes)
    if missing_proto:
        raise RuntimeError(
            "Missing near-frontal prototype for sequences:\n"
            + ", ".join(sorted(missing_proto)[:30])
        )

    folders = sorted(good)
    folder_index = {folder: i for i, folder in enumerate(folders)}

    frontal_matrix = np.stack([frontal[f] for f in folders])
    prototype_matrix = np.stack([prototypes[f] for f in folders])

    return folders, folder_index, frontal_matrix, prototype_matrix


# ---------------------------------------------------------------------
# REPRESENTATION METRICS
# ---------------------------------------------------------------------

def representation_metrics(
    metadata,
    embeddings,
    good,
    folders,
    folder_index,
    frontal_matrix,
    prototype_matrix,
):
    keep = metadata.folder.isin(good)

    m = metadata.loc[keep].reset_index(drop=True)
    e = l2_normalize_rows(embeddings[keep.to_numpy()])

    idx = np.array([folder_index[clean(x)] for x in m.folder])

    F = l2_normalize_rows(frontal_matrix)
    P = l2_normalize_rows(prototype_matrix)

    # Similarity to exact frontal representation.
    frontal_cos = np.sum(e * F[idx], axis=1)

    # Similarity to every expression prototype.
    sims = e @ P.T

    own_proto = sims[np.arange(len(m)), idx]

    rival_sims = sims.copy()
    rival_sims[np.arange(len(m)), idx] = -np.inf

    rival_idx = np.argmax(rival_sims, axis=1)
    rival_proto = rival_sims[np.arange(len(m)), rival_idx]

    retrieval_idx = np.argmax(sims, axis=1)

    out = m[
        [
            "folder",
            "gender",
            "expression",
            "viewpoint",
            "angle",
            "image_path",
        ]
    ].copy()

    out["side"] = np.where(
        out.angle < 0,
        "left",
        np.where(out.angle > 0, "right", "frontal"),
    )

    out["abs_angle"] = out.angle.abs()

    out["R_frontal_angular_deg"] = cosine_to_angle(frontal_cos)
    out["R_frontal_cosine"] = frontal_cos
    out["R_frontal_1_minus_cos"] = 1.0 - frontal_cos

    out["R_own_prototype_cosine"] = own_proto
    out["R_rival_prototype_cosine"] = rival_proto
    out["R_expression_margin"] = own_proto - rival_proto

    out["R_retrieval_correct"] = (
        np.array([folders[i] for i in retrieval_idx])
        == out.folder.to_numpy()
    ).astype(int)

    out["R_rival_folder"] = [
        folders[i] for i in rival_idx
    ]

    return out


# ---------------------------------------------------------------------
# MERGE WITH VALIDATED PER-VIEW RESULTS
# ---------------------------------------------------------------------

def merge_with_per_view(rep, per_view):
    required = [
        "folder",
        "expression",
        "viewpoint",
        "A_angular_distance_deg",
        "C_margin",
        "B_predicted_folder",
    ]

    missing = set(required) - set(per_view.columns)
    if missing:
        raise RuntimeError(
            "per_view_metrics_multimetric.csv is missing:\n"
            + ", ".join(sorted(missing))
        )

    cols = required.copy()

    if "angle" in per_view.columns:
        cols.append("angle")

    pv = per_view[cols].copy()

    for c in ["folder", "expression", "B_predicted_folder"]:
        pv[c] = pv[c].map(clean)

    for c in [
        "viewpoint",
        "A_angular_distance_deg",
        "C_margin",
    ]:
        pv[c] = pd.to_numeric(pv[c], errors="coerce")

    if "angle" in pv.columns:
        pv["angle"] = pd.to_numeric(pv["angle"], errors="coerce")

    # If per-view angle is absent/bad, use viewpoint-centered angle.
    if "angle" not in pv.columns:
        pv["angle"] = pv["viewpoint"] - FRONTAL
    else:
        derived = pv["viewpoint"] - FRONTAL
        bad = pv["angle"].isna() | (
            np.abs(pv["angle"] - derived) > 1e-6
        )
        pv.loc[bad, "angle"] = derived.loc[bad]

    pv["viewpoint"] = pv["viewpoint"].astype(int)

    z = rep.merge(
        pv,
        on=["folder", "expression", "viewpoint", "angle"],
        how="inner",
        validate="one_to_one",
    )

    if z.empty:
        # More robust fallback: merge without angle and reconstruct.
        pv2 = pv.drop(columns=["angle"]).copy()

        z = rep.merge(
            pv2,
            on=["folder", "expression", "viewpoint"],
            how="inner",
            validate="one_to_one",
        )

    if z.empty:
        raise RuntimeError(
            "No alignment between embeddings and "
            "per_view_metrics_multimetric.csv."
        )

    z["B_failure_pointwise"] = (
        z["B_predicted_folder"] != z["folder"]
    ).astype(int)

    z["gender"] = z["gender"].astype(str)

    return z


# ---------------------------------------------------------------------
# ORDERING / BOUNDARIES
# ---------------------------------------------------------------------

def side_order(group, side):
    x = group[
        (group.side == side)
        & (group.viewpoint != FRONTAL)
    ].copy()

    x["d"] = x.abs_angle

    return x.sort_values(
        ["d", "viewpoint"]
    ).reset_index(drop=True)


def boundary(group, side, kind):
    x = side_order(group, side)

    if x.empty:
        return np.nan

    if kind == "A":
        mask = x.R_frontal_angular_deg >= A_THRESHOLD

    elif kind == "C":
        mask = x.R_expression_margin <= C_THRESHOLD

    elif kind == "H":
        # Representation identity/expression margin becomes negative.
        mask = x.R_expression_margin < 0

    elif kind == "B":
        mask = x.B_failure_pointwise.astype(bool)

    else:
        raise ValueError(kind)

    i = sustained_first(mask)

    if i is None:
        return np.nan

    return float(x.iloc[i].d)


def compute_boundaries(df):
    rows = []

    for folder, g in df.groupby("folder"):
        row = {
            "folder": folder,
            "expression": clean(g.expression.iloc[0]),
            "gender": clean(g.gender.iloc[0]),
        }

        for side in ["left", "right"]:
            for kind in ["A", "C", "H", "B"]:
                row[f"{kind}_boundary_{side}"] = boundary(
                    g, side, kind
                )

            a = row[f"A_boundary_{side}"]
            c = row[f"C_boundary_{side}"]
            b = row[f"B_boundary_{side}"]

            row[f"A_leads_B_{side}"] = (
                int(a < b)
                if finite(a) and finite(b)
                else np.nan
            )

            row[f"B_minus_A_{side}"] = (
                b - a
                if finite(a) and finite(b)
                else np.nan
            )

            row[f"C_leads_B_{side}"] = (
                int(c < b)
                if finite(c) and finite(b)
                else np.nan
            )

            row[f"B_minus_C_{side}"] = (
                b - c
                if finite(c) and finite(b)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# SUMMARIES
# ---------------------------------------------------------------------

def viewpoint_profile(df):
    x = df[df.viewpoint != FRONTAL].copy()

    if x.empty:
        return pd.DataFrame()

    return (
        x.groupby(
            ["gender", "side", "abs_angle"],
            as_index=False,
        )
        .agg(
            drift_mean=("R_frontal_angular_deg", "mean"),
            drift_median=("R_frontal_angular_deg", "median"),
            margin_mean=("R_expression_margin", "mean"),
            margin_median=("R_expression_margin", "median"),
            representation_accuracy=(
                "R_retrieval_correct",
                "mean",
            ),
            B_accuracy=(
                "B_failure_pointwise",
                lambda z: 1.0 - z.mean(),
            ),
            n_sequences=("folder", "nunique"),
            n_views=("folder", "size"),
        )
        .sort_values(
            ["gender", "side", "abs_angle"]
        )
    )


def sequence_summary(df):
    rows = []

    for folder, g in df.groupby("folder"):
        row = {
            "folder": folder,
            "expression": clean(g.expression.iloc[0]),
            "gender": clean(g.gender.iloc[0]),
        }

        for side in ["left", "right"]:
            x = side_order(g, side)

            if x.empty:
                continue

            row[f"{side}_mean_drift_deg"] = x.R_frontal_angular_deg.mean()
            row[f"{side}_median_drift_deg"] = x.R_frontal_angular_deg.median()
            row[f"{side}_mean_margin"] = x.R_expression_margin.mean()
            row[f"{side}_min_margin"] = x.R_expression_margin.min()
            row[f"{side}_representation_accuracy"] = x.R_retrieval_correct.mean()
            row[f"{side}_B_accuracy"] = 1.0 - x.B_failure_pointwise.mean()

        rows.append(row)

    return pd.DataFrame(rows)


def expression_summary(df):
    rows = []

    for (gender, expression), g in df.groupby(
        ["gender", "expression"]
    ):
        row = {
            "gender": gender,
            "expression": expression,
            "n_sequences": g.folder.nunique(),
        }

        for side in ["left", "right"]:
            x = g[
                (g.side == side)
                & (g.viewpoint != FRONTAL)
            ]

            if x.empty:
                continue

            row[f"{side}_mean_drift_deg"] = x.R_frontal_angular_deg.mean()
            row[f"{side}_mean_margin"] = x.R_expression_margin.mean()
            row[f"{side}_min_margin"] = x.R_expression_margin.min()
            row[f"{side}_representation_accuracy"] = x.R_retrieval_correct.mean()

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PERMUTATION TESTS
# ---------------------------------------------------------------------

def permutation_paired(a, b, reps, seed):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    mask = np.isfinite(a) & np.isfinite(b)

    d = a[mask] - b[mask]

    if len(d) == 0:
        return np.nan, np.nan

    observed = float(d.mean())

    rng = np.random.default_rng(seed)

    null = np.empty(reps)

    for i in range(reps):
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=len(d),
        )
        null[i] = float(np.mean(d * signs))

    p = (
        np.sum(np.abs(null) >= abs(observed)) + 1
    ) / (reps + 1)

    return observed, float(p)


def permutation_groups(a, b, reps, seed):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan

    observed = float(a.mean() - b.mean())

    pool = np.concatenate([a, b])
    n_a = len(a)

    rng = np.random.default_rng(seed)

    null = np.empty(reps)

    for i in range(reps):
        shuffled = rng.permutation(pool)
        null[i] = (
            shuffled[:n_a].mean()
            - shuffled[n_a:].mean()
        )

    p = (
        np.sum(np.abs(null) >= abs(observed)) + 1
    ) / (reps + 1)

    return observed, float(p)


# ---------------------------------------------------------------------
# BOOTSTRAP
# ---------------------------------------------------------------------

def bootstrap_mean(x, reps, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    values = np.empty(reps)

    for i in range(reps):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )
        values[i] = sample.mean()

    return (
        float(x.mean()),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def bootstrap_lead(boundaries, reps):
    rows = []
    rng = np.random.default_rng(SEED + 20)

    for gender in ["Female", "Male"]:
        x = boundaries[boundaries.gender == gender]

        for side in ["left", "right"]:
            a = x[f"A_boundary_{side}"].to_numpy(float)
            b = x[f"B_boundary_{side}"].to_numpy(float)

            mask = np.isfinite(a) & np.isfinite(b)

            if not mask.any():
                continue

            lead = b[mask] - a[mask]

            boot = np.empty(reps)

            for i in range(reps):
                sample = rng.choice(
                    lead,
                    size=len(lead),
                    replace=True,
                )
                boot[i] = sample.mean()

            rows.append(
                {
                    "gender": gender,
                    "side": side,
                    "n": len(lead),
                    "mean_lead": float(lead.mean()),
                    "ci_low": float(np.percentile(boot, 2.5)),
                    "ci_high": float(np.percentile(boot, 97.5)),
                    "A_before_B_rate": float(np.mean(lead > 0)),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# FEMALE / MALE REPRESENTATION TESTS
# ---------------------------------------------------------------------

def identity_tests(df, reps):
    female = (
        df[
            df.gender.str.lower().str.startswith("female")
        ]
        .groupby(["expression", "viewpoint", "side"])
        .agg(
            female_margin=("R_expression_margin", "mean"),
            female_drift=("R_frontal_angular_deg", "mean"),
        )
        .reset_index()
    )

    male = (
        df[
            df.gender.str.lower().str.startswith("male")
        ]
        .groupby(["expression", "viewpoint", "side"])
        .agg(
            male_margin=("R_expression_margin", "mean"),
            male_drift=("R_frontal_angular_deg", "mean"),
        )
        .reset_index()
    )

    paired = female.merge(
        male,
        on=["expression", "viewpoint", "side"],
        how="inner",
    )

    if paired.empty:
        return paired, pd.DataFrame()

    margin_diff, margin_p = permutation_paired(
        paired.female_margin,
        paired.male_margin,
        reps,
        SEED + 1,
    )

    drift_diff, drift_p = permutation_paired(
        paired.female_drift,
        paired.male_drift,
        reps,
        SEED + 2,
    )

    tests = pd.DataFrame(
        [
            {
                "test": "Female_minus_Male_representation_margin",
                "n_paired": len(paired),
                "difference": margin_diff,
                "p_value": margin_p,
            },
            {
                "test": "Female_minus_Male_representation_drift",
                "n_paired": len(paired),
                "difference": drift_diff,
                "p_value": drift_p,
            },
        ]
    )

    return paired, tests


# ---------------------------------------------------------------------
# LEAD / LAG
# ---------------------------------------------------------------------

def lead_summary(boundaries):
    rows = []

    for gender in ["Female", "Male"]:
        x = boundaries[boundaries.gender == gender]

        for side in ["left", "right"]:
            a = x[f"A_boundary_{side}"].to_numpy(float)
            c = x[f"C_boundary_{side}"].to_numpy(float)
            b = x[f"B_boundary_{side}"].to_numpy(float)

            ma = np.isfinite(a) & np.isfinite(b)
            mc = np.isfinite(c) & np.isfinite(b)

            rows.append(
                {
                    "gender": gender,
                    "side": side,
                    "n_A_B": int(ma.sum()),
                    "A_before_B_rate": (
                        float(np.mean(a[ma] < b[ma]))
                        if ma.any()
                        else np.nan
                    ),
                    "mean_B_minus_A_deg": (
                        float(np.mean(b[ma] - a[ma]))
                        if ma.any()
                        else np.nan
                    ),
                    "median_B_minus_A_deg": (
                        float(np.median(b[ma] - a[ma]))
                        if ma.any()
                        else np.nan
                    ),
                    "n_C_B": int(mc.sum()),
                    "C_before_B_rate": (
                        float(np.mean(c[mc] < b[mc]))
                        if mc.any()
                        else np.nan
                    ),
                    "mean_B_minus_C_deg": (
                        float(np.mean(b[mc] - c[mc]))
                        if mc.any()
                        else np.nan
                    ),
                    "median_B_minus_C_deg": (
                        float(np.median(b[mc] - c[mc]))
                        if mc.any()
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# BENJAMINI-HOCHBERG FDR
# ---------------------------------------------------------------------

def benjamini_hochberg(p_values):
    p = np.asarray(p_values, float)

    q = np.full(len(p), np.nan)

    finite_mask = np.isfinite(p)

    if not finite_mask.any():
        return q

    ids = np.where(finite_mask)[0]

    order = np.argsort(p[finite_mask])

    sorted_p = p[finite_mask][order]
    n = len(sorted_p)

    raw = sorted_p * n / np.arange(1, n + 1)

    # Enforce monotonicity from right to left.
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]

    adjusted = np.clip(adjusted, 0.0, 1.0)

    q[ids[order]] = adjusted

    return q


def fdr_tests(df):
    rows = []

    for (gender, expression), g in df.groupby(
        ["gender", "expression"]
    ):
        for side in ["left", "right"]:
            x = g[
                (g.side == side)
                & (g.viewpoint != FRONTAL)
            ]

            near = x[
                x.abs_angle <= PROTO_DEG
            ].R_expression_margin.dropna().to_numpy(float)

            far = x[
                x.abs_angle >= 20
            ].R_expression_margin.dropna().to_numpy(float)

            if len(near) >= 2 and len(far) >= 2:
                effect = float(far.mean() - near.mean())

                _, p = permutation_groups(
                    near,
                    far,
                    5000,
                    SEED + len(rows),
                )
            else:
                effect = np.nan
                p = np.nan

            rows.append(
                {
                    "gender": gender,
                    "expression": expression,
                    "side": side,
                    "n_viewpoints": len(x),
                    "effect_far_minus_near": effect,
                    "p_value": p,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["q_value_fdr_bh"] = benjamini_hochberg(
        result.p_value.to_numpy(float)
    )

    result["fdr_significant_alpha_0_05"] = (
        result.q_value_fdr_bh <= ALPHA
    )

    return result


# ---------------------------------------------------------------------
# JSON-SAFE CONVERSION
# ---------------------------------------------------------------------

def json_safe(obj):
    """
    Converts:
      np.int64 -> int
      np.float64 -> float
      NaN / +inf / -inf -> None
      ndarray -> list

    This prevents:
      ValueError: Out of range float values are not JSON compliant: nan
    """
    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if pd.isna(obj):
        return None

    return obj


def dataframe_records(df):
    if df is None or df.empty:
        return []

    return json_safe(
        df.to_dict(orient="records")
    )


# ---------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------

def plot_profiles(profile):
    if profile.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    for (gender, side), x in profile.groupby(
        ["gender", "side"]
    ):
        ax.plot(
            x.abs_angle,
            x.drift_mean,
            marker="o",
            label=f"{gender}/{side}",
        )

    ax.axhline(
        A_THRESHOLD,
        linestyle="--",
        label="Validated A threshold",
    )

    ax.set_xlabel("Absolute viewpoint angle")
    ax.set_ylabel("Representation drift (deg)")
    ax.set_title("Stage 9: Representation Drift")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUT / "representation_viewpoint_curves.png",
        dpi=220,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))

    for (gender, side), x in profile.groupby(
        ["gender", "side"]
    ):
        ax.plot(
            x.abs_angle,
            x.margin_mean,
            marker="o",
            label=f"{gender}/{side}",
        )

    ax.axhline(
        C_THRESHOLD,
        linestyle="--",
        label="Validated C threshold",
    )

    ax.axhline(0, linestyle=":")

    ax.set_xlabel("Absolute viewpoint angle")
    ax.set_ylabel("Representation margin")
    ax.set_title(
        "Stage 9: Expression Representation Separability"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUT / "representation_margin_curves.png",
        dpi=220,
    )
    plt.close(fig)


def plot_identity(profile):
    if profile.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    for (gender, side), x in profile.groupby(
        ["gender", "side"]
    ):
        ax.plot(
            x.abs_angle,
            x.representation_accuracy,
            marker="o",
            label=f"{gender}/{side}",
        )

    ax.set_xlabel("Absolute viewpoint angle")
    ax.set_ylabel("Representation retrieval accuracy")
    ax.set_title(
        "Stage 9: Representation Retrieval by Identity"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUT / "representation_identity_curves.png",
        dpi=220,
    )
    plt.close(fig)


def plot_same(pair):
    if pair.empty:
        return

    same = pair.same_expression_similarity.dropna()
    rival = pair.different_expression_similarity.dropna()

    if len(same) == 0 or len(rival) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(
        [same, rival],
        labels=[
            "Same expression",
            "Strongest rival",
        ],
    )

    ax.set_ylabel("Cosine similarity")
    ax.set_title(
        "Representation Retention: Same vs Rival Expression"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUT / "representation_same_vs_different.png",
        dpi=220,
    )
    plt.close(fig)


def plot_lead(lead):
    if lead.empty:
        return

    labels = [
        f"{r.gender}/{r.side}"
        for _, r in lead.iterrows()
    ]

    values = lead.mean_B_minus_A_deg.to_numpy(float)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(np.arange(len(values)), values)
    ax.axhline(0, linestyle="--")

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "B boundary - A representation boundary (deg)"
    )
    ax.set_title(
        "Representation Instability Lead/Lag"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUT / "representation_lead_lag.png",
        dpi=220,
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 9 representation-level validation"
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
    )

    args = parser.parse_args()

    if args.bootstrap < 100:
        raise ValueError(
            "--bootstrap must be >= 100"
        )

    if args.permutations < 100:
        raise ValueError(
            "--permutations must be >= 100"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner(
        "STAGE 9 — REPRESENTATION-LEVEL VALIDATION"
    )

    print("Project:", ROOT)
    print("Fixed A:", A_THRESHOLD)
    print("Fixed C:", C_THRESHOLD)
    print("Frontal:", FRONTAL)
    print("Expected viewpoints:", NVIEW)
    print("Sustained rule:", SUSTAINED)
    print("Prototype window:", PROTO_DEG)

    if not PER_VIEW.exists():
        raise FileNotFoundError(
            f"Missing validated per-view file:\n{PER_VIEW}"
        )

    per_view = pd.read_csv(
        PER_VIEW,
        low_memory=False,
    )

    print(
        "Per-view rows:",
        f"{len(per_view):,}",
    )

    # -------------------------------------------------------------
    # 1. EMBEDDINGS
    # -------------------------------------------------------------

    shard_pairs = load_shards()

    metadata, embeddings = load_embeddings(
        shard_pairs
    )

    # -------------------------------------------------------------
    # 2. COMPLETE SEQUENCES
    # -------------------------------------------------------------

    good, incomplete = complete_sequences(
        metadata
    )

    print(
        f"\nComplete sequences: {len(good):,}"
    )

    print(
        f"Incomplete sequences excluded: "
        f"{len(incomplete):,}"
    )

    # -------------------------------------------------------------
    # 3. REFERENCES
    # -------------------------------------------------------------

    folders, folder_index, frontal, prototypes = (
        build_references(
            metadata,
            embeddings,
            good,
        )
    )

    # -------------------------------------------------------------
    # 4. REPRESENTATION METRICS
    # -------------------------------------------------------------

    representation = representation_metrics(
        metadata,
        embeddings,
        good,
        folders,
        folder_index,
        frontal,
        prototypes,
    )

    # -------------------------------------------------------------
    # 5. ALIGN WITH VALIDATED A/B/C METRICS
    # -------------------------------------------------------------

    df = merge_with_per_view(
        representation,
        per_view,
    )

    print(
        f"Aligned representation/per-view rows: "
        f"{len(df):,}"
    )

    # -------------------------------------------------------------
    # 6. BOUNDARIES
    # -------------------------------------------------------------

    boundaries = compute_boundaries(df)

    # -------------------------------------------------------------
    # 7. SUMMARIES
    # -------------------------------------------------------------

    seq_summary = sequence_summary(df)
    expr_summary = expression_summary(df)
    profile = viewpoint_profile(df)

    # -------------------------------------------------------------
    # 8. SAME EXPRESSION VS RIVAL
    # -------------------------------------------------------------

    pair = df[
        [
            "folder",
            "gender",
            "expression",
            "viewpoint",
            "angle",
            "side",
            "R_own_prototype_cosine",
            "R_rival_prototype_cosine",
            "R_expression_margin",
        ]
    ].rename(
        columns={
            "R_own_prototype_cosine":
                "same_expression_similarity",
            "R_rival_prototype_cosine":
                "different_expression_similarity",
        }
    )

    pair["representation_margin"] = (
        pair.same_expression_similarity
        - pair.different_expression_similarity
    )

    same = pair.same_expression_similarity.to_numpy(float)
    rival = pair.different_expression_similarity.to_numpy(float)

    same_diff, same_p = permutation_groups(
        same,
        rival,
        args.permutations,
        SEED + 30,
    )

    pair_test = pd.DataFrame(
        [
            {
                "test":
                    "same_vs_different_expression_similarity",
                "n":
                    int(
                        np.isfinite(same).sum()
                        + np.isfinite(rival).sum()
                    ),
                "same_mean":
                    float(np.nanmean(same)),
                "different_mean":
                    float(np.nanmean(rival)),
                "difference":
                    same_diff,
                "p_value":
                    same_p,
            }
        ]
    )

    # -------------------------------------------------------------
    # 9. FEMALE / MALE REPRESENTATION COMPARISON
    # -------------------------------------------------------------

    paired_identity, identity_tests_df = identity_tests(
        df,
        args.permutations,
    )

    # -------------------------------------------------------------
    # 10. LEAD / LAG
    # -------------------------------------------------------------

    lead = lead_summary(boundaries)

    bootstrap = bootstrap_lead(
        boundaries,
        args.bootstrap,
    )

    # -------------------------------------------------------------
    # 11. EXPRESSION FDR
    # -------------------------------------------------------------

    fdr = fdr_tests(df)

    # -------------------------------------------------------------
    # 12. SAVE CSVs
    # -------------------------------------------------------------

    outputs = {
        "representation_view_metrics.csv": df,
        "representation_boundaries.csv": boundaries,
        "representation_sequence_summary.csv": seq_summary,
        "representation_expression_summary.csv": expr_summary,
        "representation_viewpoint_profile.csv": profile,
        "representation_pairwise_summary.csv": pair,
        "representation_pairwise_test.csv": pair_test,
        "representation_paired_identity.csv": paired_identity,
        "representation_identity_tests.csv": identity_tests_df,
        "representation_lead_lag.csv": lead,
        "representation_bootstrap.csv": bootstrap,
        "representation_fdr.csv": fdr,
        "representation_incomplete_sequences.csv": incomplete,
    }

    banner("SAVING RESULTS")

    for filename, table in outputs.items():
        path = OUT / filename

        table.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(path)

    # -------------------------------------------------------------
    # 13. PLOTS
    # -------------------------------------------------------------

    if not args.no_plots:
        banner("BUILDING PLOTS")

        plot_profiles(profile)
        plot_identity(profile)
        plot_same(pair)
        plot_lead(lead)

        for filename in [
            "representation_viewpoint_curves.png",
            "representation_margin_curves.png",
            "representation_identity_curves.png",
            "representation_same_vs_different.png",
            "representation_lead_lag.png",
        ]:
            path = OUT / filename
            if path.exists():
                print(path)

    # -------------------------------------------------------------
    # 14. JSON REPORT
    # -------------------------------------------------------------

    report = {
        "stage": 9,
        "project": "FER Reliability Benchmark",
        "question":
            "Does viewpoint-induced representation "
            "instability precede validated prediction failure?",

        "fixed_configuration": {
            "A_threshold": A_THRESHOLD,
            "C_threshold": C_THRESHOLD,
            "sustained_viewpoints": SUSTAINED,
            "frontal_viewpoint": FRONTAL,
            "expected_viewpoints": NVIEW,
            "prototype_window_deg": PROTO_DEG,
            "embedding_dimension": EMBED_DIM,
        },

        "embedding_storage": {
            "directory": str(EMB_DIR),
            "shards": len(shard_pairs),
            "metadata_naming":
                "metadata_00000.csv ... metadata_00009.csv",
            "embedding_naming":
                "embeddings_00000.npy ... embeddings_00009.npy",
        },

        "coverage": {
            "embedding_rows": int(len(embeddings)),
            "complete_sequences": int(len(good)),
            "incomplete_sequences": int(len(incomplete)),
            "aligned_rows": int(len(df)),
            "expressions":
                int(
                    df.expression.nunique()
                ),
        },

        "lead_lag": dataframe_records(lead),
        "bootstrap": dataframe_records(bootstrap),
        "same_vs_different": dataframe_records(pair_test),
        "identity_tests": dataframe_records(identity_tests_df),
        "fdr": dataframe_records(fdr),

        "scientific_caution":
            "Positive B_boundary - representation_boundary "
            "means temporal/statistical precedence, not causality. "
            "A/C thresholds are fixed and are not re-estimated in Stage 9.",
    }

    report = json_safe(report)

    report_path = OUT / "representation_report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(report_path)

    # -------------------------------------------------------------
    # 15. README
    # -------------------------------------------------------------

    readme = f"""# Stage 9 — Representation-Level Validation

## Purpose

This stage tests whether viewpoint-induced representation instability
appears before the already-validated prediction-failure boundary.

## Fixed configuration

- A threshold: {A_THRESHOLD:.8f}
- C threshold: {C_THRESHOLD:.8f}
- Sustained viewpoints: {SUSTAINED}
- Frontal viewpoint: V{FRONTAL}
- Expected viewpoints per complete sequence: {NVIEW}
- Near-frontal prototype window: ±{PROTO_DEG} degrees
- Embedding dimension: {EMBED_DIM}

## Viewpoint convention

The benchmark uses V107 as frontal.

Signed angle is reconstructed deterministically as:

    angle = viewpoint - 107

Therefore:

- V000 = -107 degrees
- V106 = -1 degree
- V107 = 0 degrees
- V108 = +1 degree
- V214 = +107 degrees

Left/right traversal is outward from V107.

## Representation measures

`R_frontal_angular_deg`
: angular distance between the current representation and the exact
  frontal V107 representation of the same sequence.

`R_expression_margin`
: similarity to the sequence's own near-frontal expression prototype
  minus similarity to the strongest rival expression prototype.

`R_retrieval_correct`
: whether the own-expression prototype is the nearest prototype.

## Boundary interpretation

A positive:

    B_boundary - representation_boundary

means representation instability appears earlier than the validated
prediction-failure boundary.

This is temporal/statistical precedence, not causal evidence.

## Important implementation detail

Embedding shards are paired by their exact numeric suffix:

    embeddings_00000.npy <-> metadata_00000.csv
    embeddings_00001.npy <-> metadata_00001.csv
    ...
    embeddings_00009.npy <-> metadata_00009.csv

The script does not assume `metadata_0.csv`.

Invalid/legacy viewpoint strings such as `V000` are converted to integer
viewpoints automatically. Angle is reconstructed from the benchmark's
V107-centered convention.

JSON NaN and infinite values are converted to JSON `null`, so the report
is always valid JSON.
"""

    (OUT / "README_representation_validation.md").write_text(
        readme,
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------

    banner("FINAL STAGE 9 SUMMARY")

    print(
        f"Embedding rows: {len(embeddings):,}"
    )

    print(
        f"Complete sequences: {len(good):,}"
    )

    print(
        f"Incomplete excluded: {len(incomplete):,}"
    )

    print(
        f"Expressions: {df.expression.nunique():,}"
    )

    print(
        f"Aligned rows: {len(df):,}"
    )

    if not lead.empty:
        print("\nLEAD / LAG")
        print(
            lead.to_string(index=False)
        )

    if not pair_test.empty:
        row = pair_test.iloc[0]
        print(
            "\nSame-vs-rival difference="
            f"{row['difference']:.6f}; "
            "permutation p="
            f"{row['p_value']:.6f}"
        )

    print("\nDONE")
    print(report_path)


if __name__ == "__main__":
    main()
