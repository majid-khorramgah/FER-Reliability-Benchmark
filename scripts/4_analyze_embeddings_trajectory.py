"""
analyze_embeddings_trajectory.py

Multi-metric representation-trajectory analysis for the FER Reliability Benchmark.

A metrics:
  1) cosine distance to V107
  2) angular distance to V107
  3) Euclidean distance on L2-normalized embeddings
  4) cumulative path length from V107
  5) local representation-change rate per viewpoint degree
  6) discrete curvature
  7) second-difference trajectory instability

B:
  prototype-retrieval expression consistency (no six-class classifier)

C:
  own-prototype margin against strongest rival.

Only complete 215-view folders are analyzed.
Expected embedding model: DINOv2 ViT-B/14, 768 dimensions.

Run:
    python analyze/analyze_embeddings_trajectory.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_VIEWPOINTS = 215
MIN_VIEWPOINT = 0
MAX_VIEWPOINT = 214
FRONTAL_VIEWPOINT = 107

BASELINE_DEG = 5
PROTOTYPE_WINDOW_DEG = 5
SUSTAINED = 3
A_QUANTILE = 0.95
C_QUANTILE = 0.05
BATCH_SIZE = 4096
DEFAULT_BOOTSTRAP = 500
EPS = 1e-12

REQUIRED = {
    "folder", "gender", "expression",
    "viewpoint", "angle", "image_path"
}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[0]


def natural_key(path: Path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else -1


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, EPS)


def normalize_vector(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), EPS)


def angular_deg(c):
    return np.degrees(
        np.arccos(np.clip(c, -1.0, 1.0))
    ).astype(np.float32)


def trapz_mean(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 2:
        return np.nan
    order = np.argsort(x)
    y, x = y[order], x[order]
    width = x[-1] - x[0]
    if width <= 0:
        return np.nan
    area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return float(area / width)


def sustained_first(mask, run=SUSTAINED):
    mask = np.asarray(mask, dtype=bool)
    count = 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        if count >= run:
            return i - run + 1
    return None


# ---------------------------------------------------------------------------
# Metadata / shards
# ---------------------------------------------------------------------------

def find_embedding_dir(root):
    for p in (root / "data" / "embeddings", root / "embeddings"):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Embeddings directory not found: {root / 'data' / 'embeddings'}"
    )


def normalize_viewpoints(s):
    raw = s.astype(str).str.strip()
    s = pd.to_numeric(raw.str.replace(r"^[Vv]", "", regex=True), errors="coerce")
    if s.isna().any():
        raise RuntimeError(
            "Invalid viewpoint values. Expected 0..214 or V000..V214."
        )
    if not np.allclose(s.to_numpy(), np.round(s.to_numpy())):
        raise RuntimeError("Viewpoint values must be integers.")
    s = s.astype(int)
    bad = s[(s < MIN_VIEWPOINT) | (s > MAX_VIEWPOINT)]
    if len(bad):
        raise RuntimeError(f"Viewpoint outside 0..214: {bad.unique()[:20]}")
    return s


def load_metadata(root):
    emb_dir = find_embedding_dir(root)
    paths = sorted(emb_dir.glob("metadata_*.csv"), key=natural_key)

    if paths:
        frames = []
        for p in paths:
            df = pd.read_csv(p)
            missing = REQUIRED - set(df.columns)
            if missing:
                raise RuntimeError(f"{p.name} missing columns: {sorted(missing)}")
            frames.append(df)
        meta = pd.concat(frames, ignore_index=True)
    else:
        p = root / "data" / "metadata.csv"
        if not p.exists():
            raise FileNotFoundError("No metadata shards or data/metadata.csv.")
        meta = pd.read_csv(p)

    meta["viewpoint"] = normalize_viewpoints(meta["viewpoint"])

    raw = meta["angle"].astype(str).str.strip()
    raw = (
        raw.str.replace("°", "", regex=False)
           .str.replace("degrees", "", regex=False, case=False)
           .str.replace("degree", "", regex=False, case=False)
           .str.strip()
    )
    meta["angle"] = pd.to_numeric(raw, errors="coerce")
    if meta["angle"].isna().any():
        raise RuntimeError("Invalid angle values in metadata.")

    meta["angle"] = meta["angle"].astype(float)
    return meta


def load_shards(emb_dir):
    files = sorted(emb_dir.glob("embeddings_*.npy"), key=natural_key)
    pairs = []
    for emb in files:
        m = re.search(r"(\d+)$", emb.stem)
        if not m:
            continue
        meta = emb_dir / f"metadata_{m.group(1)}.csv"
        if not meta.exists():
            raise FileNotFoundError(f"Missing {meta.name}")
        pairs.append((emb, meta))
    if not pairs:
        raise FileNotFoundError(f"No embeddings_*.npy in {emb_dir}")
    return pairs


def validate_shards(shards, global_meta):
    total = 0
    for emb_path, meta_path in shards:
        emb = np.load(emb_path, mmap_mode="r")
        meta = pd.read_csv(meta_path)
        if len(emb) != len(meta):
            raise RuntimeError(
                f"{emb_path.name}: {len(emb)} embeddings vs "
                f"{len(meta)} metadata rows."
            )
        if emb.ndim != 2 or emb.shape[1] != 768:
            raise RuntimeError(
                f"{emb_path.name} has shape {emb.shape}; expected (N, 768)."
            )
        total += len(emb)
    if total != len(global_meta):
        raise RuntimeError(
            f"Embedding rows {total:,} != metadata rows {len(global_meta):,}"
        )
    print(f"Validated {len(shards)} embedding shards.")
    print(f"Total embedding rows: {total:,}")


def complete_folders(meta):
    counts = meta.groupby("folder")["viewpoint"].nunique().sort_index()
    complete = counts[counts == EXPECTED_VIEWPOINTS].index.tolist()
    incomplete = counts[counts != EXPECTED_VIEWPOINTS].reset_index()
    incomplete.columns = ["folder", "viewpoint_count"]

    print("\nSEQUENCE COVERAGE")
    print("-" * 78)
    print(f"Total folders:              {len(counts):,}")
    print(f"Complete 215-view folders:  {len(complete):,}")
    print(f"Excluded incomplete:        {len(incomplete):,}")

    for _, r in incomplete.iterrows():
        print(f"  {r.folder}: {int(r.viewpoint_count)} viewpoints")

    return complete, incomplete


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def build_references(shards, complete):
    frontal = {}
    proto_sum = {}
    proto_count = {}

    for emb_path, meta_path in shards:
        emb = np.load(emb_path).astype(np.float32)
        meta = pd.read_csv(meta_path)
        meta["viewpoint"] = normalize_viewpoints(meta["viewpoint"])
        meta["angle"] = pd.to_numeric(meta["angle"], errors="coerce")

        for i in np.flatnonzero(meta["folder"].isin(complete).to_numpy()):
            r = meta.iloc[i]
            f = r["folder"]
            vp = int(r["viewpoint"])
            a = float(r["angle"])
            v = normalize_vector(emb[i])

            if vp == FRONTAL_VIEWPOINT:
                if f in frontal:
                    raise RuntimeError(f"Duplicate V107: {f}")
                frontal[f] = v.copy()

            if vp != FRONTAL_VIEWPOINT and 1 <= abs(a) <= PROTOTYPE_WINDOW_DEG:
                proto_sum.setdefault(f, np.zeros(emb.shape[1], dtype=np.float32))
                proto_count[f] = proto_count.get(f, 0) + 1
                proto_sum[f] += v

    for f in complete:
        if f not in frontal:
            raise RuntimeError(f"Missing V107: {f}")
        if proto_count.get(f, 0) == 0:
            raise RuntimeError(f"Cannot build prototype: {f}")

    folders = sorted(complete)
    frontal_matrix = np.stack([frontal[f] for f in folders]).astype(np.float32)
    prototype_matrix = np.stack([
        normalize_vector(proto_sum[f] / proto_count[f]) for f in folders
    ]).astype(np.float32)

    return (
        folders,
        {f: i for i, f in enumerate(folders)},
        frontal_matrix,
        prototype_matrix,
    )


# ---------------------------------------------------------------------------
# Base A/B/C
# ---------------------------------------------------------------------------

def analyze_base(shards, complete, folder_index, frontal, prototypes, folders):
    records = []
    prototypes = normalize_rows(prototypes)

    total = sum(len(np.load(p, mmap_mode="r")) for p, _ in shards)
    processed = 0
    t0 = time.time()

    for emb_path, meta_path in shards:
        emb = np.load(emb_path).astype(np.float32)
        meta = pd.read_csv(meta_path)
        meta["viewpoint"] = normalize_viewpoints(meta["viewpoint"])
        meta["angle"] = pd.to_numeric(meta["angle"], errors="coerce")

        ids = np.flatnonzero(meta["folder"].isin(complete).to_numpy())

        for s in range(0, len(ids), BATCH_SIZE):
            idx = ids[s:s+BATCH_SIZE]
            z = normalize_rows(emb[idx])

            own = np.array(
                [folder_index[meta.iloc[i]["folder"]] for i in idx],
                dtype=np.int64,
            )
            ref = frontal[own]

            cos = np.clip(np.sum(z * ref, axis=1), -1.0, 1.0)
            cos_dist = 1.0 - cos
            ang_dist = angular_deg(cos)
            euclid = np.linalg.norm(z - ref, axis=1)

            sim = z @ prototypes.T
            best = np.argmax(sim, axis=1)
            best_sim = sim[np.arange(len(idx)), best]

            own_sim = sim[np.arange(len(idx)), own]
            rivals = sim.copy()
            rivals[np.arange(len(idx)), own] = -np.inf
            rival = np.argmax(rivals, axis=1)
            rival_sim = rivals[np.arange(len(idx)), rival]
            margin = own_sim - rival_sim

            for j, row_i in enumerate(idx):
                r = meta.iloc[row_i]
                pred = folders[int(best[j])]
                records.append({
                    "folder": r["folder"],
                    "gender": r["gender"],
                    "expression": r["expression"],
                    "viewpoint": int(r["viewpoint"]),
                    "angle": float(r["angle"]),
                    "image_path": r["image_path"],

                    "A_cosine_distance": float(cos_dist[j]),
                    "A_angular_distance_deg": float(ang_dist[j]),
                    "A_euclidean_distance": float(euclid[j]),

                    "B_predicted_folder": pred,
                    "B_correct": int(pred == r["folder"]),
                    "B_top1_similarity": float(best_sim[j]),

                    "C_rival_folder": folders[int(rival[j])],
                    "C_own_similarity": float(own_sim[j]),
                    "C_rival_similarity": float(rival_sim[j]),
                    "C_margin": float(margin[j]),
                })

            processed += len(idx)

        print(
            f"Processed {processed:,}/{total:,} "
            f"({processed / max(time.time()-t0, 1e-9):,.0f} rows/s)"
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Representation trajectory geometry
# ---------------------------------------------------------------------------

def add_trajectory_metrics(per_view, shards, complete):
    """
    For each complete expression sequence:

      step = ||z_i - z_(i-1)||

      cumulative path from V107
          = accumulated step length outward from the frontal point

      rate
          = step / viewpoint-degree

      turning angle
          = angle between two consecutive displacement vectors

      curvature
          = turning angle / local arc length

      instability
          = ||z_(i+1) - 2z_i + z_(i-1)||

    Embeddings are normalized before these calculations.
    """

    print("\nBUILDING REPRESENTATION TRAJECTORY METRICS")
    print("-" * 78)

    seq = {f: {} for f in complete}

    for emb_path, meta_path in shards:
        emb = np.load(emb_path, mmap_mode="r")
        meta = pd.read_csv(meta_path)
        meta["viewpoint"] = normalize_viewpoints(meta["viewpoint"])

        ids = np.flatnonzero(meta["folder"].isin(complete).to_numpy())

        for i in ids:
            r = meta.iloc[i]
            seq[r["folder"]][int(r["viewpoint"])] = normalize_vector(emb[i])

    rows = []

    for folder in sorted(complete):
        if len(seq[folder]) != EXPECTED_VIEWPOINTS:
            raise RuntimeError(
                f"{folder}: expected 215 viewpoints, got {len(seq[folder])}"
            )

        vps = sorted(seq[folder])
        z = np.stack([seq[folder][v] for v in vps]).astype(np.float32)

        # 0..214 -> -107..107
        angles = np.arange(215, dtype=float) - FRONTAL_VIEWPOINT

        delta = z[1:] - z[:-1]
        step = np.linalg.norm(delta, axis=1)
        rate = step  # 1 viewpoint = 1 degree

        # Cumulative path from V107, separately on each side.
        cum = np.zeros(215, dtype=float)

        # right side
        for i in range(FRONTAL_VIEWPOINT + 1, 215):
            cum[i] = cum[i-1] + step[i-1]

        # left side
        for i in range(FRONTAL_VIEWPOINT - 1, -1, -1):
            cum[i] = cum[i+1] + step[i]

        turning = np.full(215, np.nan)
        curvature = np.full(215, np.nan)
        instability = np.full(215, np.nan)

        for i in range(1, 214):
            d1 = z[i] - z[i-1]
            d2 = z[i+1] - z[i]
            n1 = float(np.linalg.norm(d1))
            n2 = float(np.linalg.norm(d2))

            if n1 > EPS and n2 > EPS:
                c = np.clip(float(np.dot(d1, d2) / (n1*n2)), -1.0, 1.0)
                theta = float(np.arccos(c))
                turning[i] = theta
                curvature[i] = theta / max(0.5*(n1+n2), EPS)

            instability[i] = float(
                np.linalg.norm(z[i+1] - 2*z[i] + z[i-1])
            )

        for i, vp in enumerate(vps):
            rows.append({
                "folder": folder,
                "viewpoint": int(vp),
                "angle": float(angles[i]),

                "A_cumulative_path_from_V107": float(cum[i]),

                "A_rate_per_degree": (
                    float(rate[i-1]) if i > 0 else np.nan
                ),

                "A_turning_angle_rad": (
                    float(turning[i]) if np.isfinite(turning[i]) else np.nan
                ),

                "A_curvature": (
                    float(curvature[i]) if np.isfinite(curvature[i]) else np.nan
                ),

                "A_trajectory_instability": (
                    float(instability[i])
                    if np.isfinite(instability[i]) else np.nan
                ),
            })

    traj = pd.DataFrame(rows)

    return per_view.merge(
        traj,
        on=["folder", "viewpoint", "angle"],
        how="left",
        validate="one_to_one",
    )


# ---------------------------------------------------------------------------
# Thresholds and boundaries
# ---------------------------------------------------------------------------

def estimate_thresholds(df):
    b = df[df["angle"].abs() <= BASELINE_DEG]

    q = {
        "A_cosine_threshold": "A_cosine_distance",
        "A_angular_threshold": "A_angular_distance_deg",
        "A_euclidean_threshold": "A_euclidean_distance",
        "A_path_threshold": "A_cumulative_path_from_V107",
        "A_rate_threshold": "A_rate_per_degree",
        "A_curvature_threshold": "A_curvature",
        "A_instability_threshold": "A_trajectory_instability",
    }

    out = {}
    for name, col in q.items():
        values = pd.to_numeric(b[col], errors="coerce").dropna()
        out[name] = float(np.quantile(values, A_QUANTILE))

    out["C_margin_threshold"] = float(
        np.quantile(b["C_margin"].dropna(), C_QUANTILE)
    )
    return out


def first_boundary(group, col, threshold, greater=True):
    angles = group["angle"].to_numpy()
    values = group[col].to_numpy()

    mask = values > threshold if greater else values < threshold

    neg = np.where((angles <= 0) & (angles >= -107))[0]
    neg = neg[np.argsort(angles[neg])[::-1]]

    pos = np.where((angles >= 0) & (angles <= 107))[0]
    pos = pos[np.argsort(angles[pos])]

    result = []

    for ids in (neg, pos):
        idx = sustained_first(mask[ids])
        if idx is None:
            result.append((np.nan, np.nan))
        else:
            a = float(angles[ids[idx]])
            result.append((a, abs(a)))

    return result[0], result[1]


def build_boundaries(df, thresholds):
    specs = [
        ("A_cosine", "A_cosine_distance", thresholds["A_cosine_threshold"], True),
        ("A_angular", "A_angular_distance_deg", thresholds["A_angular_threshold"], True),
        ("A_euclidean", "A_euclidean_distance", thresholds["A_euclidean_threshold"], True),
        ("A_path", "A_cumulative_path_from_V107", thresholds["A_path_threshold"], True),
        ("A_rate", "A_rate_per_degree", thresholds["A_rate_threshold"], True),
        ("A_curvature", "A_curvature", thresholds["A_curvature_threshold"], True),
        ("A_instability", "A_trajectory_instability", thresholds["A_instability_threshold"], True),
        ("C", "C_margin", thresholds["C_margin_threshold"], False),
    ]

    rows = []

    for folder, g in df.groupby("folder", sort=True):
        g = g.sort_values("angle").reset_index(drop=True)
        row = {"folder": folder}

        for name, col, threshold, greater in specs:
            left, right = first_boundary(
                g, col, threshold, greater
            )
            row[f"{name}_left_angle"] = left[0]
            row[f"{name}_left_abs_angle"] = left[1]
            row[f"{name}_right_angle"] = right[0]
            row[f"{name}_right_abs_angle"] = right[1]

        # B: first sustained wrong top-1 retrieval.
        left, right = first_boundary(
            g, "B_correct", 0.5, greater=False
        )
        row["B_left_angle"] = left[0]
        row["B_left_abs_angle"] = left[1]
        row["B_right_angle"] = right[0]
        row["B_right_abs_angle"] = right[1]

        # Hard C boundary: own prototype loses to a rival.
        left, right = first_boundary(
            g, "C_margin", 0.0, greater=False
        )
        row["C_hard_left_angle"] = left[0]
        row["C_hard_left_abs_angle"] = left[1]
        row["C_hard_right_angle"] = right[0]
        row["C_hard_right_abs_angle"] = right[1]

        # Integrated quantities.
        for side, mask in [
            ("left", g["angle"] <= 0),
            ("right", g["angle"] >= 0),
        ]:
            s = g[mask].sort_values("angle")
            for col in [
                "A_cosine_distance",
                "A_angular_distance_deg",
                "A_euclidean_distance",
                "A_cumulative_path_from_V107",
                "A_rate_per_degree",
                "A_curvature",
                "A_trajectory_instability",
                "C_margin",
                "B_correct",
            ]:
                row[f"{col}_AUC_{side}"] = trapz_mean(
                    s[col].to_numpy(),
                    s["angle"].to_numpy(),
                )

        # Lead/lag relative to B.
        for side in ("left", "right"):
            b = row[f"B_{side}_abs_angle"]
            for metric in (
                "A_cosine",
                "A_angular",
                "A_euclidean",
                "A_path",
                "A_rate",
                "A_curvature",
                "A_instability",
                "C",
            ):
                a = row[f"{metric}_{side}_abs_angle"]
                row[f"{metric}_leads_B_{side}"] = (
                    int(a < b) if np.isfinite(a) and np.isfinite(b) else np.nan
                )
                row[f"B_minus_{metric}_{side}"] = (
                    b-a if np.isfinite(a) and np.isfinite(b) else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Population / bootstrap
# ---------------------------------------------------------------------------

def population_profile(df):
    x = df.copy()
    x["abs_angle"] = x["angle"].abs()
    return (
        x.groupby("abs_angle")
        .agg(
            cosine=("A_cosine_distance", "mean"),
            angular=("A_angular_distance_deg", "mean"),
            euclidean=("A_euclidean_distance", "mean"),
            path=("A_cumulative_path_from_V107", "mean"),
            rate=("A_rate_per_degree", "mean"),
            curvature=("A_curvature", "mean"),
            instability=("A_trajectory_instability", "mean"),
            B_accuracy=("B_correct", "mean"),
            C_margin=("C_margin", "mean"),
            N=("folder", "nunique"),
        )
        .reset_index()
        .sort_values("abs_angle")
    )


def directional_profile(df):
    x = df.copy()
    x["direction"] = np.where(
        x["angle"] < 0, "left",
        np.where(x["angle"] > 0, "right", "frontal")
    )
    x["abs_angle"] = x["angle"].abs()
    return (
        x.groupby(["direction", "abs_angle"])
        .agg(
            cosine=("A_cosine_distance", "mean"),
            angular=("A_angular_distance_deg", "mean"),
            euclidean=("A_euclidean_distance", "mean"),
            path=("A_cumulative_path_from_V107", "mean"),
            rate=("A_rate_per_degree", "mean"),
            curvature=("A_curvature", "mean"),
            instability=("A_trajectory_instability", "mean"),
            B_accuracy=("B_correct", "mean"),
            C_margin=("C_margin", "mean"),
            N=("folder", "nunique"),
        )
        .reset_index()
        .sort_values(["direction", "abs_angle"])
    )


def bootstrap_boundaries(summary, n, seed):
    rng = np.random.default_rng(seed)
    rows = []

    cols = [
        c for c in summary.columns
        if c.endswith("_left_abs_angle")
        or c.endswith("_right_abs_angle")
    ]

    for col in cols:
        values = pd.to_numeric(
            summary[col], errors="coerce"
        ).dropna().to_numpy()

        if len(values) == 0:
            continue

        med = np.empty(n, dtype=np.float32)

        for i in range(n):
            sample = rng.choice(values, size=len(values), replace=True)
            med[i] = np.median(sample)

        rows.append({
            "metric": col,
            "n_sequences": len(values),
            "observed_median": float(np.median(values)),
            "ci_low_95": float(np.percentile(med, 2.5)),
            "ci_high_95": float(np.percentile(med, 97.5)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plots(profile, out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; plots skipped.")
        return

    x = profile["abs_angle"]

    plt.figure(figsize=(10, 6))
    for col, label in [
        ("cosine", "Cosine distance"),
        ("angular", "Angular distance"),
        ("euclidean", "Euclidean distance"),
    ]:
        plt.plot(x, profile[col], label=label)
    plt.xlabel("Absolute viewpoint angle from frontal (degrees)")
    plt.ylabel("Representation change")
    plt.title("A — Reference-based representation metrics")
    plt.grid(True, alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "A_reference_metrics_multimetric.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for col, label in [
        ("path", "Cumulative path"),
        ("rate", "Change rate"),
        ("curvature", "Curvature"),
        ("instability", "Trajectory instability"),
    ]:
        plt.plot(x, profile[col], label=label)
    plt.xlabel("Absolute viewpoint angle from frontal (degrees)")
    plt.ylabel("Trajectory quantity")
    plt.title("A — Representation trajectory geometry")
    plt.grid(True, alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "A_trajectory_geometry.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, profile["B_accuracy"], linewidth=2)
    plt.ylim(-.02, 1.02)
    plt.xlabel("Absolute viewpoint angle from frontal (degrees)")
    plt.ylabel("Top-1 expression consistency")
    plt.title("B — Expression consistency")
    plt.grid(True, alpha=.25)
    plt.tight_layout()
    plt.savefig(out / "B_expression_consistency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, profile["C_margin"], linewidth=2)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Absolute viewpoint angle from frontal (degrees)")
    plt.ylabel("Own prototype − strongest rival")
    plt.title("C — Expression separability")
    plt.grid(True, alpha=.25)
    plt.tight_layout()
    plt.savefig(out / "C_expression_separability.png", dpi=180)
    plt.close()

    print("Plots saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    root = project_root()
    emb_dir = find_embedding_dir(root)
    out = root / "analysis" / "4_analyze_embeddings_trajectory"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FER REPRESENTATION RELIABILITY BENCHMARK")
    print("MULTI-METRIC REPRESENTATION TRAJECTORY ANALYSIS")
    print("=" * 78)
    print(f"Project root:       {root}")
    print(f"Embedding directory:{emb_dir}")
    print(f"Output directory:   {out}")

    meta = load_metadata(root)
    print(f"\nMetadata rows: {len(meta):,}")

    shards = load_shards(emb_dir)
    validate_shards(shards, meta)

    complete, incomplete = complete_folders(meta)
    complete_set = set(complete)

    if len(complete) < 2:
        raise RuntimeError("Need at least two complete sequences.")

    print("\nBUILDING REFERENCES")
    print("-" * 78)

    folders, folder_index, frontal, prototypes = build_references(
        shards, complete_set
    )

    print(f"Complete expression sequences: {len(folders):,}")
    print(f"Frontal reference matrix:       {frontal.shape}")
    print(f"B/C prototype matrix:            {prototypes.shape}")

    print("\nANALYZING BASE A / B / C")
    print("-" * 78)

    df = analyze_base(
        shards,
        complete_set,
        folder_index,
        frontal,
        prototypes,
        folders,
    )

    df = add_trajectory_metrics(df, shards, complete_set)

    thresholds = estimate_thresholds(df)

    print("\nROBUST THRESHOLDS")
    print("-" * 78)
    for k, v in thresholds.items():
        print(f"{k:<34}: {v:.8f}")

    summary = build_boundaries(df, thresholds)

    info = (
        meta[meta["folder"].isin(complete_set)]
        .groupby("folder")
        .first()
        .reset_index()[["folder", "gender", "expression"]]
    )
    summary = info.merge(summary, on="folder", how="right")

    profile = population_profile(df)
    directional = directional_profile(df)
    boot = bootstrap_boundaries(
        summary, args.bootstrap, args.seed
    )

    # Save CSVs.
    paths = {
        "per_view": out / "per_view_metrics_multimetric.csv",
        "summary": out / "per_expression_summary_multimetric.csv",
        "profile": out / "population_profile_multimetric.csv",
        "directional": out / "directional_summary_multimetric.csv",
        "bootstrap": out / "bootstrap_boundaries_multimetric.csv",
    }

    df.to_csv(paths["per_view"], index=False, encoding="utf-8")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8")
    profile.to_csv(paths["profile"], index=False, encoding="utf-8")
    directional.to_csv(paths["directional"], index=False, encoding="utf-8")
    boot.to_csv(paths["bootstrap"], index=False, encoding="utf-8")

    report = {
        "model": "DINOv2 ViT-B/14",
        "dimension": 768,
        "complete_sequences": len(complete),
        "images_analyzed": len(df),
        "baseline_degrees": BASELINE_DEG,
        "sustained_viewpoints": SUSTAINED,
        "thresholds": thresholds,
        "metrics": [
            "cosine_distance",
            "angular_distance",
            "euclidean_distance",
            "cumulative_path_length",
            "rate_of_representation_change",
            "curvature",
            "trajectory_instability",
            "expression_consistency_B",
            "expression_separability_C",
        ],
        "interpretation": (
            "The mathematical operations are standard. The robustness "
            "question is whether the representation-before-prediction-"
            "failure phenomenon survives across different geometric "
            "descriptions of the same controlled viewpoint trajectory."
        ),
    }

    (out / "trajectory_analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""# Multi-Metric Trajectory Analysis

This run evaluates the same controlled viewpoint experiment with several
representation geometries.

## A metrics

- Cosine distance
- Angular distance
- Euclidean distance on normalized embeddings
- Cumulative path length from V107
- Local representation-change rate
- Discrete curvature
- Second-difference trajectory instability

The default A threshold for each metric is its 95th percentile in the
near-frontal |angle| <= {BASELINE_DEG} degree baseline.

A boundary requires {SUSTAINED} consecutive viewpoints above threshold.

## B

B is prototype retrieval consistency, not a six-class emotion classifier.

## C

C is:

    own-prototype similarity - strongest rival similarity

## Scientific purpose

The individual metrics are not claimed to be novel. The purpose is to test
whether the same representation-instability-before-prediction-failure
phenomenon is robust to the choice of representation geometry.

A finding that appears only for cosine would be metric-dependent.
Agreement across substantially different metrics would be stronger evidence.
"""
    (out / "README_trajectory_analysis.md").write_text(
        readme, encoding="utf-8"
    )

    if not args.no_plots:
        plots(profile, out)

    print("\n" + "=" * 78)
    print("MULTI-METRIC ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Complete sequences analyzed : {len(complete):,}")
    print(f"Images analyzed              : {len(df):,}")

    print("\nMEDIAN BOUNDARIES")
    print("-" * 78)

    metrics = [
        "A_cosine",
        "A_angular",
        "A_euclidean",
        "A_path",
        "A_rate",
        "A_curvature",
        "A_instability",
        "C",
        "B",
    ]

    for m in metrics:
        l = pd.to_numeric(
            summary[f"{m}_left_abs_angle"], errors="coerce"
        ).dropna()
        r = pd.to_numeric(
            summary[f"{m}_right_abs_angle"], errors="coerce"
        ).dropna()

        print(
            f"{m:<22} "
            f"left={l.median():.1f}° "
            f"right={r.median():.1f}°"
            if len(l) and len(r)
            else f"{m:<22} insufficient detections"
        )

    print("\nOutput files:")
    for p in paths.values():
        print(p)
    print(out / "trajectory_analysis_report.json")
    print(out / "README_trajectory_analysis.md")


if __name__ == "__main__":
    main()
