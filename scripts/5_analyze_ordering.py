from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

N_BOOTSTRAP = 500

BASELINE_HALF_WIDTH = 5.0

# Number of consecutive viewpoints required to accept
# a detected boundary as a sustained boundary.
SUSTAINED_VIEWPOINTS = 3

# Primary A metric.
#
# We use angular representation drift as the main A measure
# because it is directly interpretable in degrees.
PRIMARY_A = "A_angular"

# Available A metrics.
A_METRICS = [
    "A_cosine",
    "A_angular",
    "A_euclidean",
    "A_path",
    "A_rate",
    "A_curvature",
    "A_instability",
]

# ============================================================
# PATHS
# ============================================================


def project_root() -> Path:
    return Path(__file__).resolve().parents[0]


def analysis_root() -> Path:
    return project_root() / "analysis"


def input_file() -> Path:
    return analysis_root() / "4_analyze_embeddings_trajectory" / "per_view_metrics_multimetric.csv"


# ============================================================
# PRINTING
# ============================================================


def section(title: str) -> None:
    print()
    print("#" * 70)
    print(title)
    print("#" * 70)


# ============================================================
# COLUMN HELPERS
# ============================================================


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:

    mapping = {}

    mapping["expression"] = find_column(
        df,
        [
            "expression",
            "expression_name",
        ],
    )

    mapping["folder"] = find_column(
        df,
        [
            "folder",
            "folder_name",
        ],
    )

    mapping["viewpoint"] = find_column(
        df,
        [
            "viewpoint",
            "angle",
        ],
    )

    mapping["A_cosine"] = find_column(
        df,
        [
            "A_cosine_distance",
            "A_cosine",
        ],
    )

    mapping["A_angular"] = find_column(
        df,
        [
            "A_angular_distance_deg",
            "A_angular",
        ],
    )

    mapping["A_euclidean"] = find_column(
        df,
        [
            "A_euclidean_distance",
            "A_euclidean",
        ],
    )

    mapping["A_path"] = find_column(
        df,
        [
            "A_cumulative_path_from_V107",
            "A_path",
        ],
    )

    mapping["A_rate"] = find_column(
        df,
        [
            "A_rate_per_degree",
            "A_rate",
        ],
    )

    mapping["A_curvature"] = find_column(
        df,
        [
            "A_curvature",
        ],
    )

    mapping["A_instability"] = find_column(
        df,
        [
            "A_trajectory_instability",
            "A_instability",
        ],
    )

    mapping["B"] = find_column(
        df,
        [
            "B_predicted_folder",
            "B_prediction",
            "predicted_folder",
        ],
    )

    mapping["C_margin"] = find_column(
        df,
        [
            "C_margin",
        ],
    )

    return mapping


# ============================================================
# THRESHOLDS
# ============================================================


def load_thresholds() -> dict[str, float]:

    report_file = (
        analysis_root()
        / "trajectory_analysis_report.json"
    )

    thresholds = {}

    if report_file.exists():

        try:

            with open(
                report_file,
                "r",
                encoding="utf-8",
            ) as f:

                report = json.load(f)

            raw = report.get("thresholds", {})

            for key, value in raw.items():

                try:
                    thresholds[key] = float(value)
                except Exception:
                    pass

        except Exception as exc:

            print(
                "Warning: could not load trajectory report:",
                exc,
            )

    # Fallback values from the trajectory analysis.
    defaults = {
        "A_cosine": 0.02737410,
        "A_angular": 13.43702602,
        "A_euclidean": 0.23398377,
        "A_path": 0.41527239,
        "A_rate": 0.10537625,
        "A_curvature": 43.19991900,
        "A_instability": 0.16426415,
        "C_margin": 0.00237080,
    }

    for key, value in defaults.items():

        if key not in thresholds:
            thresholds[key] = value

    return thresholds


# ============================================================
# DATA LOADING
# ============================================================


def load_data() -> tuple[pd.DataFrame, dict[str, str | None]]:

    path = input_file()

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    print("Loading:")
    print(path)

    df = pd.read_csv(path)

    print("Rows loaded:", len(df))
    print("Columns found:", len(df.columns))

    mapping = detect_columns(df)

    section("COLUMN MAPPING")

    for key, value in mapping.items():
        print(f"{key:25s}: {value}")

    required = [
        "expression",
        "folder",
        "viewpoint",
        "B",
        "C_margin",
        PRIMARY_A,
    ]

    missing = [
        x
        for x in required
        if mapping.get(x) is None
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    return df, mapping


# ============================================================
# CLEANING
# ============================================================


def clean_data(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
) -> pd.DataFrame:

    out = df.copy()

    out["_expression"] = (
        out[mapping["expression"]]
        .astype(str)
        .str.strip()
    )

    out["_folder"] = (
        out[mapping["folder"]]
        .astype(str)
        .str.strip()
    )

    out["_viewpoint"] = pd.to_numeric(
        out[mapping["viewpoint"]],
        errors="coerce",
    )

    for metric in A_METRICS:

        col = mapping.get(metric)

        if col is not None:

            out[f"_{metric}"] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

        else:

            out[f"_{metric}"] = np.nan

    out["_C_margin"] = pd.to_numeric(
        out[mapping["C_margin"]],
        errors="coerce",
    )

    out["_B_prediction"] = (
        out[mapping["B"]]
        .astype(str)
        .str.strip()
    )

    return out


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def safe_float(x) -> float:

    try:

        value = float(x)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def first_sustained_true(
    values: np.ndarray,
    viewpoints: np.ndarray,
    n: int = SUSTAINED_VIEWPOINTS,
) -> float:

    if len(values) == 0:
        return np.nan

    values = np.asarray(values, dtype=bool)
    viewpoints = np.asarray(viewpoints)

    if len(values) < n:
        return np.nan

    for i in range(len(values) - n + 1):

        window = values[i:i + n]

        if np.all(window):

            return float(viewpoints[i])

    return np.nan


# ============================================================
# A BOUNDARY
# ============================================================


def detect_A_boundary(
    group: pd.DataFrame,
    metric: str,
    threshold: float,
    direction: str,
) -> float:

    values = group[f"_{metric}"].to_numpy(
        dtype=float
    )

    viewpoints = group["_viewpoint"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(viewpoints)
    )

    if valid.sum() < SUSTAINED_VIEWPOINTS:
        return np.nan

    values = values[valid]
    viewpoints = viewpoints[valid]

    order = np.argsort(viewpoints)

    values = values[order]
    viewpoints = viewpoints[order]

    if direction == "right":

        flags = values >= threshold

    else:

        flags = values >= threshold

    # For left-side analysis we process viewpoints
    # from high angle toward low angle.
    if direction == "left":

        values = values[::-1]
        viewpoints = viewpoints[::-1]

        flags = values >= threshold

    return first_sustained_true(
        flags,
        viewpoints,
    )


# ============================================================
# C BOUNDARY
# ============================================================


def detect_C_boundary(
    group: pd.DataFrame,
    threshold: float,
    direction: str,
) -> float:

    margin = group["_C_margin"].to_numpy(
        dtype=float
    )

    viewpoints = group["_viewpoint"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(margin)
        & np.isfinite(viewpoints)
    )

    if valid.sum() < SUSTAINED_VIEWPOINTS:
        return np.nan

    margin = margin[valid]
    viewpoints = viewpoints[valid]

    order = np.argsort(viewpoints)

    margin = margin[order]
    viewpoints = viewpoints[order]

    # Low margin = loss of separability.
    flags = margin <= threshold

    if direction == "left":

        margin = margin[::-1]
        viewpoints = viewpoints[::-1]

        flags = margin <= threshold

    return first_sustained_true(
        flags,
        viewpoints,
    )


# ============================================================
# B BOUNDARY
# ============================================================


def detect_B_boundary(
    group: pd.DataFrame,
    true_folder: str,
    direction: str,
) -> float:

    prediction = (
        group["_B_prediction"]
        .astype(str)
        .str.strip()
    )

    viewpoints = group["_viewpoint"].to_numpy(
        dtype=float
    )

    valid_view = np.isfinite(viewpoints)

    prediction = prediction.to_numpy()[valid_view]
    viewpoints = viewpoints[valid_view]

    if len(viewpoints) < SUSTAINED_VIEWPOINTS:
        return np.nan

    order = np.argsort(viewpoints)

    prediction = prediction[order]
    viewpoints = viewpoints[order]

    failure = np.array(
        [
            p != true_folder
            and p.lower() not in {
                "",
                "nan",
                "none",
                "null",
            }
            for p in prediction
        ],
        dtype=bool,
    )

    if direction == "left":

        failure = failure[::-1]
        viewpoints = viewpoints[::-1]

    return first_sustained_true(
        failure,
        viewpoints,
    )


# ============================================================
# EXPRESSION ANALYSIS
# ============================================================


def analyze_expression(
    group: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict:

    expression = str(
        group["_expression"].iloc[0]
    )

    folder = str(
        group["_folder"].iloc[0]
    )

    result = {
        "expression": expression,
        "folder": folder,
    }

    viewpoints = pd.to_numeric(
        group["_viewpoint"],
        errors="coerce",
    )

    unique_viewpoints = sorted(
        viewpoints.dropna().unique()
    )

    result["n_rows"] = int(len(group))
    result["n_viewpoints"] = int(
        len(unique_viewpoints)
    )

    result["complete_215"] = (
        len(unique_viewpoints) == 215
        and set(unique_viewpoints)
        == set(range(215))
    )

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    for metric in A_METRICS:

        threshold = thresholds.get(metric)

        if threshold is None:
            continue

        left = detect_A_boundary(
            group,
            metric,
            threshold,
            "left",
        )

        right = detect_A_boundary(
            group,
            metric,
            threshold,
            "right",
        )

        result[f"{metric}_left"] = left
        result[f"{metric}_right"] = right

        result[f"{metric}_detected_left"] = bool(
            np.isfinite(left)
        )

        result[f"{metric}_detected_right"] = bool(
            np.isfinite(right)
        )

    # --------------------------------------------------------
    # C
    # --------------------------------------------------------

    C_left = detect_C_boundary(
        group,
        thresholds["C_margin"],
        "left",
    )

    C_right = detect_C_boundary(
        group,
        thresholds["C_margin"],
        "right",
    )

    result["C_left"] = C_left
    result["C_right"] = C_right

    result["C_detected_left"] = bool(
        np.isfinite(C_left)
    )

    result["C_detected_right"] = bool(
        np.isfinite(C_right)
    )

    # --------------------------------------------------------
    # B
    # --------------------------------------------------------

    B_left = detect_B_boundary(
        group,
        folder,
        "left",
    )

    B_right = detect_B_boundary(
        group,
        folder,
        "right",
    )

    result["B_left"] = B_left
    result["B_right"] = B_right

    result["B_detected_left"] = bool(
        np.isfinite(B_left)
    )

    result["B_detected_right"] = bool(
        np.isfinite(B_right)
    )

    # --------------------------------------------------------
    # Missing reasons
    # --------------------------------------------------------

    reasons_left = []
    reasons_right = []

    if not np.isfinite(
        result[f"{PRIMARY_A}_left"]
    ):
        reasons_left.append("A")

    if not np.isfinite(C_left):
        reasons_left.append("C")

    if not np.isfinite(B_left):
        reasons_left.append("B")

    if not np.isfinite(
        result[f"{PRIMARY_A}_right"]
    ):
        reasons_right.append("A")

    if not np.isfinite(C_right):
        reasons_right.append("C")

    if not np.isfinite(B_right):
        reasons_right.append("B")

    result["missing_left"] = ",".join(
        reasons_left
    )

    result["missing_right"] = ",".join(
        reasons_right
    )

    # --------------------------------------------------------
    # Ordering
    # --------------------------------------------------------

    result["left_ordering"] = classify_order(
        result[f"{PRIMARY_A}_left"],
        C_left,
        B_left,
    )

    result["right_ordering"] = classify_order(
        result[f"{PRIMARY_A}_right"],
        C_right,
        B_right,
    )

    return result


# ============================================================
# ORDER CLASSIFICATION
# ============================================================


def classify_order(
    A: float,
    C: float,
    B: float,
) -> str:

    if not (
        np.isfinite(A)
        and np.isfinite(C)
        and np.isfinite(B)
    ):

        return "incomplete"

    values = {
        "A": float(A),
        "C": float(C),
        "B": float(B),
    }

    ordered = sorted(
        values.items(),
        key=lambda x: x[1],
    )

    # Handle ties.
    tolerance = 1e-9

    if (
        abs(ordered[0][1] - ordered[1][1])
        <= tolerance
        or
        abs(ordered[1][1] - ordered[2][1])
        <= tolerance
    ):

        return "tie"

    return " < ".join(
        x[0]
        for x in ordered
    )


# ============================================================
# ALL ORDERING COUNTS
# ============================================================


def ordering_counts(
    df: pd.DataFrame,
    side: str,
) -> Counter:

    col = f"{side}_ordering"

    return Counter(
        df[col].fillna("incomplete")
    )


# ============================================================
# BOOTSTRAP
# ============================================================


def bootstrap_orderings(
    expression_df: pd.DataFrame,
    side: str,
    repetitions: int = N_BOOTSTRAP,
) -> pd.DataFrame:

    rng = np.random.default_rng(42)

    rows = []

    if len(expression_df) == 0:
        return pd.DataFrame()

    for rep in range(repetitions):

        sample_indices = rng.integers(
            0,
            len(expression_df),
            size=len(expression_df),
        )

        sample = expression_df.iloc[
            sample_indices
        ]

        counts = ordering_counts(
            sample,
            side,
        )

        total = len(sample)

        for ordering, count in counts.items():

            rows.append(
                {
                    "bootstrap": rep,
                    "side": side,
                    "ordering": ordering,
                    "count": count,
                    "proportion": (
                        count / total
                        if total
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================


def bootstrap_summary(
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:

    if bootstrap_df.empty:
        return pd.DataFrame()

    rows = []

    for (
        side,
        ordering,
    ), group in bootstrap_df.groupby(
        ["side", "ordering"]
    ):

        values = group["proportion"].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "side": side,
                "ordering": ordering,
                "median": float(
                    np.nanmedian(values)
                ),
                "ci_2_5": float(
                    np.nanpercentile(
                        values,
                        2.5,
                    )
                ),
                "ci_97_5": float(
                    np.nanpercentile(
                        values,
                        97.5,
                    )
                ),
                "n_bootstrap": int(
                    np.sum(
                        np.isfinite(values)
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# POPULATION SUMMARY
# ============================================================


def population_summary(
    expression_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    total = len(expression_df)

    for side in ["left", "right"]:

        counts = ordering_counts(
            expression_df,
            side,
        )

        for ordering, count in counts.items():

            rows.append(
                {
                    "side": side,
                    "ordering": ordering,
                    "count": int(count),
                    "proportion": (
                        float(count / total)
                        if total
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# DIAGNOSTIC SUMMARY
# ============================================================


def diagnostic_summary(
    expression_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for side in ["left", "right"]:

        Acol = f"{PRIMARY_A}_{side}"
        Ccol = f"C_{side}"
        Bcol = f"B_{side}"

        total = len(expression_df)

        A = np.isfinite(
            expression_df[Acol].to_numpy(
                dtype=float
            )
        )

        C = np.isfinite(
            expression_df[Ccol].to_numpy(
                dtype=float
            )
        )

        B = np.isfinite(
            expression_df[Bcol].to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "side": side,
                "total_expressions": total,
                "A_detected": int(A.sum()),
                "C_detected": int(C.sum()),
                "B_detected": int(B.sum()),
                "A_missing": int((~A).sum()),
                "C_missing": int((~C).sum()),
                "B_missing": int((~B).sum()),
                "all_three": int(
                    (A & C & B).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BOUNDARY TABLE
# ============================================================


def make_boundary_table(
    expression_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in expression_df.iterrows():

        for side in ["left", "right"]:

            A = row[
                f"{PRIMARY_A}_{side}"
            ]

            C = row[
                f"C_{side}"
            ]

            B = row[
                f"B_{side}"
            ]

            rows.append(
                {
                    "expression": row[
                        "expression"
                    ],
                    "folder": row[
                        "folder"
                    ],
                    "side": side,
                    "A_boundary": A,
                    "C_boundary": C,
                    "B_boundary": B,
                    "ordering": row[
                        f"{side}_ordering"
                    ],
                    "missing": row[
                        f"missing_{side}"
                    ],
                    "complete_215": row[
                        "complete_215"
                    ],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PLOT ORDERINGS
# ============================================================


def plot_orderings(
    population: pd.DataFrame,
    output_path: Path,
) -> None:

    if population.empty:
        return

    pivot = population.pivot_table(
        index="ordering",
        columns="side",
        values="proportion",
        aggfunc="sum",
        fill_value=0,
    )

    if pivot.empty:
        return

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.set_ylabel(
        "Proportion of expressions"
    )

    ax.set_xlabel(
        "Boundary ordering"
    )

    ax.set_title(
        "A / C / B Boundary Ordering"
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            float(
                pivot.to_numpy().max()
            ) * 1.15,
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# PLOT BOUNDARIES
# ============================================================


def plot_boundaries(
    expression_df: pd.DataFrame,
    output_path: Path,
) -> None:

    rows = []

    for side in ["left", "right"]:

        for _, row in expression_df.iterrows():

            A = safe_float(
                row[f"{PRIMARY_A}_{side}"]
            )

            C = safe_float(
                row[f"C_{side}"]
            )

            B = safe_float(
                row[f"B_{side}"]
            )

            if (
                np.isfinite(A)
                and np.isfinite(C)
                and np.isfinite(B)
            ):

                rows.extend(
                    [
                        {
                            "side": side,
                            "metric": "A",
                            "boundary": A,
                        },
                        {
                            "side": side,
                            "metric": "C",
                            "boundary": C,
                        },
                        {
                            "side": side,
                            "metric": "B",
                            "boundary": B,
                        },
                    ]
                )

    if not rows:
        return

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    data = []

    labels = []

    for side in ["left", "right"]:

        for metric in ["A", "C", "B"]:

            values = plot_df[
                (
                    plot_df["side"] == side
                )
                &
                (
                    plot_df["metric"] == metric
                )
            ]["boundary"].dropna()

            if len(values):

                data.append(values.to_numpy())
                labels.append(
                    f"{side}\n{metric}"
                )

    if not data:
        plt.close(fig)
        return

    ax.boxplot(
        data,
        tick_labels=labels,
    )

    ax.set_ylabel(
        "Critical viewpoint (degrees)"
    )

    ax.set_title(
        "A / C / B Critical Boundaries"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# REPORT
# ============================================================


def write_report(
    output_path: Path,
    expression_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    population_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:

    report = {
        "primary_A_metric": PRIMARY_A,
        "n_expressions": int(
            len(expression_df)
        ),
        "thresholds": thresholds,
        "diagnostics": diagnostic_df.to_dict(
            orient="records"
        ),
        "population_orderings": population_df.to_dict(
            orient="records"
        ),
        "bootstrap_orderings": bootstrap_summary_df.to_dict(
            orient="records"
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    root = project_root()
    analysis = analysis_root() / "5_analyze_ordering"

    analysis.mkdir(
        parents=True,
        exist_ok=True,
    )

    section(
        "A / B / C ORDERING ANALYSIS — FIXED"
    )

    print()
    print("Project root:")
    print(root)

    print()
    print("Analysis directory:")
    print(analysis)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df, mapping = load_data()

    df = clean_data(
        df,
        mapping,
    )

    thresholds = load_thresholds()

    section("THRESHOLDS")

    for key in [
        "A_cosine",
        "A_angular",
        "A_euclidean",
        "A_path",
        "A_rate",
        "A_curvature",
        "A_instability",
        "C_margin",
    ]:

        print(
            f"{key:25s}: "
            f"{thresholds.get(key, np.nan):.8f}"
        )

    print()
    print(
        "Primary A metric:",
        PRIMARY_A,
    )

    # --------------------------------------------------------
    # COMPLETE SEQUENCES
    # --------------------------------------------------------

    expression_groups = []

    for (
        expression,
        folder,
    ), group in df.groupby(
        ["_expression", "_folder"],
        sort=False,
    ):

        viewpoints = sorted(
            group["_viewpoint"]
            .dropna()
            .unique()
        )

        if (
            len(viewpoints) == 215
            and set(viewpoints)
            == set(range(215))
        ):

            expression_groups.append(
                group
            )

    print()
    print(
        "Complete expression sequences:",
        len(expression_groups),
    )

    if not expression_groups:

        raise RuntimeError(
            "No complete 215-viewpoint sequences found."
        )

    # --------------------------------------------------------
    # ANALYZE EXPRESSIONS
    # --------------------------------------------------------

    section(
        "ANALYZING EXPRESSION BOUNDARIES"
    )

    results = []

    for index, group in enumerate(
        expression_groups,
        start=1,
    ):

        results.append(
            analyze_expression(
                group,
                thresholds,
            )
        )

    expression_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    section(
        "DETECTION DIAGNOSTICS"
    )

    diagnostics = diagnostic_summary(
        expression_df
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BOUNDARY TABLE
    # --------------------------------------------------------

    boundary_df = make_boundary_table(
        expression_df
    )

    # --------------------------------------------------------
    # POPULATION ORDERINGS
    # --------------------------------------------------------

    section(
        "POPULATION ORDERINGS"
    )

    population_df = population_summary(
        expression_df
    )

    for side in ["left", "right"]:

        print()
        print(
            side.upper(),
            "VIEWPOINTS"
        )

        side_df = population_df[
            population_df["side"] == side
        ].sort_values(
            "proportion",
            ascending=False,
        )

        if side_df.empty:

            print(
                "No complete orderings."
            )

        else:

            print(
                side_df.to_string(
                    index=False
                )
            )

    # --------------------------------------------------------
    # EXACT A < C < B
    # --------------------------------------------------------

    print()

    for side in ["left", "right"]:

        values = expression_df[
            f"{side}_ordering"
        ]

        complete = (
            values != "incomplete"
        )

        complete_n = int(
            complete.sum()
        )

        target_n = int(
            (
                values == "A < C < B"
            ).sum()
        )

        percentage = (
            100.0 * target_n / complete_n
            if complete_n
            else np.nan
        )

        print()
        print(
            f"{side.upper()}:"
        )

        print(
            "Complete A/B/C cases:",
            complete_n,
        )

        print(
            f"A < C < B: "
            f"{target_n}/{complete_n} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # MISSING REASONS
    # --------------------------------------------------------

    section(
        "MISSING BOUNDARY DIAGNOSTICS"
    )

    for side in ["left", "right"]:

        print()
        print(
            side.upper(),
            "MISSING COMPONENTS"
        )

        counts = Counter()

        for value in expression_df[
            f"missing_{side}"
        ]:

            if value == "":
                counts["none"] += 1
            else:
                counts[value] += 1

        for key, value in counts.most_common():

            print(
                f"{key:15s}: "
                f"{value}"
            )

    # --------------------------------------------------------
    # BOOTSTRAP
    # --------------------------------------------------------

    section(
        f"BOOTSTRAP ({N_BOOTSTRAP} repetitions)"
    )

    bootstrap_frames = []

    for side in ["left", "right"]:

        boot = bootstrap_orderings(
            expression_df,
            side,
            N_BOOTSTRAP,
        )

        bootstrap_frames.append(
            boot
        )

    bootstrap_df = pd.concat(
        bootstrap_frames,
        ignore_index=True,
    )

    bootstrap_summary_df = bootstrap_summary(
        bootstrap_df
    )

    if not bootstrap_summary_df.empty:

        print(
            bootstrap_summary_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    section(
        "SAVING RESULTS"
    )

    expression_file = (
        analysis
        / "expression_ordering_fixed.csv"
    )

    population_file = (
        analysis
        / "ordering_population_summary_fixed.csv"
    )

    diagnostics_file = (
        analysis
        / "ordering_detection_diagnostics.csv"
    )

    bootstrap_file = (
        analysis
        / "ordering_bootstrap_ci_fixed.csv"
    )

    boundary_file = (
        analysis
        / "A_C_B_boundaries_fixed.csv"
    )

    report_file = (
        analysis
        / "ordering_analysis_fixed_report.json"
    )

    pattern_plot = (
        analysis
        / "ordering_patterns_fixed.png"
    )

    boundary_plot = (
        analysis
        / "ordering_boundaries_fixed.png"
    )

    expression_df.to_csv(
        expression_file,
        index=False,
    )

    population_df.to_csv(
        population_file,
        index=False,
    )

    diagnostics.to_csv(
        diagnostics_file,
        index=False,
    )

    bootstrap_summary_df.to_csv(
        bootstrap_file,
        index=False,
    )

    boundary_df.to_csv(
        boundary_file,
        index=False,
    )

    plot_orderings(
        population_df,
        pattern_plot,
    )

    plot_boundaries(
        expression_df,
        boundary_plot,
    )

    write_report(
        report_file,
        expression_df,
        diagnostics,
        population_df,
        bootstrap_summary_df,
        thresholds,
    )

    # --------------------------------------------------------
    # EXAMPLE EXPRESSIONS
    # --------------------------------------------------------

    section(
        "EXAMPLE EXPRESSION RESULTS"
    )

    display_cols = [
        "expression",
        "folder",
        f"{PRIMARY_A}_left",
        "C_left",
        "B_left",
        "left_ordering",
        "missing_left",
        f"{PRIMARY_A}_right",
        "C_right",
        "B_right",
        "right_ordering",
        "missing_right",
    ]

    existing_cols = [
        c
        for c in display_cols
        if c in expression_df.columns
    ]

    print(
        expression_df[
            existing_cols
        ].head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    section("DONE")

    print()
    print(
        "Primary A metric:",
        PRIMARY_A,
    )

    print(
        "Complete expressions:",
        len(expression_df),
    )

    print()
    print("Output files:")

    print(expression_file)
    print(population_file)
    print(diagnostics_file)
    print(bootstrap_file)
    print(boundary_file)
    print(report_file)
    print(pattern_plot)
    print(boundary_plot)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The ordering A < C < B is NOT assumed."
    )

    print(
        "The script reports whatever ordering "
        "is actually observed in the data."
    )

    print(
        "If A, B, or C is missing, the exact "
        "missing component is reported instead "
        "of silently labeling the case incomplete."
    )


if __name__ == "__main__":
    main()