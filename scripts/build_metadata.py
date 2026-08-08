from pathlib import Path
import argparse
import csv
import re
import sys


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_VIEWPOINTS = 215

MIN_VIEWPOINT = 0
MAX_VIEWPOINT = 214

# According to the current dataset definition:
# viewpoint 107 is the frontal reference.
FRONTAL_VIEWPOINT = 107

# IMPORTANT:
# "relative" means:
#   V000 -> -107 degrees
#   V107 ->    0 degrees
#   V214 -> +107 degrees
#
# If your DAZ camera actually uses 0..214 as absolute angles,
# change this to "absolute".
VIEWPOINT_ANGLE_MODE = "relative"


# ------------------------------------------------------------
# Expressions that are useful for the main FER experiment.
# ------------------------------------------------------------

CANONICAL_EXPRESSIONS = {
    "anger": {
        "anger",
        "angry",
        "expressions anger",
    },
    "happiness": {
        "happiness",
        "happy",
        "joy",
    },
    "sadness": {
        "sadness",
        "sad",
    },
    "fear": {
        "fear",
    },
        "surprise": {
        "surprise",
    },
    "disgust": {
        "disgust",
    },
    "contempt": {
        "contempt",
    },
    "confusion": {
        "confusion",
        "confused",
    },
    "desire": {
        "desire",
    },
    "excitement": {
        "excitement",
    },
}


# ------------------------------------------------------------
# Keywords that indicate the folder is NOT a clean expression
# condition for the main viewpoint experiment.
#
# These are kept in metadata, but are not automatically included
# in the strict FER representation analysis set.
# ------------------------------------------------------------

CONTROL_KEYWORDS = [
    "eyeglasses",
    "right hand",
    "left hand",
    "water bottle",
    "comic villain",
    "forehead wrinkles",
    "wink",
    "winking",
    "breathing",
    "coughing",
    "crying",
    "choking",
    "scream",
    "straining",
    "sleepy",
    "sleep",
    "talking",
    "subtle",
    "mouth open",
    "mouth",
    "facial hair",
    "beard",
    "mustache",
    "hair",
    "pose",
]


# ------------------------------------------------------------
# Folders that contain special/non-standard semantic conditions.
# ------------------------------------------------------------

SPECIAL_KEYWORDS = [
    "insanity",
    "insane",
    "wicked",
    "terror",
    "hostile",
    "amimosity",
    "animosity",
    "crazed",
    "crazy",
    "determined",
    "bored",
    "cute",
    "disdain",
    "distress",
    "empathy",
    "romantic",
    "seriousness",
    "silly",
    "humiliated",
    "judgemental",
    "judgmental",
    "nervous",
    "sleepy",
    "straining",
    "perky",
    "poofy",
    "savage",
    "dark intentions",
    "giggling",
    "forehead wrinkles",
]


FIELDNAMES = [
    "image_id",
    "identity",
    "expression_name",
    "expression_family",
    "expression_instance",

    "viewpoint_id",
    "viewpoint_index",
    "viewpoint_angle_deg",
    "angle_mode",

    "is_frontal_reference",
    "is_complete_folder",
    "folder_image_count",

    "condition_type",
    "is_canonical_expression",
    "is_control_condition",
    "is_special_condition",
    "is_analysis_candidate",

    "facial_hair",

    "source_folder",
    "file_name",
    "file_path",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize folder names for comparison.

    Examples:
        "Female_Expressions Anger 01"
        "Expressions Anger"
        "Female_ Happy Perky"

    become comparable strings.
    """

    text = text.strip().lower()

    # Replace repeated whitespace with one space.
    text = re.sub(r"\s+", " ", text)

    # Remove accidental spaces immediately after underscore.
    text = re.sub(r"_\s+", "_", text)

    return text.strip()


def clean_expression_name(expression_name: str) -> str:
    """
    Remove artificial naming prefixes such as:

        Expressions Anger
        Expressions Happiness

    ->

        Anger
        Happiness
    """

    expression_name = expression_name.strip()

    expression_name = re.sub(
        r"^expressions\s+",
        "",
        expression_name,
        flags=re.IGNORECASE,
    )

    return expression_name.strip()


# ============================================================
# FOLDER PARSING
# ============================================================

def parse_folder_name(folder_name: str):
    """
    Parse folders such as:

        Female_Angry 01
        Female_Angry 05
        Female_Expressions Anger 01
        Female_Happy
        Male_Sad 04
        Male_Expressions Happiness 03
        Female_Right Hand On Face

    Returns:

        identity
        expression_name
        expression_instance

    """

    normalized = normalize_text(folder_name)

    match = re.fullmatch(
        r"(female|male)_(.+?)(?:\s+(\d+))?",
        normalized,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    identity = match.group(1).lower()

    expression_name = match.group(2).strip()

    expression_instance = match.group(3) or "01"

    expression_name = clean_expression_name(
        expression_name
    )

    return (
        identity,
        expression_name,
        expression_instance,
    )


# ============================================================
# VIEWPOINT PARSING
# ============================================================

def parse_viewpoint_from_filename(file_name: str):
    """
    Accept filenames such as:

        Female_Angry 01_000.png
        Female_ Angry 01_000.png
        Anything_107.png
        render_214.PNG

    The final three digits before the extension are treated
    as the viewpoint index.
    """

    match = re.search(
        r"_(\d{3})\.png$",
        file_name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(1))


# ============================================================
# ANGLE
# ============================================================

def viewpoint_to_angle(viewpoint_index: int) -> float:
    """
    Convert viewpoint index into the angle used by the experiment.
    """

    if VIEWPOINT_ANGLE_MODE == "relative":
        return viewpoint_index - FRONTAL_VIEWPOINT

    if VIEWPOINT_ANGLE_MODE == "absolute":
        return viewpoint_index

    raise ValueError(
        f"Unknown VIEWPOINT_ANGLE_MODE: "
        f"{VIEWPOINT_ANGLE_MODE}"
    )


# ============================================================
# EXPRESSION CLASSIFICATION
# ============================================================

def classify_expression(expression_name: str):
    """
    Determine whether a folder represents:

        canonical expression
        control condition
        special condition
        other

    Returns:

        expression_family
        is_canonical_expression
        is_control_condition
        is_special_condition
        condition_type
    """

    normalized = normalize_text(expression_name)

    # Remove common prefixes.
    normalized = re.sub(
        r"^expressions\s+",
        "",
        normalized,
    )

    # --------------------------------------------------------
    # Canonical FER expression
    # --------------------------------------------------------

    for family, aliases in CANONICAL_EXPRESSIONS.items():

        for alias in aliases:

            alias_normalized = normalize_text(alias)

            if (
                normalized == alias_normalized
                or normalized.startswith(
                    alias_normalized + " "
                )
            ):
                return (
                    family,
                    True,
                    False,
                    False,
                    "canonical_expression",
                )

    # --------------------------------------------------------
    # Control condition
    # --------------------------------------------------------

    for keyword in CONTROL_KEYWORDS:

        if keyword in normalized:

            return (
                "control",
                False,
                True,
                False,
                "control",
            )

    # --------------------------------------------------------
    # Special semantic expression
    # --------------------------------------------------------

    for keyword in SPECIAL_KEYWORDS:

        if keyword in normalized:

            return (
                "special",
                False,
                False,
                True,
                "special_expression",
            )

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------

    return (
        "other",
        False,
        False,
        False,
        "other",
    )


# ============================================================
# FILENAME PREFIX CHECK
# ============================================================

def expected_filename_prefix(folder_name: str) -> str:
    """
    Return the expected prefix based on the folder name.

    We tolerate accidental spaces in filenames, therefore this
    function is used only for diagnostics.
    """

    return f"{folder_name}_"


def filename_matches_folder(
    file_name: str,
    folder_name: str,
) -> bool:

    stem = Path(file_name).stem

    # Remove the viewpoint suffix.
    stem_without_viewpoint = re.sub(
        r"_\d{3}$",
        "",
        stem,
        flags=re.IGNORECASE,
    )

    expected = normalize_text(
        folder_name
    )

    actual = normalize_text(
        stem_without_viewpoint
    )

    return actual == expected


# ============================================================
# BUILD METADATA
# ============================================================

def build_metadata(
    dataset_root: Path,
    output_file: Path,
    analysis_output_file: Path,
):

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{dataset_root}"
        )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory:\n"
            f"{dataset_root}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    analysis_rows = []

    invalid_folders = []
    invalid_files = []

    folder_warnings = []

    duplicate_viewpoints = []
    missing_viewpoints = []

    unexpected_viewpoints = []

    wrong_filename_prefix = []

    non_png_files = []

    folders = sorted(
        [
            p
            for p in dataset_root.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name.lower(),
    )

    for folder in folders:

        parsed = parse_folder_name(
            folder.name
        )

        if parsed is None:

            invalid_folders.append(
                folder.name
            )

            continue

        (
            identity,
            expression_name,
            expression_instance,
        ) = parsed

        (
            expression_family,
            is_canonical_expression,
            is_control_condition,
            is_special_condition,
            condition_type,
        ) = classify_expression(
            expression_name
        )

        # ----------------------------------------------------
        # Collect PNG files
        # ----------------------------------------------------

        png_files = sorted(
            [
                p
                for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".png"
            ],
            key=lambda p: p.name.lower(),
        )

        # ----------------------------------------------------
        # Detect non-PNG files
        # ----------------------------------------------------

        for p in folder.iterdir():

            if (
                p.is_file()
                and p.suffix.lower() != ".png"
            ):

                non_png_files.append(
                    str(
                        p.relative_to(
                            dataset_root
                        )
                    )
                )

        folder_image_count = len(
            png_files
        )

        is_complete_folder = (
            folder_image_count
            == EXPECTED_VIEWPOINTS
        )

        if not is_complete_folder:

            folder_warnings.append(
                {
                    "folder": folder.name,
                    "actual": folder_image_count,
                    "expected": EXPECTED_VIEWPOINTS,
                }
            )

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        seen_viewpoints = {}

        folder_rows = []

        for image_file in png_files:

            viewpoint_index = (
                parse_viewpoint_from_filename(
                    image_file.name
                )
            )

            if viewpoint_index is None:

                invalid_files.append(
                    str(
                        image_file.relative_to(
                            dataset_root
                        )
                    )
                )

                continue

            if not (
                MIN_VIEWPOINT
                <= viewpoint_index
                <= MAX_VIEWPOINT
            ):

                unexpected_viewpoints.append(
                    {
                        "folder": folder.name,
                        "file": image_file.name,
                        "viewpoint": viewpoint_index,
                    }
                )

                continue

            # ------------------------------------------------
            # Duplicate viewpoint
            # ------------------------------------------------

            if viewpoint_index in seen_viewpoints:

                duplicate_viewpoints.append(
                    {
                        "folder": folder.name,
                        "viewpoint": viewpoint_index,
                        "files": [
                            seen_viewpoints[
                                viewpoint_index
                            ].name,
                            image_file.name,
                        ],
                    }
                )

                continue

            seen_viewpoints[
                viewpoint_index
            ] = image_file

            # ------------------------------------------------
            # Filename diagnostic
            # ------------------------------------------------

            if not filename_matches_folder(
                image_file.name,
                folder.name,
            ):

                wrong_filename_prefix.append(
                    {
                        "folder": folder.name,
                        "file": image_file.name,
                        "expected_prefix":
                            expected_filename_prefix(
                                folder.name
                            ),
                    }
                )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            image_id = image_file.stem

            facial_hair = "unknown"

            relative_path = (
                image_file
                .relative_to(dataset_root)
                .as_posix()
            )

            angle_deg = viewpoint_to_angle(
                viewpoint_index
            )

            is_frontal_reference = (
                viewpoint_index
                == FRONTAL_VIEWPOINT
            )

            # ------------------------------------------------
            # Strict analysis candidate
            #
            # Requirements:
            #
            # 1. complete 215-view folder
            # 2. canonical FER expression
            # 3. not a control
            # 4. not special
            #
            # This prevents pose/accessory/special folders
            # from contaminating the main representation test.
            # ------------------------------------------------

            is_analysis_candidate = (
                is_complete_folder
                and is_canonical_expression
                and not is_control_condition
                and not is_special_condition
            )

            row = {
                "image_id": image_id,

                "identity": identity,

                "expression_name":
                    expression_name,

                "expression_family":
                    expression_family,

                "expression_instance":
                    expression_instance,

                "viewpoint_id":
                    f"V{viewpoint_index:03d}",

                "viewpoint_index":
                    viewpoint_index,

                "viewpoint_angle_deg":
                    angle_deg,

                "angle_mode":
                    VIEWPOINT_ANGLE_MODE,

                "is_frontal_reference":
                    is_frontal_reference,

                "is_complete_folder":
                    is_complete_folder,

                "folder_image_count":
                    folder_image_count,

                "condition_type":
                    condition_type,

                "is_canonical_expression":
                    is_canonical_expression,

                "is_control_condition":
                    is_control_condition,

                "is_special_condition":
                    is_special_condition,

                "is_analysis_candidate":
                    is_analysis_candidate,

                "facial_hair":
                    facial_hair,

                "source_folder":
                    folder.name,

                "file_name":
                    image_file.name,

                "file_path":
                    relative_path,
            }

            folder_rows.append(row)

        # ----------------------------------------------------
        # Missing viewpoints
        # ----------------------------------------------------

        expected = set(
            range(
                MIN_VIEWPOINT,
                MAX_VIEWPOINT + 1,
            )
        )

        actual = set(
            seen_viewpoints.keys()
        )

        missing = sorted(
            expected - actual
        )

        if missing:

            missing_viewpoints.append(
                {
                    "folder":
                        folder.name,
                    "missing":
                        missing,
                }
            )

        rows.extend(folder_rows)

        # ----------------------------------------------------
        # Strict analysis rows
        # ----------------------------------------------------

        for row in folder_rows:

            if row[
                "is_analysis_candidate"
            ]:

                analysis_rows.append(row)

    # ========================================================
    # SORT
    # ========================================================

    def sort_key(row):

        try:
            instance = int(
                row["expression_instance"]
            )
        except Exception:
            instance = 0

        return (
            row["identity"],
            row["expression_family"],
            row["expression_name"].lower(),
            instance,
            int(row["viewpoint_index"]),
        )

    rows.sort(
        key=sort_key
    )

    analysis_rows.sort(
        key=sort_key
    )

    # ========================================================
    # WRITE COMPLETE METADATA
    # ========================================================

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # WRITE STRICT ANALYSIS DATASET
    # ========================================================

    with analysis_output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(
            analysis_rows
        )

    # ========================================================
    # REPORT
    # ========================================================

    valid_folder_count = (
        len(folders)
        - len(invalid_folders)
    )

    complete_folder_count = sum(
        1
        for folder in folders
        if (
            parse_folder_name(
                folder.name
            )
            is not None
            and len(
                [
                    p
                    for p in folder.iterdir()
                    if p.is_file()
                    and p.suffix.lower()
                    == ".png"
                ]
            )
            == EXPECTED_VIEWPOINTS
        )
    )

    candidate_folders = len(
        {
            (
                row["identity"],
                row["source_folder"],
            )
            for row in analysis_rows
        }
    )

    return {
        "folders":
            len(folders),

        "valid_folders":
            valid_folder_count,

        "complete_folders":
            complete_folder_count,

        "rows":
            len(rows),

        "analysis_rows":
            len(analysis_rows),

        "analysis_candidate_folders":
            candidate_folders,

        "invalid_folders":
            invalid_folders,

        "invalid_files":
            invalid_files,

        "folder_warnings":
            folder_warnings,

        "duplicate_viewpoints":
            duplicate_viewpoints,

        "missing_viewpoints":
            missing_viewpoints,

        "unexpected_viewpoints":
            unexpected_viewpoints,

        "wrong_filename_prefix":
            wrong_filename_prefix,

        "non_png_files":
            non_png_files,
    }


# ============================================================
# REPORT
# ============================================================

def print_report(
    report,
    dataset_root,
    output_file,
    analysis_output_file,
):

    print()

    print("=" * 78)
    print(
        "FER REPRESENTATION RELIABILITY "
        "BENCHMARK - METADATA REPORT"
    )
    print("=" * 78)

    print(
        f"Dataset root              : "
        f"{dataset_root}"
    )

    print(
        f"Folders found             : "
        f"{report['folders']}"
    )

    print(
        f"Valid folders             : "
        f"{report['valid_folders']}"
    )

    print(
        f"Complete 215-view folders : "
        f"{report['complete_folders']}"
    )

    print(
        f"Metadata rows             : "
        f"{report['rows']}"
    )

    print(
        f"Strict analysis rows      : "
        f"{report['analysis_rows']}"
    )

    print(
        f"Analysis candidate folders: "
        f"{report['analysis_candidate_folders']}"
    )

    print()

    print(
        f"Viewpoint range           : "
        f"{MIN_VIEWPOINT}-"
        f"{MAX_VIEWPOINT}"
    )

    print(
        f"Frontal viewpoint         : "
        f"V{FRONTAL_VIEWPOINT:03d}"
    )

    print(
        f"Angle mode                : "
        f"{VIEWPOINT_ANGLE_MODE}"
    )

    if VIEWPOINT_ANGLE_MODE == "relative":

        print(
            "Angle mapping             : "
            "V000=-107°, "
            "V107=0°, "
            "V214=+107°"
        )

    else:

        print(
            "Angle mapping             : "
            "V000=0°, "
            "V107=107°, "
            "V214=214°"
        )

    print()

    print(
        f"Expected images/folder    : "
        f"{EXPECTED_VIEWPOINTS}"
    )

    print()

    print(
        f"Folder count warnings     : "
        f"{len(report['folder_warnings'])}"
    )

    print(
        f"Invalid folders           : "
        f"{len(report['invalid_folders'])}"
    )

    print(
        f"Invalid files             : "
        f"{len(report['invalid_files'])}"
    )

    print(
        f"Duplicate viewpoints      : "
        f"{len(report['duplicate_viewpoints'])}"
    )

    print(
        f"Missing viewpoints        : "
        f"{len(report['missing_viewpoints'])}"
    )

    print(
        f"Unexpected viewpoints     : "
        f"{len(report['unexpected_viewpoints'])}"
    )

    print(
        f"Filename warnings         : "
        f"{len(report['wrong_filename_prefix'])}"
    )

    print(
        f"Non-PNG files             : "
        f"{len(report['non_png_files'])}"
    )

    # ========================================================
    # INCOMPLETE FOLDERS
    # ========================================================

    if report["folder_warnings"]:

        print()
        print(
            "--- INCOMPLETE / UNEXPECTED FOLDERS ---"
        )

        for item in report[
            "folder_warnings"
        ][:50]:

            print(
                f"{item['folder']}: "
                f"{item['actual']} images "
                f"(expected "
                f"{item['expected']})"
            )

    # ========================================================
    # INVALID FOLDERS
    # ========================================================

    if report["invalid_folders"]:

        print()
        print(
            "--- INVALID FOLDERS ---"
        )

        for item in report[
            "invalid_folders"
        ][:50]:

            print(item)

    # ========================================================
    # DUPLICATES
    # ========================================================

    if report[
        "duplicate_viewpoints"
    ]:

        print()
        print(
            "--- DUPLICATE VIEWPOINTS ---"
        )

        for item in report[
            "duplicate_viewpoints"
        ][:50]:

            print(
                f"{item['folder']} | "
                f"viewpoint "
                f"{item['viewpoint']} | "
                f"{item['files']}"
            )

    # ========================================================
    # UNEXPECTED VIEWPOINTS
    # ========================================================

    if report[
        "unexpected_viewpoints"
    ]:

        print()
        print(
            "--- UNEXPECTED VIEWPOINTS ---"
        )

        for item in report[
            "unexpected_viewpoints"
        ][:50]:

            print(
                f"{item['folder']} | "
                f"{item['file']} | "
                f"viewpoint "
                f"{item['viewpoint']}"
            )

    # ========================================================
    # FILENAME WARNINGS
    # ========================================================

    if report[
        "wrong_filename_prefix"
    ]:

        print()
        print(
            "--- FILENAME WARNINGS "
            "(NON-FATAL) ---"
        )

        for item in report[
            "wrong_filename_prefix"
        ][:20]:

            print(
                f"Folder: "
                f"{item['folder']} | "
                f"File: "
                f"{item['file']}"
            )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print()
    print(
        f"Complete metadata CSV      : "
        f"{output_file}"
    )

    print(
        f"Strict analysis CSV        : "
        f"{analysis_output_file}"
    )

    print()

    print("=" * 78)
    print(
        "Metadata generation finished."
    )
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build robust metadata for the "
            "FER Representation Reliability "
            "Benchmark."
        )
    )

    parser.add_argument(
        "--dataset-root",
        default=(
            r"E:\extras\khorramgah\1404"
            r"\Daz Studio Library"
            r"\Render_Images_Sequence"
        ),
        help=(
            "Root directory containing "
            "the rendered image folders."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            r"data\metadata.csv"
        ),
        help=(
            "Complete metadata CSV."
        ),
    )

    parser.add_argument(
        "--analysis-output",
        default=(
            r"data\analysis_candidates.csv"
        ),
        help=(
            "Strict CSV containing only "
            "complete canonical-expression "
            "folders suitable for the main "
            "representation experiment."
        ),
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset_root
    )

    output_file = Path(
        args.output
    )

    analysis_output_file = Path(
        args.analysis_output
    )

    try:

        report = build_metadata(
            dataset_root=dataset_root,
            output_file=output_file,
            analysis_output_file=(
                analysis_output_file
            ),
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    print_report(
        report=report,
        dataset_root=dataset_root,
        output_file=output_file,
        analysis_output_file=(
            analysis_output_file
        ),
    )


if __name__ == "__main__":
    main()
