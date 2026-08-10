# ============================================================
# build_metadata.py
# ============================================================
#
# FER Representation Reliability Benchmark
#
# Builds a complete metadata.csv from the DAZ Studio
# Render_Images_Sequence dataset.
#
# IMPORTANT:
# - Original expression names are preserved.
# - No conversion to 6 basic emotion classes.
# - Viewpoints are V000 ... V214.
# - V107 is the frontal reference.
# - Relative angle:
#       V000 = -107°
#       V107 =    0°
#       V214 = +107°
#
# Required columns for extract_embeddings.py:
#
#   folder
#   gender
#   expression
#   viewpoint
#   angle
#   image_path
#
# Additional useful columns are also stored.
#
# ============================================================

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

FRONTAL_VIEWPOINT = 107


DEFAULT_DATASET_ROOT = (
    r"Render_Images_Sequence"
)

DEFAULT_OUTPUT = r"data\metadata.csv"


# ============================================================
# CSV COLUMNS
# ============================================================
#
# The first six columns are REQUIRED by extract_embeddings.py.
#
# Additional columns are intentionally retained because they are
# useful later for A / B / C analysis.
#
# ============================================================

FIELDNAMES = [
    # Required by extract_embeddings.py
    "folder",
    "gender",
    "expression",
    "viewpoint",
    "angle",
    "image_path",

    # Additional metadata
    "image_id",
    "expression_instance",
    "viewpoint_index",
    "viewpoint_angle_deg",
    "is_frontal_reference",
    "facial_hair",
]


# ============================================================
# VIEWPOINT -> ANGLE
# ============================================================

def viewpoint_to_angle(viewpoint_index):
    """
    Convert viewpoint index to relative angle.

    V000 = -107°
    V107 =    0°
    V214 = +107°
    """

    return viewpoint_index - FRONTAL_VIEWPOINT


# ============================================================
# PARSE VIEWPOINT
# ============================================================

def parse_viewpoint_from_filename(filename):
    """
    Extract the final viewpoint number from filenames such as:

        Female_Happy 01_000.png
        Female_Happy 01_107.png
        Female_Happy 01_214.png

    Returns:
        int viewpoint
        None if parsing fails
    """

    stem = Path(filename).stem

    # Expected structure:
    #
    # anything_000
    # anything_001
    # ...
    # anything_214
    #
    match = re.search(r"_(\d{3})$", stem)

    if match is None:
        return None

    return int(match.group(1))


# ============================================================
# PARSE FOLDER INFORMATION
# ============================================================

def parse_folder_information(folder_name):
    """
    Extract gender and expression information from the folder name.

    Expected examples:

        Female_Happy 01
        Female_Judgemental 01
        Male_Angry 02
        Male_Terror Dark Intentions

    IMPORTANT:
    We DO NOT map expressions into six basic emotion classes.

    For example:

        Judgemental -> Judgemental
        Perky -> Perky
        Terror Dark Intentions -> Terror Dark Intentions
        Happy -> Happy

    Returns:

        gender
        expression
        expression_instance
    """

    parts = folder_name.split("_", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse folder name: {folder_name}"
        )

    gender = parts[0].strip()

    remainder = parts[1].strip()

    if gender not in {"Female", "Male"}:
        raise ValueError(
            f"Unknown gender in folder name: {folder_name}"
        )

    # --------------------------------------------------------
    # Try to detect trailing expression instance number.
    #
    # Examples:
    #
    # Happy 01
    # Angry 02
    # Judgemental 01
    #
    # If there is no trailing number, instance = ""
    # --------------------------------------------------------

    match = re.match(
        r"^(.*?)(?:\s+(\d+))$",
        remainder
    )

    if match:
        expression = match.group(1).strip()
        expression_instance = match.group(2)
    else:
        expression = remainder
        expression_instance = ""

    return (
        gender,
        expression,
        expression_instance,
    )


# ============================================================
# FACIAL HAIR
# ============================================================

def infer_facial_hair(gender):
    """
    Dataset-specific metadata.

    Based on the current dataset description:
        Male   -> facial hair
        Female -> no facial hair

    This is stored only as metadata.
    It does not affect inclusion/exclusion.
    """

    if gender == "Male":
        return True

    if gender == "Female":
        return False

    return False


# ============================================================
# BUILD METADATA
# ============================================================

def build_metadata(
    dataset_root,
    output_file,
):
    """
    Scan dataset and generate metadata.csv.
    """

    dataset_root = Path(dataset_root)
    output_file = Path(output_file)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist:\n{dataset_root}"
        )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root is not a directory:\n{dataset_root}"
        )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    folders = []

    invalid_folders = []

    folder_warnings = []

    invalid_files = []

    filename_warnings = []

    duplicate_viewpoints = []

    missing_viewpoints = []

    unexpected_viewpoints = []

    non_png_files = []

    rows = []

    # --------------------------------------------------------
    # Find expression folders
    # --------------------------------------------------------

    folders = sorted(
        [
            p
            for p in dataset_root.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name.lower(),
    )

    expected_viewpoints = set(
        range(
            MIN_VIEWPOINT,
            MAX_VIEWPOINT + 1,
        )
    )

    # ========================================================
    # PROCESS EACH FOLDER
    # ========================================================

    for folder in folders:

        # ----------------------------------------------------
        # Parse folder information
        # ----------------------------------------------------

        try:

            (
                gender,
                expression,
                expression_instance,
            ) = parse_folder_information(
                folder.name
            )

        except Exception:

            invalid_folders.append(
                folder.name
            )

            continue

        # ----------------------------------------------------
        # Find files
        # ----------------------------------------------------

        all_files = sorted(
            [
                p
                for p in folder.iterdir()
                if p.is_file()
            ],
            key=lambda p: p.name.lower(),
        )

        png_files = [
            p
            for p in all_files
            if p.suffix.lower() == ".png"
        ]

        # ----------------------------------------------------
        # Non-PNG files
        # ----------------------------------------------------

        for file_path in all_files:

            if file_path.suffix.lower() != ".png":

                non_png_files.append(
                    str(
                        file_path.relative_to(
                            dataset_root
                        )
                    )
                )

        # ----------------------------------------------------
        # EXPECTED FOLDER SIZE
        # ----------------------------------------------------

        if len(png_files) != EXPECTED_VIEWPOINTS:

            folder_warnings.append(
                {
                    "folder": folder.name,
                    "actual": len(png_files),
                    "expected": EXPECTED_VIEWPOINTS,
                }
            )

        # ----------------------------------------------------
        # Expected filename prefix
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # We do NOT reject a file just because its filename
        # prefix differs from the folder name.
        #
        # Example:
        #
        # Folder:
        #     Female_S Wink 02
        #
        # File:
        #     Female_Sexy Wink 02_000.png
        #
        # This is a naming inconsistency, but the viewpoint
        # information is still usable.
        #
        # ----------------------------------------------------

        expected_prefix = folder.name + "_"

        # ----------------------------------------------------
        # Track viewpoints
        # ----------------------------------------------------

        seen_viewpoints = {}

        folder_rows = []

        # ====================================================
        # PROCESS IMAGES
        # ====================================================

        for image_file in png_files:

            # ------------------------------------------------
            # Parse viewpoint
            # ------------------------------------------------

            viewpoint_index = (
                parse_viewpoint_from_filename(
                    image_file.name
                )
            )

            # ------------------------------------------------
            # Invalid filename
            # ------------------------------------------------

            if viewpoint_index is None:

                invalid_files.append(
                    str(
                        image_file.relative_to(
                            dataset_root
                        )
                    )
                )

                continue

            # ------------------------------------------------
            # Viewpoint outside expected range
            # ------------------------------------------------

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
            # Filename warning
            # ------------------------------------------------

            if not image_file.name.startswith(
                expected_prefix
            ):

                filename_warnings.append(
                    {
                        "folder": folder.name,
                        "file": image_file.name,
                        "expected_prefix": expected_prefix,
                    }
                )

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
            # IMAGE INFORMATION
            # ------------------------------------------------

            image_id = image_file.stem

            relative_path = (
                Path("Render_Images_Sequence") /
                image_file.relative_to(dataset_root)
            ).as_posix()

            viewpoint_angle = (
                viewpoint_to_angle(
                    viewpoint_index
                )
            )

            facial_hair = infer_facial_hair(
                gender
            )

            viewpoint_id = (
                f"V{viewpoint_index:03d}"
            )

            # ------------------------------------------------
            # Create metadata row
            # ------------------------------------------------
            #
            # IMPORTANT:
            #
            # folder
            # gender
            # expression
            # viewpoint
            # angle
            # image_path
            #
            # are the six fields expected by
            # extract_embeddings.py.
            #
            # ------------------------------------------------

            folder_rows.append(
                {
                    # ========================================
                    # REQUIRED FIELDS
                    # ========================================

                    "folder": folder.name,

                    "gender": gender,

                    "expression": expression,

                    "viewpoint": viewpoint_id,

                    "angle": viewpoint_angle,

                    "image_path": relative_path,

                    # ========================================
                    # ADDITIONAL FIELDS
                    # ========================================

                    "image_id": image_id,

                    "expression_instance": (
                        expression_instance
                    ),

                    "viewpoint_index": (
                        viewpoint_index
                    ),

                    "viewpoint_angle_deg": (
                        viewpoint_angle
                    ),

                    "is_frontal_reference": (
                        viewpoint_index
                        == FRONTAL_VIEWPOINT
                    ),

                    "facial_hair": (
                        facial_hair
                    ),
                }
            )

        # ====================================================
        # MISSING VIEWPOINTS
        # ====================================================

        actual_viewpoints = set(
            seen_viewpoints.keys()
        )

        missing = sorted(
            expected_viewpoints
            - actual_viewpoints
        )

        if missing:

            missing_viewpoints.append(
                {
                    "folder": folder.name,
                    "missing": missing,
                }
            )

        # ----------------------------------------------------
        # Add rows
        # ----------------------------------------------------

        rows.extend(
            folder_rows
        )

    # ========================================================
    # SORT METADATA
    # ========================================================

    rows.sort(
        key=lambda r: (
            r["gender"],
            r["expression"].lower(),
            r["folder"].lower(),
            int(r["viewpoint_index"]),
        )
    )

    # ========================================================
    # WRITE CSV
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

        writer.writerows(rows)

    # ========================================================
    # COMPLETE FOLDERS
    # ========================================================

    warning_folder_names = {
        item["folder"]
        for item in folder_warnings
    }

    complete_folders = sum(
        1
        for folder in folders
        if (
            folder.name
            not in warning_folder_names
            and folder.name
            not in invalid_folders
        )
    )

    # ========================================================
    # RETURN REPORT
    # ========================================================

    return {
        "folders": len(folders),

        "valid_folders": (
            len(folders)
            - len(invalid_folders)
        ),

        "complete_folders": (
            complete_folders
        ),

        "rows": len(rows),

        "invalid_folders": (
            invalid_folders
        ),

        "invalid_files": (
            invalid_files
        ),

        "folder_warnings": (
            folder_warnings
        ),

        "filename_warnings": (
            filename_warnings
        ),

        "duplicate_viewpoints": (
            duplicate_viewpoints
        ),

        "missing_viewpoints": (
            missing_viewpoints
        ),

        "unexpected_viewpoints": (
            unexpected_viewpoints
        ),

        "non_png_files": (
            non_png_files
        ),
    }


# ============================================================
# REPORT
# ============================================================

def print_report(
    report,
    dataset_root,
    output_file,
):
    """
    Print human-readable validation report.
    """

    print()
    print("=" * 78)
    print(
        "FER REPRESENTATION RELIABILITY BENCHMARK "
        "- METADATA REPORT"
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

    print()

    print(
        f"Viewpoint range           : "
        f"{MIN_VIEWPOINT}-{MAX_VIEWPOINT}"
    )

    print(
        f"Frontal viewpoint         : "
        f"V{FRONTAL_VIEWPOINT:03d}"
    )

    print(
        "Angle mode                : relative"
    )

    print(
        "Angle mapping             : "
        "V000=-107°, "
        "V107=0°, "
        "V214=+107°"
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
        f"{len(report['filename_warnings'])}"
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
        ]:

            print(
                f"{item['folder']}: "
                f"{item['actual']} images "
                f"(expected {item['expected']})"
            )

    # ========================================================
    # INVALID FOLDERS
    # ========================================================

    if report["invalid_folders"]:

        print()
        print(
            "--- INVALID FOLDERS ---"
        )

        for folder in report[
            "invalid_folders"
        ]:

            print(folder)

    # ========================================================
    # FILENAME WARNINGS
    # ========================================================

    if report["filename_warnings"]:

        print()
        print(
            "--- FILENAME WARNINGS (NON-FATAL) ---"
        )

        # Do not flood terminal.
        # Show first 50.
        limit = 50

        for item in report[
            "filename_warnings"
        ][:limit]:

            print(
                f"Folder: {item['folder']} | "
                f"File: {item['file']}"
            )

        remaining = (
            len(
                report[
                    "filename_warnings"
                ]
            )
            - limit
        )

        if remaining > 0:

            print(
                f"... and {remaining} more "
                f"filename warnings."
            )

    # ========================================================
    # MISSING VIEWPOINTS
    # ========================================================

    if report["missing_viewpoints"]:

        print()
        print(
            "--- MISSING VIEWPOINTS ---"
        )

        for item in report[
            "missing_viewpoints"
        ]:

            missing = item["missing"]

            # Prevent extremely long output.
            if len(missing) > 20:

                preview = (
                    missing[:10]
                    + ["..."]
                    + missing[-10:]
                )

            else:

                preview = missing

            print(
                f"{item['folder']}: "
                f"{preview}"
            )

    # ========================================================
    # UNEXPECTED VIEWPOINTS
    # ========================================================

    if report["unexpected_viewpoints"]:

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
    # DUPLICATE VIEWPOINTS
    # ========================================================

    if report["duplicate_viewpoints"]:

        print()
        print(
            "--- DUPLICATE VIEWPOINTS ---"
        )

        for item in report[
            "duplicate_viewpoints"
        ][:50]:

            print(
                f"{item['folder']} | "
                f"V{item['viewpoint']:03d} | "
                f"{item['files']}"
            )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        f"Complete metadata CSV      : "
        f"{output_file}"
    )

    print()
    print("=" * 78)
    print(
        "Metadata generation finished."
    )
    print("=" * 78)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build metadata.csv for the "
            "FER Representation Reliability "
            "Benchmark from DAZ Studio "
            "rendered images."
        )
    )

    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Root directory containing "
            "the expression folders."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV path. "
            "Default: data\\metadata.csv"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    dataset_root = Path(
        args.dataset_root
    )

    output_file = Path(
        args.output
    )

    try:

        report = build_metadata(
            dataset_root=dataset_root,
            output_file=output_file,
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
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
