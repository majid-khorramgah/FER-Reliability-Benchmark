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

# In your dataset:
# V000 = -107 degrees relative to frontal
# V107 =   0 degrees (frontal)
# V214 = +107 degrees relative to frontal
FRONTAL_VIEWPOINT = 107

ANGLE_MODE = "relative"

FIELDNAMES = [
    "image_id",
    "identity",
    "expression_name",
    "expression_instance",
    "viewpoint_id",
    "viewpoint_index",
    "viewpoint_angle_deg",
    "is_frontal_reference",
    "facial_hair",
    "source_folder",
    "file_path",
]


# ============================================================
# FOLDER PARSING
# ============================================================

def parse_folder_name(folder_name: str):
    """
    Parse dataset folder names.

    Examples:
        Female_Angry 01
        Female_Happy
        Female_Judgemental 01
        Male_Sad 02
        Male_Right Hand On Face

    Returns:
        identity,
        expression_name,
        expression_instance
    """

    match = re.fullmatch(
        r"(Female|Male)_(.+?)(?:\s+(\d+))?",
        folder_name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    identity = match.group(1).lower()

    expression_name = match.group(2).strip()

    expression_instance = match.group(3) or "01"

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
    Extract viewpoint index from filename.

    Examples:
        Female_Angry 01_000.png -> 0
        Female_Angry 01_107.png -> 107
        Female_Angry 01_214.png -> 214

    The final three digits before .png are treated
    as the viewpoint index.
    """

    match = re.fullmatch(
        r"(.+)_(\d{3})\.png",
        file_name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(2))


# ============================================================
# VIEWPOINT ANGLE
# ============================================================

def viewpoint_to_angle(viewpoint_index: int):
    """
    Convert viewpoint index into angle relative to frontal.

    Mapping:

        V000 -> -107°
        V001 -> -106°
        ...
        V107 ->    0°
        ...
        V213 -> +106°
        V214 -> +107°

    This is a RELATIVE angle representation.

    It does NOT mean the original DAZ camera was literally
    positioned at -107° to +107°.

    It means:
        frontal = 0°
        viewpoint index is measured relative to V107.
    """

    return viewpoint_index - FRONTAL_VIEWPOINT


# ============================================================
# FACIAL HAIR
# ============================================================

def infer_facial_hair(identity: str):
    """
    We currently do NOT infer facial hair from expression names.

    Male/female identity is preserved, but facial-hair configuration
    is kept as unknown because the folder naming does not reliably
    encode all appearance configurations.
    """

    return "unknown"


# ============================================================
# BUILD METADATA
# ============================================================

def build_metadata(
    dataset_root: Path,
    output_file: Path,
):

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{dataset_root}"
        )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory:\n{dataset_root}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    invalid_folders = []
    invalid_files = []

    folder_warnings = []

    duplicate_viewpoints = []

    missing_viewpoints = []

    unexpected_viewpoints = []

    filename_warnings = []

    non_png_files = []

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
    # PROCESS EVERY FOLDER
    # ========================================================

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

        # ----------------------------------------------------
        # ALL FILES
        # ----------------------------------------------------

        all_files = sorted(
            [
                p
                for p in folder.iterdir()
                if p.is_file()
            ],
            key=lambda p: p.name.lower(),
        )

        # ----------------------------------------------------
        # NON-PNG FILES
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
        # PNG FILES
        # ----------------------------------------------------

        png_files = sorted(
            [
                p
                for p in all_files
                if p.is_file()
                and p.suffix.lower() == ".png"
            ],
            key=lambda p: p.name.lower(),
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
        # EXPECTED PREFIX
        #
        # IMPORTANT:
        #
        # We do NOT reject files if their prefix differs from
        # the folder name.
        #
        # Example:
        #
        # Folder:
        # Female_S Wink 02
        #
        # Files:
        # Female_Sexy Wink 02_000.png
        #
        # This is a naming inconsistency, NOT a reason to
        # throw away the image.
        # ----------------------------------------------------

        expected_prefix = folder.name + "_"

        seen_viewpoints = {}

        folder_rows = []

        # ----------------------------------------------------
        # PROCESS IMAGES
        # ----------------------------------------------------

        for image_file in png_files:

            viewpoint_index = parse_viewpoint_from_filename(
                image_file.name
            )

            # ------------------------------------------------
            # INVALID FILENAME
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
            # VIEWPOINT OUTSIDE 0-214
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
            # FILENAME PREFIX WARNING
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
            # DUPLICATE VIEWPOINT
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
            # IMAGE METADATA
            # ------------------------------------------------

            image_id = image_file.stem

            relative_path = (
                image_file
                .relative_to(dataset_root)
                .as_posix()
            )

            viewpoint_angle = viewpoint_to_angle(
                viewpoint_index
            )

            facial_hair = infer_facial_hair(
                identity
            )

            folder_rows.append(
                {
                    "image_id": image_id,

                    "identity": identity,

                    # IMPORTANT:
                    # We preserve the ORIGINAL expression.
                    #
                    # No conversion to:
                    # Happy / Angry / Sad / Fear / Surprise /
                    # Disgust.
                    #
                    # Therefore:
                    # Judgemental stays Judgemental.
                    # Perky stays Perky.
                    # Terror Dark Intentions stays Terror Dark Intentions.
                    # etc.
                    "expression_name": expression_name,

                    "expression_instance": (
                        expression_instance
                    ),

                    "viewpoint_id": (
                        f"V{viewpoint_index:03d}"
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

                    "facial_hair": facial_hair,

                    "source_folder": (
                        folder.name
                    ),

                    "file_path": (
                        relative_path
                    ),
                }
            )

        # ----------------------------------------------------
        # MISSING VIEWPOINTS
        # ----------------------------------------------------

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

        rows.extend(folder_rows)

    # ========================================================
    # SORT METADATA
    # ========================================================

    rows.sort(
        key=lambda r: (
            r["identity"],
            r["expression_name"].lower(),
            int(r["expression_instance"]),
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
    # RETURN REPORT
    # ========================================================

    return {
        "folders": len(folders),

        "valid_folders": (
            len(folders)
            - len(invalid_folders)
        ),

        "complete_folders": sum(
            1
            for folder in folders
            if folder.name
            not in {
                item["folder"]
                for item in folder_warnings
            }
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

        "duplicate_viewpoints": (
            duplicate_viewpoints
        ),

        "missing_viewpoints": (
            missing_viewpoints
        ),

        "unexpected_viewpoints": (
            unexpected_viewpoints
        ),

        "filename_warnings": (
            filename_warnings
        ),

        "non_png_files": (
            non_png_files
        ),
    }


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    report,
    dataset_root,
    output_file,
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
        f"Angle mode                : "
        f"{ANGLE_MODE}"
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
        f"Duplicate viewpoints     : "
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
        ][:50]:

            print(
                f"{item['folder']}: "
                f"{item['actual']} images "
                f"(expected "
                f"{item['expected']})"
            )

    # ========================================================
    # FILENAME WARNINGS
    # ========================================================

    if report["filename_warnings"]:

        print()

        print(
            "--- FILENAME WARNINGS "
            "(NON-FATAL) ---"
        )

        for item in report[
            "filename_warnings"
        ][:50]:

            print(
                f"Folder: "
                f"{item['folder']} | "
                f"File: "
                f"{item['file']}"
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
    # INVALID FILES
    # ========================================================

    if report["invalid_files"]:

        print()

        print(
            "--- INVALID FILES ---"
        )

        for item in report[
            "invalid_files"
        ][:50]:

            print(item)

    # ========================================================
    # DUPLICATES
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
                f"viewpoint "
                f"{item['viewpoint']} | "
                f"{item['files']}"
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
        ][:50]:

            missing = item["missing"]

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
    # FINAL
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
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build metadata.csv for the "
            "FER Representation Reliability "
            "Benchmark from rendered "
            "DAZ Studio image folders."
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
            "the expression folders."
        ),
    )

    parser.add_argument(
        "--output",
        default=r"data\metadata.csv",
        help=(
            "Output CSV path. "
            "Default: data\\metadata.csv"
        ),
    )

    args = parser.parse_args()

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
