from pathlib import Path
import argparse
import csv
import re
import sys


# ============================================================
# DATASET CONFIGURATION
# ============================================================

EXPECTED_VIEWPOINTS = 215

MIN_VIEWPOINT = 0
MAX_VIEWPOINT = 214

# According to the dataset construction:
# 000 -> 0 degrees
# 001 -> 1 degree
# ...
# 107 -> 107 degrees (frontal reference)
# ...
# 214 -> 214 degrees

FRONTAL_VIEWPOINT = 107


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
# FOLDER NAME PARSER
# ============================================================

def parse_folder_name(folder_name: str):
    """
    Parse dataset folder names.

    Supported examples:

        Female_Angry 01
        Female_Angry 02
        Female_Happy
        Female_Expressions Anger 01
        Female_-20
        Female_+10
        Male_Sad 01
        Male_Angry 05
        Male_Breath Taking Over The Moon

    Returns:
        (identity, expression_name, expression_instance)

    Examples:

        Female_Angry 01
            -> ("female", "Angry", "01")

        Female_Happy
            -> ("female", "Happy", "01")

        Male_Expressions Anger 05
            -> ("male", "Expressions Anger", "05")

        Female_-20
            -> ("female", "-20", "01")

    Important:
        The optional instance number is recognized only when it
        appears after whitespace at the END of the folder name.
    """

    match = re.fullmatch(
        r"(Female|Male)_(.+?)(?:\s+(\d+))?",
        folder_name.strip(),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    identity = match.group(1).lower()

    expression_name = match.group(2).strip()

    expression_instance = match.group(3)

    if expression_instance is None:
        expression_instance = "01"

    # Normalize instance number:
    # 1 -> 01
    # 2 -> 02
    # 10 -> 10
    expression_instance = f"{int(expression_instance):02d}"

    return (
        identity,
        expression_name,
        expression_instance,
    )


# ============================================================
# VIEWPOINT PARSER
# ============================================================

def parse_viewpoint_from_filename(file_name: str):
    """
    Extract viewpoint index from the final three digits before .png.

    Examples:

        Female_Angry 01_000.png -> 0
        Female_Angry 01_049.png -> 49
        Female_Angry 01_107.png -> 107
        Female_Angry 01_214.png -> 214

    The dataset convention is:

        filename _000 -> 0 degrees
        filename _001 -> 1 degree
        ...
        filename _214 -> 214 degrees
    """

    match = re.fullmatch(
        r".+_(\d{3})\.png",
        file_name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(1))


# ============================================================
# EXPECTED FILE PREFIX
# ============================================================

def expected_filename_prefix(folder_name: str) -> str:
    """
    Return the expected filename prefix for a folder.

    Example:

        Female_Angry 01
            -> Female_Angry 01_

        Female_Happy
            -> Female_Happy_
    """

    return folder_name.strip() + "_"


# ============================================================
# BUILD METADATA
# ============================================================

def build_metadata(dataset_root: Path, output_file: Path):

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

    wrong_filename_prefix = []

    folder_warnings = []

    duplicate_viewpoints = []

    missing_viewpoints = []

    unexpected_viewpoints = []

    non_png_files = []

    # --------------------------------------------------------
    # Find folders
    # --------------------------------------------------------

    folders = sorted(
        [
            p
            for p in dataset_root.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name.lower(),
    )

    # --------------------------------------------------------
    # Expected viewpoint set
    # --------------------------------------------------------

    expected_viewpoints = set(
        range(
            MIN_VIEWPOINT,
            MAX_VIEWPOINT + 1,
        )
    )

    # --------------------------------------------------------
    # Process each expression folder
    # --------------------------------------------------------

    for folder in folders:

        parsed = parse_folder_name(folder.name)

        if parsed is None:
            invalid_folders.append(
                folder.name
            )
            continue

        identity, expression_name, expression_instance = parsed

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
        # Detect other file types
        # ----------------------------------------------------

        other_files = [
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() != ".png"
        ]

        for file_path in other_files:
            non_png_files.append(
                str(
                    file_path.relative_to(
                        dataset_root
                    )
                )
            )

        # ----------------------------------------------------
        # Check image count
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
        # Process files
        # ----------------------------------------------------

        seen_viewpoints = {}

        folder_rows = []

        expected_prefix = expected_filename_prefix(
            folder.name
        )

        for image_file in png_files:

            # ------------------------------------------------
            # Check filename prefix
            # ------------------------------------------------

            if not image_file.name.startswith(
                expected_prefix
            ):

                wrong_filename_prefix.append(
                    {
                        "folder": folder.name,
                        "file": image_file.name,
                        "expected_prefix": expected_prefix,
                    }
                )

                continue

            # ------------------------------------------------
            # Parse viewpoint
            # ------------------------------------------------

            viewpoint_index = parse_viewpoint_from_filename(
                image_file.name
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

            # ------------------------------------------------
            # Validate viewpoint range
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
            # Detect duplicate viewpoint
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
            # Build metadata row
            # ------------------------------------------------

            image_id = image_file.stem

            relative_path = (
                image_file
                .relative_to(dataset_root)
                .as_posix()
            )

            # We DO NOT infer facial hair from the filename.
            # The current folder structure does not explicitly
            # encode this variable.
            facial_hair = "unknown"

            folder_rows.append(
                {
                    "image_id": image_id,

                    "identity": identity,

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
                        viewpoint_index
                    ),

                    "is_frontal_reference": (
                        viewpoint_index
                        == FRONTAL_VIEWPOINT
                    ),

                    "facial_hair": facial_hair,

                    "source_folder": (
                        folder.name
                    ),

                    "file_path": relative_path,
                }
            )

        # ----------------------------------------------------
        # Check missing viewpoints
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

        "rows": len(rows),

        "invalid_folders": (
            invalid_folders
        ),

        "invalid_files": (
            invalid_files
        ),

        "wrong_filename_prefix": (
            wrong_filename_prefix
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
        "FER RELIABILITY BENCHMARK - "
        "METADATA REPORT"
    )
    print("=" * 78)

    print(
        f"Dataset root       : {dataset_root}"
    )

    print(
        f"Folders found      : "
        f"{report['folders']}"
    )

    print(
        f"Valid folders      : "
        f"{report['valid_folders']}"
    )

    print(
        f"Metadata rows      : "
        f"{report['rows']}"
    )

    print(
        f"CSV output         : "
        f"{output_file}"
    )

    print(
        f"Viewpoint range    : "
        f"{MIN_VIEWPOINT}-{MAX_VIEWPOINT}"
    )

    print(
        f"Frontal reference  : "
        f"{FRONTAL_VIEWPOINT} degrees"
    )

    print(
        f"Expected images/folder : "
        f"{EXPECTED_VIEWPOINTS}"
    )

    print()

    print(
        "Folder count warnings :",
        len(report["folder_warnings"])
    )

    print(
        "Invalid folders       :",
        len(report["invalid_folders"])
    )

    print(
        "Invalid files         :",
        len(report["invalid_files"])
    )

    print(
        "Wrong filename prefix :",
        len(
            report[
                "wrong_filename_prefix"
            ]
        )
    )

    print(
        "Duplicate viewpoints  :",
        len(
            report[
                "duplicate_viewpoints"
            ]
        )
    )

    print(
        "Missing viewpoints    :",
        len(
            report[
                "missing_viewpoints"
            ]
        )
    )

    print(
        "Unexpected viewpoints :",
        len(
            report[
                "unexpected_viewpoints"
            ]
        )
    )

    print(
        "Non-PNG files         :",
        len(
            report[
                "non_png_files"
            ]
        )
    )

    # ========================================================
    # WARNINGS
    # ========================================================

    if report["folder_warnings"]:

        print()
        print(
            "--- FOLDERS WITH UNEXPECTED "
            "IMAGE COUNTS ---"
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

    if report[
        "wrong_filename_prefix"
    ]:

        print()
        print(
            "--- WRONG FILENAME PREFIX ---"
        )

        for item in report[
            "wrong_filename_prefix"
        ][:50]:

            print(
                f"Folder: {item['folder']} | "
                f"File: {item['file']} | "
                f"Expected prefix: "
                f"{item['expected_prefix']}"
            )

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

    if report[
        "missing_viewpoints"
    ]:

        print()
        print(
            "--- MISSING VIEWPOINTS ---"
        )

        for item in report[
            "missing_viewpoints"
        ][:50]:

            print(
                f"{item['folder']}: "
                f"{item['missing']}"
            )

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
                f"viewpoint="
                f"{item['viewpoint']}"
            )

    # ========================================================

    print()
    print("=" * 78)
    print(
        "Metadata generation finished."
    )
    print("=" * 78)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build metadata.csv for the "
            "FER-Reliability-Benchmark "
            "from rendered DAZ Studio images."
        )
    )

    parser.add_argument(
        "--dataset-root",
        default=r"E:\extras\khorramgah\1404\Daz Studio Library\Render_Images_Sequence",
        help=(
            r"Root directory containing the "
            r"image folders. "
            r"Default: D:\123"
        ),
    )

    parser.add_argument(
        "--output",
        default=r"data\metadata.csv",
        help=(
            r"Output CSV path. "
            r"Default: data\metadata.csv"
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
