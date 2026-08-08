from pathlib import Path
import argparse
import csv
import re
import sys


EXPECTED_VIEWPOINTS = 215
MIN_VIEWPOINT = 0
MAX_VIEWPOINT = 214
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


def parse_folder_name(folder_name: str):
    """
    Supported examples:
        Female_Angry 01
        Female_Angry 02
        Female_Happy
        Male_Sad 01

    Returns:
        identity, expression_name, expression_instance
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

    return identity, expression_name, expression_instance


def parse_viewpoint_from_filename(file_name: str):
    """
    Example:
        Female_Angry 01_107.png -> 107

    The final three digits before .png are treated as the
    viewpoint index.
    """

    match = re.fullmatch(
        r"(.+)_(\d{3})\.png",
        file_name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(2))


def build_metadata(dataset_root: Path, output_file: Path):
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_root}"
        )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {dataset_root}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    invalid_folders = []
    invalid_files = []
    folder_warnings = []
    duplicate_viewpoints = []
    missing_viewpoints = []

    folders = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )

    for folder in folders:
        parsed = parse_folder_name(folder.name)

        if parsed is None:
            invalid_folders.append(folder.name)
            continue

        identity, expression_name, expression_instance = parsed

        png_files = sorted(
            [
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            ],
            key=lambda p: p.name.lower(),
        )

        if len(png_files) != EXPECTED_VIEWPOINTS:
            folder_warnings.append(
                {
                    "folder": folder.name,
                    "actual": len(png_files),
                    "expected": EXPECTED_VIEWPOINTS,
                }
            )

        seen_viewpoints = {}
        folder_rows = []

        for image_file in png_files:
            viewpoint_index = parse_viewpoint_from_filename(image_file.name)

            if viewpoint_index is None:
                invalid_files.append(
                    str(image_file.relative_to(dataset_root))
                )
                continue

            if not (MIN_VIEWPOINT <= viewpoint_index <= MAX_VIEWPOINT):
                invalid_files.append(
                    f"{image_file.relative_to(dataset_root)} "
                    f"(viewpoint {viewpoint_index} outside 0-214)"
                )
                continue

            if viewpoint_index in seen_viewpoints:
                duplicate_viewpoints.append(
                    {
                        "folder": folder.name,
                        "viewpoint": viewpoint_index,
                        "files": [
                            seen_viewpoints[viewpoint_index].name,
                            image_file.name,
                        ],
                    }
                )
                continue

            seen_viewpoints[viewpoint_index] = image_file

            image_id = image_file.stem

            # The current dataset description does not encode facial-hair
            # configuration in the folder/file name, so we do not guess it.
            facial_hair = "unknown"

            relative_path = image_file.relative_to(dataset_root).as_posix()

            folder_rows.append(
                {
                    "image_id": image_id,
                    "identity": identity,
                    "expression_name": expression_name,
                    "expression_instance": expression_instance,
                    "viewpoint_id": f"V{viewpoint_index:03d}",
                    "viewpoint_index": viewpoint_index,
                    "viewpoint_angle_deg": viewpoint_index,
                    "is_frontal_reference": (
                        viewpoint_index == FRONTAL_VIEWPOINT
                    ),
                    "facial_hair": facial_hair,
                    "source_folder": folder.name,
                    "file_path": relative_path,
                }
            )

        expected = set(range(MIN_VIEWPOINT, MAX_VIEWPOINT + 1))
        actual = set(seen_viewpoints.keys())
        missing = sorted(expected - actual)

        if missing:
            missing_viewpoints.append(
                {
                    "folder": folder.name,
                    "missing": missing,
                }
            )

        rows.extend(folder_rows)

    rows.sort(
        key=lambda r: (
            r["identity"],
            r["expression_name"].lower(),
            int(r["expression_instance"]),
            int(r["viewpoint_index"]),
        )
    )

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

    return {
        "folders": len(folders),
        "rows": len(rows),
        "invalid_folders": invalid_folders,
        "invalid_files": invalid_files,
        "folder_warnings": folder_warnings,
        "duplicate_viewpoints": duplicate_viewpoints,
        "missing_viewpoints": missing_viewpoints,
    }


def print_report(report, dataset_root, output_file):
    print()
    print("=" * 72)
    print("FER RELIABILITY BENCHMARK - METADATA REPORT")
    print("=" * 72)
    print(f"Dataset root       : {dataset_root}")
    print(f"Folders found      : {report['folders']}")
    print(f"Metadata rows      : {report['rows']}")
    print(f"CSV output         : {output_file}")
    print(f"Viewpoint range    : {MIN_VIEWPOINT}-{MAX_VIEWPOINT}")
    print(f"Frontal reference  : {FRONTAL_VIEWPOINT} degrees")
    print()

    print("Folder count warnings:", len(report["folder_warnings"]))
    print("Invalid folders     :", len(report["invalid_folders"]))
    print("Invalid files       :", len(report["invalid_files"]))
    print("Duplicate viewpoints:", len(report["duplicate_viewpoints"]))
    print("Missing viewpoints  :", len(report["missing_viewpoints"]))

    if report["folder_warnings"]:
        print()
        print("--- FOLDERS WITH UNEXPECTED IMAGE COUNTS ---")
        for item in report["folder_warnings"][:50]:
            print(
                f"{item['folder']}: "
                f"{item['actual']} images "
                f"(expected {item['expected']})"
            )

    if report["invalid_folders"]:
        print()
        print("--- INVALID FOLDERS ---")
        for item in report["invalid_folders"][:50]:
            print(item)

    if report["invalid_files"]:
        print()
        print("--- INVALID FILES ---")
        for item in report["invalid_files"][:50]:
            print(item)

    if report["duplicate_viewpoints"]:
        print()
        print("--- DUPLICATE VIEWPOINTS ---")
        for item in report["duplicate_viewpoints"][:50]:
            print(
                f"{item['folder']} | viewpoint {item['viewpoint']} | "
                f"{item['files']}"
            )

    if report["missing_viewpoints"]:
        print()
        print("--- MISSING VIEWPOINTS ---")
        for item in report["missing_viewpoints"][:50]:
            print(
                f"{item['folder']}: {item['missing']}"
            )

    print()
    print("=" * 72)
    print("Metadata generation finished.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build metadata.csv for the FER-Reliability-Benchmark "
            "from the rendered DAZ Studio image folders."
        )
    )

    parser.add_argument(
        "--dataset-root",
        default=r"D:\123",
        help=r"Root directory containing the image folders. Default: D:\123",
    )

    parser.add_argument(
        "--output",
        default=r"data\metadata.csv",
        help=r"Output CSV path. Default: data\metadata.csv",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_file = Path(args.output)

    try:
        report = build_metadata(
            dataset_root=dataset_root,
            output_file=output_file,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print_report(
        report=report,
        dataset_root=dataset_root,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
