"""
FER Representation Reliability Benchmark
=========================================

Extracts image embeddings for the full FER dataset.

Features
--------
- Reads data/metadata.csv
- Processes images in batches
- Automatically uses CUDA GPU when available
- Uses mixed precision on CUDA
- Does NOT load the whole dataset into RAM
- Resumes automatically after interruption
- Saves embeddings in chunks
- Saves metadata alongside embeddings
- Preserves:
    folder
    gender
    expression
    viewpoint
    angle
    image_path

Recommended GPU:
    NVIDIA RTX 3090 24GB

Recommended model:
    DINOv2 ViT-B/14

Why DINOv2?
    We want a general visual representation rather than a FER-specific
    classifier whose output space is already tied to emotion classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------
# PIL safety
# ---------------------------------------------------------------------

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_METADATA = Path("data/metadata.csv")
DEFAULT_OUTPUT = Path("data/embeddings")

# DINOv2 model
MODEL_NAME = "dinov2_vitb14"

# RTX 3090:
# Start conservatively with 32.
# If stable, try 48 or 64.
DEFAULT_BATCH_SIZE = 32

# Number of images stored per output shard.
# 10,000 images × 768 floats × 4 bytes ≈ 30.7 MB
SHARD_SIZE = 10_000

NUM_WORKERS = 8

IMAGE_SIZE = 224


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class FERMetadataDataset(Dataset):

    def __init__(
        self,
        rows: List[Dict[str, str]],
        transform=None,
    ):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        row = self.rows[idx]

        image_path = Path(row["image_path"])

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as e:
            print(
                f"\n[WARNING] Could not read image:\n"
                f"  {image_path}\n"
                f"  Error: {e}\n"
            )

            # Black fallback image.
            image = Image.new(
                "RGB",
                (IMAGE_SIZE, IMAGE_SIZE),
                (0, 0, 0),
            )

        if self.transform is not None:
            image = self.transform(image)

        return image, idx


# ---------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------

def load_metadata(metadata_path: Path) -> List[Dict[str, str]]:

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found:\n{metadata_path}"
        )

    rows = []

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(
                "metadata.csv does not contain a header."
            )

        required = [
            "folder",
            "gender",
            "expression",
            "viewpoint",
            "angle",
            "image_path",
        ]

        missing = [
            x for x in required
            if x not in reader.fieldnames
        ]

        if missing:
            raise RuntimeError(
                "metadata.csv is missing required columns:\n"
                + "\n".join(missing)
            )

        for row in reader:

            rows.append({
                "folder": row["folder"],
                "gender": row["gender"],
                "expression": row["expression"],
                "viewpoint": row["viewpoint"],
                "angle": row["angle"],
                "image_path": row["image_path"],
            })

    return rows


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def select_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("\nGPU detected:")
        print(f"  {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)

        print(
            f"  VRAM: "
            f"{props.total_memory / (1024 ** 3):.1f} GB"
        )

        return device

    print("\nWARNING: CUDA GPU not detected.")
    print("Running on CPU.")
    print("This will be considerably slower.\n")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model(device):

    print("\nLoading DINOv2 ViT-B/14...")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
    )

    model.eval()
    model.to(device)

    # We never need gradients.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print("Model loaded.")

    return model


# ---------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------

def create_transform():

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


# ---------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------

def state_path(output_dir: Path):

    return output_dir / "progress.json"


def load_progress(output_dir: Path):

    path = state_path(output_dir)

    if not path.exists():
        return {
            "processed": 0,
            "shard_index": 0,
        }

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        print(
            "[WARNING] progress.json is corrupted. "
            "Starting from zero."
        )

        return {
            "processed": 0,
            "shard_index": 0,
        }


def save_progress(
    output_dir: Path,
    processed: int,
    shard_index: int,
):

    data = {
        "processed": processed,
        "shard_index": shard_index,
    }

    temp_path = output_dir / "progress.tmp"

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

    os.replace(
        temp_path,
        state_path(output_dir),
    )


# ---------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------

def save_shard(
    output_dir: Path,
    shard_index: int,
    embeddings: np.ndarray,
    rows: List[Dict[str, str]],
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_file = (
        output_dir
        / f"embeddings_{shard_index:05d}.npy"
    )

    metadata_file = (
        output_dir
        / f"metadata_{shard_index:05d}.csv"
    )

    np.save(
        embedding_file,
        embeddings,
    )

    fieldnames = [
        "folder",
        "gender",
        "expression",
        "viewpoint",
        "angle",
        "image_path",
    ]

    with metadata_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"\nSaved shard {shard_index}:"
    )

    print(
        f"  Embeddings: {embedding_file}"
    )

    print(
        f"  Metadata:   {metadata_file}"
    )

    print(
        f"  Shape:      {embeddings.shape}"
    )


# ---------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------

def extract_embeddings(
    metadata_path: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
):

    print("=" * 78)
    print("FER REPRESENTATION RELIABILITY BENCHMARK")
    print("DINOv2 EMBEDDING EXTRACTION")
    print("=" * 78)

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = select_device()

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    rows = load_metadata(
        metadata_path
    )

    total = len(rows)

    print(
        f"\nImages in metadata: {total:,}"
    )

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------

    progress = load_progress(
        output_dir
    )

    start_index = int(
        progress.get("processed", 0)
    )

    shard_index = int(
        progress.get("shard_index", 0)
    )

    if start_index > 0:

        print(
            f"\nRESUME MODE"
        )

        print(
            f"Already processed: "
            f"{start_index:,}"
        )

        print(
            f"Remaining: "
            f"{total - start_index:,}"
        )

    else:

        print(
            "\nStarting from beginning."
        )

    if start_index >= total:

        print(
            "\nAll images are already processed."
        )

        return

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    transform = create_transform()

    dataset = FERMetadataDataset(
        rows[start_index:],
        transform=transform,
    )

    # ---------------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------------

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    # Windows multiprocessing safety.
    if num_workers > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

        loader_kwargs[
            "prefetch_factor"
        ] = 2

    loader = DataLoader(
        **loader_kwargs
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = load_model(
        device
    )

    # ---------------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------------

    shard_embeddings = []
    shard_rows = []

    processed = start_index

    shard_count = shard_index

    start_time = time.time()

    print(
        "\nStarting extraction..."
    )

    print(
        f"Batch size:   {batch_size}"
    )

    print(
        f"Workers:      {num_workers}"
    )

    print(
        f"Shard size:   {SHARD_SIZE:,}"
    )

    print(
        f"Device:       {device}"
    )

    print()

    # DINOv2 outputs 768-dimensional embeddings
    embedding_dim = None

    with torch.inference_mode():

        for batch_images, batch_indices in loader:

            batch_images = batch_images.to(
                device,
                non_blocking=True,
            )

            # -------------------------------------------------------
            # Mixed precision on CUDA
            # -------------------------------------------------------

            if device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    batch_embeddings = model(
                        batch_images
                    )

            else:

                batch_embeddings = model(
                    batch_images
                )

            # -------------------------------------------------------
            # Normalize embeddings
            # -------------------------------------------------------

            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings,
                p=2,
                dim=1,
            )

            # -------------------------------------------------------
            # Convert to CPU
            # -------------------------------------------------------

            batch_embeddings = (
                batch_embeddings
                .float()
                .cpu()
                .numpy()
            )

            if embedding_dim is None:

                embedding_dim = (
                    batch_embeddings.shape[1]
                )

                print(
                    f"Embedding dimension: "
                    f"{embedding_dim}"
                )

            # -------------------------------------------------------
            # Add to current shard
            # -------------------------------------------------------

            for local_i, original_idx in enumerate(
                batch_indices.tolist()
            ):

                actual_index = (
                    start_index
                    + original_idx
                )

                row = rows[
                    actual_index
                ]

                shard_embeddings.append(
                    batch_embeddings[local_i]
                )

                shard_rows.append(
                    row
                )

                processed += 1

                # ---------------------------------------------------
                # Save shard
                # ---------------------------------------------------

                if (
                    len(shard_embeddings)
                    >= SHARD_SIZE
                ):

                    embeddings_array = np.stack(
                        shard_embeddings,
                        axis=0,
                    )

                    save_shard(
                        output_dir=output_dir,
                        shard_index=shard_count,
                        embeddings=embeddings_array,
                        rows=shard_rows,
                    )

                    shard_embeddings.clear()
                    shard_rows.clear()

                    shard_count += 1

                    save_progress(
                        output_dir=output_dir,
                        processed=processed,
                        shard_index=shard_count,
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    speed = (
                        processed - start_index
                    ) / max(elapsed, 1e-6)

                    remaining = (
                        total - processed
                    )

                    eta_seconds = (
                        remaining / speed
                        if speed > 0
                        else 0
                    )

                    print(
                        f"\nProgress:"
                    )

                    print(
                        f"  {processed:,} / "
                        f"{total:,}"
                    )

                    print(
                        f"  Speed: "
                        f"{speed:.2f} images/sec"
                    )

                    print(
                        f"  ETA: "
                        f"{eta_seconds / 60:.1f} min"
                    )

    # -----------------------------------------------------------------
    # Save final partial shard
    # -----------------------------------------------------------------

    if shard_embeddings:

        embeddings_array = np.stack(
            shard_embeddings,
            axis=0,
        )

        save_shard(
            output_dir=output_dir,
            shard_index=shard_count,
            embeddings=embeddings_array,
            rows=shard_rows,
        )

        shard_count += 1

    # -----------------------------------------------------------------
    # Final progress
    # -----------------------------------------------------------------

    save_progress(
        output_dir=output_dir,
        processed=total,
        shard_index=shard_count,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print("\n")
    print("=" * 78)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 78)

    print(
        f"Images processed : {total:,}"
    )

    print(
        f"Embedding size   : {embedding_dim}"
    )

    print(
        f"Shards created   : {shard_count}"
    )

    print(
        f"Time             : "
        f"{elapsed / 60:.1f} minutes"
    )

    if elapsed > 0:

        print(
            f"Average speed    : "
            f"{total / elapsed:.2f} images/sec"
        )

    print(
        f"Output directory : {output_dir}"
    )

    print("=" * 78)


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Extract DINOv2 embeddings from "
            "FER benchmark images."
        )
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=str(DEFAULT_METADATA),
        help="Path to metadata.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader workers",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Required for Windows multiprocessing.
    torch.multiprocessing.freeze_support()

    args = parse_args()

    extract_embeddings(
        metadata_path=Path(
            args.metadata
        ),
        output_dir=Path(
            args.output
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
