"""Dataset Inspection Experiment Script.

Inspects an image dataset for:
- Readability and decoding integrity
- Image dimensions and resolutions
- Exact binary duplicate detection (SHA-256)
- Blur score (Laplacian variance)
- Mean grayscale brightness
- Feature richness (SIFT keypoints count)

Outputs results to: outputs/reports/dataset_inspection.csv
Prints a comprehensive diagnostic summary to the console.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Add project root to sys.path to enable relative module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_INPUT_DIR,
    OUTPUTS_REPORTS_DIR,
    ensure_directories_exist,
)
from modules.image_loader import (
    discover_image_files,
    compute_file_hash,
    load_image_safely,
)
from modules.image_quality import assess_image_quality


CSV_COLUMNS = [
    "filename",
    "relative_path",
    "extension",
    "width",
    "height",
    "file_size_kb",
    "readable",
    "duplicate_hash",
    "blur_score",
    "mean_brightness",
    "sift_keypoints",
    "inspection_status",
]


def inspect_dataset(
    dataset_dir: Path | str,
    output_csv_path: Path | str,
) -> pd.DataFrame:
    """Executes the full inspection workflow on a target image directory.

    Args:
        dataset_dir: Directory containing images to inspect.
        output_csv_path: Filepath where the resulting CSV will be saved.

    Returns:
        pandas DataFrame containing inspection results.
    """
    dataset_path = Path(dataset_dir).resolve()
    csv_path = Path(output_csv_path).resolve()

    # Ensure parent output directories exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATASET INSPECTION PIPELINE")
    print("=" * 70)
    print(f"Target Dataset Directory : {dataset_path}")
    print(f"Output Report Destination: {csv_path}")

    # Check directory existence
    if not dataset_path.exists():
        print(f"\n[ERROR] Target directory does not exist: {dataset_path}")
        empty_df = pd.DataFrame(columns=CSV_COLUMNS)
        empty_df.to_csv(csv_path, index=False)
        return empty_df

    # Discover images
    image_paths = discover_image_files(dataset_path)
    total_discovered = len(image_paths)
    print(f"Discovered {total_discovered} image file(s) with supported extensions.")

    if total_discovered == 0:
        print("[WARNING] No supported images found in target directory.")
        empty_df = pd.DataFrame(columns=CSV_COLUMNS)
        empty_df.to_csv(csv_path, index=False)
        return empty_df

    records: List[Dict[str, Any]] = []

    print("\nProcessing images and extracting metrics...")
    for idx, img_path in enumerate(image_paths, 1):
        filename = img_path.name
        # Compute relative path from PROJECT_ROOT for portability
        try:
            rel_path = str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            rel_path = str(img_path).replace("\\", "/")

        # 1. Compute binary hash
        try:
            dup_hash = compute_file_hash(img_path)
        except Exception as exc:
            dup_hash = f"HASH_ERROR: {str(exc)}"

        # 2. Safe image loading & metadata
        img_array, load_meta = load_image_safely(img_path)

        # 3. Quality & feature assessment
        if load_meta["readable"] and img_array is not None:
            quality_meta = assess_image_quality(img_array)
            inspection_status = (
                "OK" if quality_meta["quality_status"] == "SUCCESS"
                else quality_meta["quality_status"]
            )
        else:
            quality_meta = {
                "blur_score": None,
                "mean_brightness": None,
                "sift_keypoints": None,
            }
            inspection_status = f"FAILED: {load_meta['error_message']}"

        record = {
            "filename": filename,
            "relative_path": rel_path,
            "extension": load_meta["extension"],
            "width": load_meta["width"],
            "height": load_meta["height"],
            "file_size_kb": load_meta["file_size_kb"],
            "readable": load_meta["readable"],
            "duplicate_hash": dup_hash,
            "blur_score": quality_meta["blur_score"],
            "mean_brightness": quality_meta["mean_brightness"],
            "sift_keypoints": quality_meta["sift_keypoints"],
            "inspection_status": inspection_status,
        }
        records.append(record)

        if idx % 10 == 0 or idx == total_discovered:
            print(f"  [{idx}/{total_discovered}] {filename} -> status: {inspection_status}")

    # Build DataFrame
    df = pd.DataFrame(records, columns=CSV_COLUMNS)

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Dataset inspection report written to: {csv_path}")

    # Print Diagnostic Summary
    print_inspection_summary(df)

    return df


def print_inspection_summary(df: pd.DataFrame) -> None:
    """Formats and prints the statistical and diagnostic summary."""
    total_images = len(df)
    readable_count = int(df["readable"].sum()) if total_images > 0 else 0
    unreadable_count = total_images - readable_count

    print("\n" + "=" * 70)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 70)
    print(f"Total images discovered       : {total_images}")
    print(f"Successfully readable images   : {readable_count}")
    print(f"Unreadable images              : {unreadable_count}")

    if total_images == 0:
        print("No image data to summarize.")
        print("=" * 70)
        return

    # Exact duplicate detection
    hash_counts = df["duplicate_hash"].value_counts()
    duplicate_groups = hash_counts[hash_counts > 1]
    num_duplicate_groups = len(duplicate_groups)
    total_duplicate_files = duplicate_groups.sum() if num_duplicate_groups > 0 else 0

    print(f"\nExact duplicate groups         : {num_duplicate_groups}")
    if num_duplicate_groups > 0:
        print(f"Total duplicate files involved : {total_duplicate_files}")
        for h, count in duplicate_groups.items():
            dup_filenames = df[df["duplicate_hash"] == h]["filename"].tolist()
            print(f"  - Hash {h[:12]}... : {count} files -> {dup_filenames}")
    else:
        print("  (No exact binary duplicates detected)")

    # Image resolutions
    readable_df = df[df["readable"] == True]
    print("\nImage Resolutions:")
    if not readable_df.empty:
        resolutions = readable_df.apply(
            lambda r: f"{int(r['width'])} x {int(r['height'])}", axis=1
        ).value_counts()
        for res, count in resolutions.items():
            print(f"  - {res}: {count} image(s)")
    else:
        print("  (None readable)")

    # Averages for readable images
    print("\nQuality & Feature Metrics (Readable Images):")
    if not readable_df.empty:
        avg_blur = readable_df["blur_score"].dropna().mean()
        avg_brightness = readable_df["mean_brightness"].dropna().mean()
        avg_sift = readable_df["sift_keypoints"].dropna().mean()

        print(f"  - Average blur score (Laplacian var) : {avg_blur:.2f}")
        print(f"  - Average brightness (grayscale)     : {avg_brightness:.2f} / 255.0")
        print(f"  - Average SIFT keypoints             : {avg_sift:.1f}")
    else:
        print("  - Average blur score                 : N/A")
        print("  - Average brightness                 : N/A")
        print("  - Average SIFT keypoints             : N/A")

    print("=" * 70)


def resolve_default_dataset_dir() -> Path:
    """Finds the most suitable dataset directory.

    Checks data/input/cat_rgb/raw/ first as specified by the user.
    If not found, checks known existing directories in data/input to provide helpful context.
    """
    default_cat_raw = PROJECT_ROOT / "data" / "input" / "cat_rgb" / "raw"
    if default_cat_raw.exists():
        return default_cat_raw

    # Fallback candidates for inspection
    dog_rgb = PROJECT_ROOT / "data" / "input" / "Data" / "dog_rgb"
    if dog_rgb.exists():
        return dog_rgb

    data_dir = PROJECT_ROOT / "data" / "input" / "Data"
    if data_dir.exists():
        return data_dir

    return default_cat_raw


def main():
    parser = argparse.ArgumentParser(
        description="Inspect multi-view image dataset for 3D reconstruction pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the directory containing dataset images (defaults to data/input/cat_rgb/raw/)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV report (defaults to outputs/reports/dataset_inspection.csv)",
    )

    args = parser.parse_args()

    ensure_directories_exist()

    if args.data_dir:
        target_dir = Path(args.data_dir)
        if not target_dir.is_absolute():
            target_dir = PROJECT_ROOT / target_dir
    else:
        # Default specified location
        cat_path = PROJECT_ROOT / "data" / "input" / "cat_rgb" / "raw"
        if cat_path.exists():
            target_dir = cat_path
        else:
            # Check fallback
            target_dir = resolve_default_dataset_dir()
            print(f"[INFO] '{cat_path}' was not found. Using detected dataset at: '{target_dir}'")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        if not output_csv.is_absolute():
            output_csv = PROJECT_ROOT / output_csv
    else:
        output_csv = OUTPUTS_REPORTS_DIR / "dataset_inspection.csv"

    inspect_dataset(target_dir, output_csv)


if __name__ == "__main__":
    main()
