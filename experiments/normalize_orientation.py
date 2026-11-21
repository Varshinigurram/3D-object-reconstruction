"""
experiments/normalize_orientation.py
=====================================
Non-Destructive Orientation Normalization Script.

Part of the "Adaptive Multi-View 3D Object Reconstruction and Geometric Analysis" project.

Objectives:
1. Preserve every original raw RGB image in `data/input/Data/dog_rgb/`.
2. Preserve every original binary mask in `data/processed/dog_rgb/masks/`.
3. Normalize all 72 RGB images into `data/processed/dog_rgb/oriented_images/`.
4. Normalize all 72 binary masks into `data/processed/dog_rgb/oriented_masks/`.
5. Establish a unified portrait coordinate convention (1704 x 2272) across all views.
6. Apply exact geometric transformations:
   - For im_0001 to im_0034 and im_0036 to im_0072: Identity (0 deg rotation).
   - For im_0035: 90 degrees Clockwise rotation (cv2.ROTATE_90_CLOCKWISE).
7. Apply identical transformation to RGB image and mask pairs.
8. Use lossless representation (PNG) for working images and masks to prevent recompression artifacts.
9. Preserve strict binary values in masks ({0, 255}) with zero interpolation artifacts.
"""

import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DOG_RGB_RAW_DIR,
    DOG_RGB_MASKS_DIR,
    DOG_RGB_ORIENTED_IMAGES_DIR,
    DOG_RGB_ORIENTED_MASKS_DIR,
)
from modules.image_loader import compute_file_hash


def verify_original_files_exist() -> Tuple[List[Path], List[Path]]:
    """Verify that all 72 raw images and masks exist and are readable."""
    if not DOG_RGB_RAW_DIR.exists():
        raise FileNotFoundError(f"Raw image directory not found: {DOG_RGB_RAW_DIR}")
    if not DOG_RGB_MASKS_DIR.exists():
        raise FileNotFoundError(f"Mask directory not found: {DOG_RGB_MASKS_DIR}")

    raw_images = sorted([p for p in DOG_RGB_RAW_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]])
    masks = sorted([p for p in DOG_RGB_MASKS_DIR.iterdir() if p.suffix.lower() == ".png"])

    if len(raw_images) != 72:
        raise ValueError(f"Expected exactly 72 raw images, found {len(raw_images)}")
    if len(masks) != 72:
        raise ValueError(f"Expected exactly 72 masks, found {len(masks)}")

    return raw_images, masks


def normalize_dataset() -> Dict[str, Any]:
    """Execute the non-destructive orientation normalization pipeline."""
    print("=" * 70)
    print("STARTING NON-DESTRUCTIVE ORIENTATION NORMALIZATION")
    print("=" * 70)

    raw_images, masks = verify_original_files_exist()
    
    # Record pre-normalization hashes to guarantee non-destructive execution
    print("\n[STEP 1] Recording pre-normalization hashes of original files...")
    pre_raw_hashes = {img.name: compute_file_hash(img) for img in raw_images}
    pre_mask_hashes = {m.name: compute_file_hash(m) for m in masks}
    print(f"  -> Recorded hashes for {len(pre_raw_hashes)} raw images and {len(pre_mask_hashes)} masks.")

    # Create destination directories safely
    DOG_RGB_ORIENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DOG_RGB_ORIENTED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 2] Destination directories initialized:")
    print(f"  -> Oriented images: {DOG_RGB_ORIENTED_IMAGES_DIR}")
    print(f"  -> Oriented masks:  {DOG_RGB_ORIENTED_MASKS_DIR}")

    normalization_log: List[Dict[str, Any]] = []

    # Process all 72 images
    print("\n[STEP 3] Normalizing images and masks...")
    for idx, raw_path in enumerate(raw_images, start=1):
        stem = raw_path.stem
        mask_path = DOG_RGB_MASKS_DIR / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing corresponding mask: {mask_path}")

        # Load RGB image (OpenCV handles EXIF orientation tag 6 automatically)
        img_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Failed to decode RGB image: {raw_path}")

        # Load mask as single-channel grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to decode mask: {mask_path}")

        orig_img_shape = img_bgr.shape[:2]
        orig_mask_shape = mask.shape[:2]

        # Verify initial spatial alignment
        if orig_img_shape != orig_mask_shape:
            raise ValueError(
                f"Dimension mismatch between image {raw_path.name} {orig_img_shape} "
                f"and mask {mask_path.name} {orig_mask_shape}"
            )

        # Determine geometric transformation
        if stem == "im_0035":
            transformation = "ROTATE_90_CLOCKWISE"
            # Apply exact 90 deg clockwise rotation (transposition + horizontal flip, zero interpolation)
            norm_img = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            norm_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        else:
            transformation = "IDENTITY"
            norm_img = img_bgr
            norm_mask = mask

        norm_h, norm_w = norm_img.shape[:2]
        norm_mh, norm_mw = norm_mask.shape[:2]

        # Rigorous assertions
        assert (norm_w, norm_h) == (1704, 2272), (
            f"Normalized image {stem} dimension is {(norm_w, norm_h)}, expected (1704, 2272)"
        )
        assert (norm_mw, norm_mh) == (1704, 2272), (
            f"Normalized mask {stem} dimension is {(norm_mw, norm_mh)}, expected (1704, 2272)"
        )

        # Verify mask binary values remain strictly in {0, 255}
        unique_vals = np.unique(norm_mask)
        if not np.all(np.isin(unique_vals, [0, 255])):
            raise ValueError(f"Mask {stem} contains non-binary values: {unique_vals}")

        # Destination paths (lossless PNG representation)
        dest_img_path = DOG_RGB_ORIENTED_IMAGES_DIR / f"{stem}.png"
        dest_mask_path = DOG_RGB_ORIENTED_MASKS_DIR / f"{stem}.png"

        # Save using lossless PNG compression (compression level 3 for fast balanced write)
        cv2.imwrite(str(dest_img_path), norm_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(dest_mask_path), norm_mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        normalization_log.append({
            "stem": stem,
            "raw_file": raw_path.name,
            "original_dimensions": f"{orig_img_shape[1]}x{orig_img_shape[0]}",
            "transformation": transformation,
            "normalized_dimensions": f"{norm_w}x{norm_h}",
            "mask_unique_values": unique_vals.tolist(),
            "output_image": dest_img_path.name,
            "output_mask": dest_mask_path.name,
        })

        if idx % 12 == 0 or stem == "im_0035":
            print(f"  [{idx:02d}/72] Processed {stem}: transformation={transformation}, output={norm_w}x{norm_h}")

    # Step 4: Verify original files were NOT touched
    print("\n[STEP 4] Verifying original file immutability via hash verification...")
    post_raw_hashes = {img.name: compute_file_hash(img) for img in raw_images}
    post_mask_hashes = {m.name: compute_file_hash(m) for m in masks}

    for name, pre_h in pre_raw_hashes.items():
        if post_raw_hashes[name] != pre_h:
            raise RuntimeError(f"CRITICAL: Raw image {name} was modified during normalization!")

    for name, pre_h in pre_mask_hashes.items():
        if post_mask_hashes[name] != pre_h:
            raise RuntimeError(f"CRITICAL: Mask {name} was modified during normalization!")

    print("  -> Immutability confirmed: 100% of original raw images and masks have identical SHA-256 hashes.")

    print("\n" + "=" * 70)
    print("ORIENTATION NORMALIZATION SUCCESSFULLY COMPLETED")
    print(f"  Total normalized images: {len(normalization_log)}")
    print(f"  Target directory: {DOG_RGB_ORIENTED_IMAGES_DIR}")
    print("=" * 70)

    return {
        "status": "SUCCESS",
        "total_processed": len(normalization_log),
        "log": normalization_log
    }


if __name__ == "__main__":
    normalize_dataset()
