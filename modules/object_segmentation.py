"""Object Segmentation and Mask Generation Module.

Implements adaptive HSV color segmentation, morphological cleanup,
connected-component analysis, GrabCut refinement evaluation, and visual overlay
rendering for multi-view object datasets (THU-MVS Dog_RGB).
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import cv2
import numpy as np


def estimate_adaptive_background_hsv(
    hsv_image: np.ndarray,
    margin: int = 50,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Dict[str, float]]:
    """Estimates adaptive HSV threshold bounds for the background by sampling image borders.

    Avoids using static hardcoded thresholds that fail on darker or brighter views.

    Args:
        hsv_image: 3D uint8 array in OpenCV HSV color space (H: 0-179, S: 0-255, V: 0-255).
        margin: Width in pixels of border margin strips to sample.

    Returns:
        Tuple of:
            - lower_bound: (h_min, s_min, v_min)
            - upper_bound: (h_max, s_max, v_max)
            - stats: dictionary of estimated background HSV statistics.
    """
    height, width = hsv_image.shape[:2]
    # Sample top, left, and right margins (exclude bottom to avoid touching turntable/paws)
    top_strip = hsv_image[:margin, :, :].reshape(-1, 3)
    left_strip = hsv_image[:, :margin, :].reshape(-1, 3)
    right_strip = hsv_image[:, -margin:, :].reshape(-1, 3)
    samples = np.concatenate([top_strip, left_strip, right_strip], axis=0)

    med_h = float(np.median(samples[:, 0]))
    med_s = float(np.median(samples[:, 1]))
    med_v = float(np.median(samples[:, 2]))

    # Blue backdrop typically centers around H=114-116.
    # Allow +/- 20 degrees in Hue around observed median
    h_min = int(max(90, med_h - 20))
    h_max = int(min(140, med_h + 20))

    # Background saturation is high; set lower bound safely below 2nd percentile
    s_min = int(max(70, np.percentile(samples[:, 1], 2) - 30))
    s_max = 255

    # Background brightness adjusts adaptively for dimmer lighting in late shots
    v_min = int(max(25, np.percentile(samples[:, 2], 1) - 20))
    v_max = 255

    stats = {
        "median_hue": round(med_h, 1),
        "median_saturation": round(med_s, 1),
        "median_value": round(med_v, 1),
        "h_min": h_min,
        "h_max": h_max,
        "s_min": s_min,
        "v_min": v_min,
    }

    return (h_min, s_min, v_min), (h_max, s_max, v_max), stats


def fill_mask_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fills interior enclosed holes inside the foreground mask using morphological flood fill."""
    im_floodfill = binary_mask.copy()
    h, w = binary_mask.shape[:2]
    # Floodfill mask needs to be 2 pixels wider and taller
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, fill_mask, (0, 0), 255)
    # Invert floodfilled image to obtain only the holes
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine original mask with holes
    filled = binary_mask | im_floodfill_inv
    return filled


def segment_object_hsv(
    image_bgr: np.ndarray,
    close_kernel_size: int = 15,
    open_kernel_size: int = 9,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Segments the foreground object using adaptive HSV thresholding and morphology.

    Pipeline:
    1. Convert to HSV color space.
    2. Adaptively estimate blue background bounds from image border samples.
    3. Generate background mask and invert to get candidate foreground.
    4. Apply morphological closing to seal internal shadow clefts/textures.
    5. Apply morphological opening to eliminate isolated noise flecks.
    6. Extract connected components and isolate the single largest foreground component.
    7. Fill any remaining enclosed internal holes.

    Args:
        image_bgr: Input BGR image array.
        close_kernel_size: Kernel dimension for morphological closing.
        open_kernel_size: Kernel dimension for morphological opening.

    Returns:
        Tuple of (binary_mask_uint8, metrics_dict)
    """
    height, width = image_bgr.shape[:2]
    total_pixels = height * width

    metrics: Dict[str, Any] = {
        "image_width": width,
        "image_height": height,
        "foreground_pixel_ratio": 0.0,
        "background_pixel_ratio": 1.0,
        "largest_component_ratio": 0.0,
        "total_components": 0,
        "segmentation_status": "PENDING",
        "error_message": "",
        "adaptive_bg_stats": {},
    }

    if total_pixels == 0:
        metrics["segmentation_status"] = "FAILED"
        metrics["error_message"] = "Empty image dimensions (0x0)."
        return np.zeros((0, 0), dtype=np.uint8), metrics

    try:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_bg, upper_bg, bg_stats = estimate_adaptive_background_hsv(hsv)
        metrics["adaptive_bg_stats"] = bg_stats

        # 1. Background thresholding
        bg_mask = cv2.inRange(
            hsv,
            np.array(lower_bg, dtype=np.uint8),
            np.array(upper_bg, dtype=np.uint8),
        )

        # 2. Invert: candidate foreground
        fg_candidate = cv2.bitwise_not(bg_mask)

        # 3. Morphological operations
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        closed = cv2.morphologyEx(fg_candidate, cv2.MORPH_CLOSE, k_close)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

        # 4. Connected-component analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        metrics["total_components"] = int(num_labels - 1)  # exclude background label 0

        if num_labels <= 1:
            metrics["segmentation_status"] = "FAILED"
            metrics["error_message"] = "No foreground connected components detected."
            return np.zeros((height, width), dtype=np.uint8), metrics

        # Isolate the largest foreground component
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas))
        largest_label = largest_idx + 1
        largest_area = int(areas[largest_idx])

        # Generate isolated component mask
        component_mask = (labels == largest_label).astype(np.uint8) * 255

        # 5. Fill internal holes
        final_mask = fill_mask_holes(component_mask)

        # Calculate final pixel metrics
        fg_pixels = int(np.count_nonzero(final_mask))
        fg_ratio = round(fg_pixels / total_pixels, 4)
        bg_ratio = round(1.0 - fg_ratio, 4)
        largest_comp_ratio = round(largest_area / total_pixels, 4)

        metrics["foreground_pixel_ratio"] = fg_ratio
        metrics["background_pixel_ratio"] = bg_ratio
        metrics["largest_component_ratio"] = largest_comp_ratio

        # Validation checks
        if fg_ratio < 0.10:
            metrics["segmentation_status"] = "FLAGGED_TOO_SMALL"
            metrics["error_message"] = (
                f"Foreground area ratio ({fg_ratio:.3f}) is abnormally small; possible object truncation."
            )
        elif fg_ratio > 0.50:
            metrics["segmentation_status"] = "FLAGGED_TOO_LARGE"
            metrics["error_message"] = (
                f"Foreground area ratio ({fg_ratio:.3f}) is abnormally large; possible background inclusion."
            )
        else:
            metrics["segmentation_status"] = "SUCCESS"
            metrics["error_message"] = ""

        return final_mask, metrics

    except Exception as exc:
        metrics["segmentation_status"] = "FAILED"
        metrics["error_message"] = f"Segmentation exception: {type(exc).__name__}: {str(exc)}"
        return np.zeros((height, width), dtype=np.uint8), metrics


def refine_mask_grabcut(
    image_bgr: np.ndarray,
    initial_mask: np.ndarray,
    iterations: int = 3,
    downscale_factor: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Evaluates GrabCut refinement on top of an initial binary mask.

    Args:
        image_bgr: Full-resolution BGR image.
        initial_mask: Binary mask (0 or 255) from HSV segmentation.
        iterations: Number of GrabCut GMM optimization iterations.
        downscale_factor: Factor to downscale image for efficient optimization.

    Returns:
        Tuple of (refined_mask_uint8, iou_with_initial_mask)
    """
    h, w = image_bgr.shape[:2]
    # Build GrabCut 4-class mask
    # GC_BGD (0): Definite background
    # GC_PR_BGD (2): Probable background
    # GC_PR_FGD (3): Probable foreground
    # GC_FGD (1): Definite foreground
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[initial_mask == 0] = cv2.GC_BGD
    gc_mask[initial_mask == 255] = cv2.GC_PR_FGD

    # Create definite foreground core via erosion
    k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    core_fg = cv2.erode(initial_mask, k_core)
    gc_mask[core_fg == 255] = cv2.GC_FGD

    # Downscale for performance
    small_h = int(h * downscale_factor)
    small_w = int(w * downscale_factor)
    small_img = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_gc_mask = cv2.resize(gc_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        small_img,
        small_gc_mask,
        None,
        bgd_model,
        fgd_model,
        iterCount=iterations,
        mode=cv2.GC_INIT_WITH_MASK,
    )

    refined_small = np.where(
        (small_gc_mask == cv2.GC_FGD) | (small_gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    refined_mask = cv2.resize(refined_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Compute IoU between initial and GrabCut masks
    intersection = np.logical_and(initial_mask == 255, refined_mask == 255).sum()
    union = np.logical_or(initial_mask == 255, refined_mask == 255).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    return refined_mask, round(iou, 4)


def create_mask_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    contour_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow BGR
    tint_color: Tuple[int, int, int] = (0, 220, 100),     # Emerald green BGR
    tint_alpha: float = 0.22,
    contour_thickness: int = 3,
) -> np.ndarray:
    """Renders a visual overlay showing the mask boundary and a translucent tint.

    Args:
        image_bgr: Original image array.
        binary_mask: Binary mask array (0 or 255).
        contour_color: BGR color for boundary contour.
        tint_color: BGR color for foreground translucent tint.
        tint_alpha: Blend weight for the foreground tint.
        contour_thickness: Pixel thickness of boundary contour line.

    Returns:
        3-channel BGR image with rendered overlay.
    """
    overlay = image_bgr.copy()

    # Apply translucent foreground tint
    if tint_alpha > 0.0:
        tint = np.zeros_like(image_bgr)
        tint[binary_mask == 255] = tint_color
        overlay = cv2.addWeighted(overlay, 1.0, tint, tint_alpha, 0)

    # Draw smooth boundary contour
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, contour_color, contour_thickness, cv2.LINE_AA)

    return overlay
