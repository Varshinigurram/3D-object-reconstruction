"""Module for computing image quality metrics and feature richness.

Calculates:
- Laplacian variance (blur / sharpness score).
- Mean grayscale brightness.
- SIFT keypoint count (feature richness).

Includes comprehensive error handling to ensure pipeline stability during dataset inspection.
"""

from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np


def convert_to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Converts a BGR or RGB image array to single-channel 8-bit grayscale.

    Args:
        image_bgr: Input image array (2D or 3D).

    Returns:
        2D uint8 numpy array representing grayscale image.
    """
    if image_bgr.ndim == 2:
        return image_bgr
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image_bgr.shape}")


def calculate_blur_score(image: np.ndarray) -> float:
    """Computes the blur score using the variance of the Laplacian.

    A higher variance indicates sharp, high-frequency details (in-focus).
    A lower variance suggests an out-of-focus or motion-blurred image.

    Args:
        image: Input image array (grayscale or BGR).

    Returns:
        Float variance of the Laplacian filter response.
    """
    gray = convert_to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return round(variance, 2)


def calculate_mean_brightness(image: np.ndarray) -> float:
    """Calculates the average pixel brightness across the grayscale representation.

    Args:
        image: Input image array (grayscale or BGR).

    Returns:
        Float mean pixel value in the range [0.0, 255.0].
    """
    gray = convert_to_grayscale(image)
    brightness = float(np.mean(gray))
    return round(brightness, 2)


def extract_sift_keypoints_count(
    image: np.ndarray,
    n_features: int = 0,
) -> int:
    """Detects SIFT keypoints and returns the total number detected.

    Indicates the feature richness of the viewpoint for epipolar matching.

    Args:
        image: Input image array (grayscale or BGR).
        n_features: Number of best features to retain (0 retains all detected).

    Returns:
        Integer count of detected SIFT keypoints.
    """
    gray = convert_to_grayscale(image)
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints = sift.detect(gray, None)
    return int(len(keypoints)) if keypoints is not None else 0


def assess_image_quality(
    image: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Combines all quality metric computations with graceful exception handling.

    Args:
        image: Decoded numpy image array, or None if unreadable.

    Returns:
        Dictionary containing:
            - blur_score: float or None
            - mean_brightness: float or None
            - sift_keypoints: int or None
            - quality_status: 'SUCCESS' or descriptive error message.
    """
    results: Dict[str, Any] = {
        "blur_score": None,
        "mean_brightness": None,
        "sift_keypoints": None,
        "quality_status": "PENDING",
    }

    if image is None:
        results["quality_status"] = "SKIPPED_UNREADABLE"
        return results

    try:
        results["blur_score"] = calculate_blur_score(image)
    except Exception as exc:
        results["blur_score"] = None
        results["quality_status"] = f"BLUR_ERROR: {str(exc)}"

    try:
        results["mean_brightness"] = calculate_mean_brightness(image)
    except Exception as exc:
        results["mean_brightness"] = None
        results["quality_status"] = f"BRIGHTNESS_ERROR: {str(exc)}"

    try:
        results["sift_keypoints"] = extract_sift_keypoints_count(image)
    except Exception as exc:
        results["sift_keypoints"] = None
        results["quality_status"] = f"SIFT_ERROR: {str(exc)}"

    if results["quality_status"] == "PENDING":
        results["quality_status"] = "SUCCESS"

    return results
