"""SIFT Feature Matching and RANSAC Epipolar Geometric Verification Module.

Provides modular tools for:
- SIFT feature detection (unmasked and masked modes).
- Brute-Force KNN descriptor matching with Lowe's ratio test.
- Robust Fundamental Matrix estimation with RANSAC geometric verification.
- Inlier/outlier match visualization with side-by-side color-coded lines.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np


def detect_sift_features(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_features: int = 0,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Detects SIFT keypoints and computes 128-dimensional descriptors.

    Args:
        image_bgr: Input BGR image array.
        mask: Optional binary mask (uint8, 0 and 255). If provided, SIFT restricts
              detection exclusively to the foreground object pixels.
        n_features: Maximum number of features to retain (0 = all detected).

    Returns:
        Tuple of (list_of_KeyPoints, descriptors_array).
        If detection fails or no keypoints are found, descriptors is None.
    """
    if image_bgr is None or image_bgr.size == 0:
        return [], None

    sift = cv2.SIFT_create(nfeatures=n_features)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        # Ensure mask is single channel uint8
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask > 0).astype(np.uint8) * 255
        keypoints, descriptors = sift.detectAndCompute(gray, mask=mask_binary)
    else:
        keypoints, descriptors = sift.detectAndCompute(gray, mask=None)

    return keypoints, descriptors


def match_descriptors(
    desc1: Optional[np.ndarray],
    desc2: Optional[np.ndarray],
    ratio_threshold: float = 0.75,
) -> Tuple[int, List[cv2.DMatch]]:
    """Matches descriptors using Brute-Force L2 matcher and Lowe's ratio test.

    Lowe's ratio test filters out ambiguous matches by requiring the closest match distance
    to be substantially smaller than the second closest match:
        m.distance < ratio_threshold * n.distance

    Args:
        desc1: Descriptors from image 1 (N x 128 float32).
        desc2: Descriptors from image 2 (M x 128 float32).
        ratio_threshold: Lowe's ratio test factor (default 0.75).

    Returns:
        Tuple of (raw_matches_count, list_of_good_DMatches).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return 0, []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches: List[cv2.DMatch] = []
    for match_pair in raw_knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return len(raw_knn_matches), good_matches


def estimate_fundamental_matrix(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    reproj_threshold: float = 3.0,
    confidence: float = 0.99,
) -> Tuple[int, float, Optional[np.ndarray], np.ndarray, str]:
    """Estimates the Fundamental Matrix using RANSAC to geometrically verify matches.

    Enforces the epipolar constraint: x2^T * F * x1 = 0.
    Identifies inliers (geometrically consistent matches) and outliers.

    Args:
        kp1: Keypoints from image 1.
        kp2: Keypoints from image 2.
        good_matches: Matches passing Lowe's ratio test.
        reproj_threshold: Maximum distance from point to epipolar line in pixels (default 3.0).
        confidence: RANSAC confidence level (default 0.99).

    Returns:
        Tuple of:
            - inliers_count (int): Number of geometrically verified inliers.
            - inlier_ratio (float): inliers_count / len(good_matches).
            - F_matrix (np.ndarray or None): 3x3 Fundamental Matrix.
            - inliers_mask (np.ndarray): 1D boolean array (True for inlier, False for outlier).
            - status (str): Status description ('COMPUTED', 'INSUFFICIENT_MATCHES', or 'FAILED').
    """
    num_matches = len(good_matches)
    if num_matches < 8:
        return 0, 0.0, None, np.zeros((num_matches,), dtype=bool), "INSUFFICIENT_MATCHES"

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
        )

        if F is None or mask is None:
            return 0, 0.0, None, np.zeros((num_matches,), dtype=bool), "DEGENERATE_CONFIGURATION"

        inliers_mask = (mask.ravel() == 1)
        inliers_count = int(np.sum(inliers_mask))
        inlier_ratio = round(inliers_count / num_matches, 4) if num_matches > 0 else 0.0

        return inliers_count, inlier_ratio, F, inliers_mask, "COMPUTED"

    except Exception as exc:
        return 0, 0.0, None, np.zeros((num_matches,), dtype=bool), f"ERROR: {str(exc)}"


def render_matches_visualization(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    inliers_mask: np.ndarray,
    label1: str = "Image 1",
    label2: str = "Image 2",
    mode_label: str = "MASKED",
    display_scale: float = 0.5,
) -> np.ndarray:
    """Renders a side-by-side match visualization with color-coded correspondence lines.

    - Green lines: Geometrically verified RANSAC inliers.
    - Red lines: Outliers rejected by epipolar geometry.

    Args:
        img1_bgr: First image array.
        img2_bgr: Second image array.
        kp1: Keypoints from image 1.
        kp2: Keypoints from image 2.
        good_matches: Lowe-filtered matches.
        inliers_mask: Boolean mask indicating which good matches are RANSAC inliers.
        label1: Text name for image 1.
        label2: Text name for image 2.
        mode_label: Mode description ('MASKED SIFT' or 'UNMASKED SIFT').
        display_scale: Downscale factor for the composite visualization.

    Returns:
        Rendered BGR image array.
    """
    # Downscale base images for balanced display
    h1, w1 = img1_bgr.shape[:2]
    h2, w2 = img2_bgr.shape[:2]

    sw1, sh1 = int(w1 * display_scale), int(h1 * display_scale)
    sw2, sh2 = int(w2 * display_scale), int(h2 * display_scale)

    small_img1 = cv2.resize(img1_bgr, (sw1, sh1), interpolation=cv2.INTER_AREA)
    small_img2 = cv2.resize(img2_bgr, (sw2, sh2), interpolation=cv2.INTER_AREA)

    header_h = 70
    canvas_w = sw1 + sw2
    canvas_h = max(sh1, sh2) + header_h

    # Dark background
    canvas = np.full((canvas_h, canvas_w, 3), (32, 34, 40), dtype=np.uint8)
    canvas[header_h : header_h + sh1, :sw1] = small_img1
    canvas[header_h : header_h + sh2, sw1 : sw1 + sw2] = small_img2

    # Draw separator line between images
    cv2.line(canvas, (sw1, header_h), (sw1, canvas_h), (80, 85, 95), 2, cv2.LINE_AA)

    # Header banner info
    inliers_count = int(np.sum(inliers_mask)) if len(inliers_mask) > 0 else 0
    total_good = len(good_matches)
    ratio_pct = (inliers_count / total_good * 100.0) if total_good > 0 else 0.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Left and Right image filenames
    cv2.putText(canvas, f"{label1}", (20, 30), font, 0.65, (220, 225, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{label2}", (sw1 + 20, 30), font, 0.65, (220, 225, 235), 2, cv2.LINE_AA)

    # Center statistics
    stats_text = (
        f"Mode: {mode_label} | Good Matches: {total_good} | "
        f"RANSAC Inliers: {inliers_count} ({ratio_pct:.1f}%)"
    )
    cv2.putText(canvas, stats_text, (20, 56), font, 0.52, (200, 230, 255), 1, cv2.LINE_AA)

    # Legend on right side of banner
    legend_text = "Legend: Green = RANSAC Inlier | Red = Outlier"
    leg_size, _ = cv2.getTextSize(legend_text, font, 0.48, 1)
    leg_x = canvas_w - leg_size[0] - 20
    cv2.putText(canvas, legend_text, (leg_x, 56), font, 0.48, (160, 220, 160), 1, cv2.LINE_AA)

    # Draw divider below banner
    cv2.line(canvas, (10, header_h - 6), (canvas_w - 10, header_h - 6), (70, 75, 85), 1, cv2.LINE_AA)

    if total_good == 0:
        cv2.putText(
            canvas,
            "NO GOOD MATCHES FOUND",
            (canvas_w // 2 - 140, canvas_h // 2),
            font,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    # Draw correspondence lines
    # Pass 1: Draw outliers in red (thin)
    for idx, match in enumerate(good_matches):
        is_inlier = inliers_mask[idx] if idx < len(inliers_mask) else False
        if not is_inlier:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt

            x1 = int(round(pt1[0] * display_scale))
            y1 = int(round(pt1[1] * display_scale)) + header_h
            x2 = int(round(pt2[0] * display_scale)) + sw1
            y2 = int(round(pt2[1] * display_scale)) + header_h

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 200), 1, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), 2, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x2, y2), 2, (0, 0, 220), -1, cv2.LINE_AA)

    # Pass 2: Draw inliers in green on top (bolder)
    for idx, match in enumerate(good_matches):
        is_inlier = inliers_mask[idx] if idx < len(inliers_mask) else False
        if is_inlier:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt

            x1 = int(round(pt1[0] * display_scale))
            y1 = int(round(pt1[1] * display_scale)) + header_h
            x2 = int(round(pt2[0] * display_scale)) + sw1
            y2 = int(round(pt2[1] * display_scale)) + header_h

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 225, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), 3, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x2, y2), 3, (0, 255, 120), -1, cv2.LINE_AA)

    return canvas
