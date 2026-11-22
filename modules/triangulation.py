"""
modules/triangulation.py
========================
Two-View Linear Triangulation and Geometric Quality Filtering Module.

Part of the "Adaptive Multi-View 3D Object Reconstruction and Geometric Analysis" project.

Core capabilities:
1. Direct Linear Transformation (DLT) triangulation from projection matrices P1 and P2.
2. Robust homogeneous scale (W) validation with explicit near-zero rejection and signed division.
3. Two-camera depth evaluation (Z1 > 0, Z2 > 0) in respective camera coordinate frames.
4. Parallax ray convergence angle calculation.
5. Sub-pixel reprojection error evaluation for Image 1, Image 2, and combined metric.
6. Transparent multi-stage geometric filtering with explicit rejection reasons.
7. Multi-view feature track association and RGB color sampling.
8. Standards-compliant ASCII PLY point cloud exporter and JSON metadata generator.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import math

import cv2
import numpy as np
import pandas as pd


@dataclass
class TriangulatedPoint:
    """Represents a triangulated 3D point and its geometric verification metadata."""
    point_id: int
    track_id: Optional[int]
    kp_idx1: int
    kp_idx2: int
    x1: float
    y1: float
    x2: float
    y2: float

    # 3D Euclidean coordinates in Camera 1 frame
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0

    # 3D Euclidean coordinates in Camera 2 frame
    X2: float = 0.0
    Y2: float = 0.0
    Z2: float = 0.0

    # Homogeneous projective scale coordinate
    W: float = 0.0

    # Parallax angle in degrees
    parallax_deg: float = 0.0

    # Reprojection errors in pixels
    reproj_err1: float = float("inf")
    reproj_err2: float = float("inf")
    reproj_err_comb: float = float("inf")

    # Sampled RGB color (0-255)
    r: int = 200
    g: int = 200
    b: int = 200

    # Filtering status
    is_valid: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    primary_rejection_reason: str = "PASSED"

    # Multi-view track metadata
    track_length: int = 2
    observing_images: List[str] = field(default_factory=list)

    @property
    def pt3D(self) -> np.ndarray:
        """Returns 3D Euclidean coordinate array [X, Y, Z] in Camera 1 frame."""
        return np.array([self.X, self.Y, self.Z], dtype=np.float64)

    @property
    def color(self) -> Tuple[int, int, int]:
        """Returns RGB color tuple."""
        return (self.r, self.g, self.b)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes point record to dictionary for CSV export."""
        return {
            "point_id": self.point_id,
            "track_id": self.track_id if self.track_id is not None else -1,
            "kp_idx1": self.kp_idx1,
            "kp_idx2": self.kp_idx2,
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "X": round(self.X, 6),
            "Y": round(self.Y, 6),
            "Z": round(self.Z, 6),
            "Z1": round(self.Z, 6),
            "Z2": round(self.Z2, 6),
            "W": round(self.W, 8),
            "parallax_deg": round(self.parallax_deg, 3),
            "reproj_err1_px": round(self.reproj_err1, 4),
            "reproj_err2_px": round(self.reproj_err2, 4),
            "reproj_err_comb_px": round(self.reproj_err_comb, 4),
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "is_valid": self.is_valid,
            "primary_rejection": self.primary_rejection_reason,
            "all_rejections": ";".join(self.rejection_reasons) if self.rejection_reasons else "NONE",
            "track_length": self.track_length,
            "observing_views_count": len(self.observing_images),
            "observing_views": ",".join(self.observing_images) if self.observing_images else "",
        }


def triangulate_point_dlt(
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    P1: np.ndarray,
    P2: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Triangulates a single 3D point using Direct Linear Transformation (Hartley & Zisserman).

    Solves A * X_homo = 0 via SVD, where:
        A = [ x1 * p1_3^T - p1_1^T ]
            [ y1 * p1_3^T - p1_2^T ]
            [ x2 * p2_3^T - p2_1^T ]
            [ y2 * p2_3^T - p2_2^T ]

    Returns:
        pt3D_homo: 4D homogeneous coordinate array [X, Y, Z, W].
        W: The 4th homogeneous coordinate.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    A = np.array([
        x1 * P1[2, :] - P1[0, :],
        y1 * P1[2, :] - P1[1, :],
        x2 * P2[2, :] - P2[0, :],
        y2 * P2[2, :] - P2[1, :],
    ], dtype=np.float64)

    # SVD solution
    _, _, Vt = np.linalg.svd(A)
    X_homo = Vt[-1, :]

    W = float(X_homo[3])
    return X_homo, W


def compute_ray_parallax_angle(
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
) -> float:
    """
    Computes ray parallax angle in degrees between back-projected pixel rays in Camera 1 frame.

        d1 = K^-1 * [x1, y1, 1]^T
        d2 = R^T * K^-1 * [x2, y2, 1]^T
        theta = arccos( (d1 / ||d1||) . (d2 / ||d2||) )
    """
    K_inv = np.linalg.inv(K)
    p1_h = np.array([pt1[0], pt1[1], 1.0], dtype=np.float64)
    p2_h = np.array([pt2[0], pt2[1], 1.0], dtype=np.float64)

    d1 = K_inv @ p1_h
    d2 = R.T @ K_inv @ p2_h

    d1_norm = np.linalg.norm(d1)
    d2_norm = np.linalg.norm(d2)

    if d1_norm < 1e-12 or d2_norm < 1e-12:
        return 0.0

    d1_u = d1 / d1_norm
    d2_u = d2 / d2_norm

    dot_prod = float(np.clip(np.dot(d1_u, d2_u), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot_prod)))


def compute_reprojection_errors(
    pt3D_cam1: np.ndarray,
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Computes Euclidean reprojection residuals in pixels for Image 1 and Image 2.

    Returns:
        err1: Distance in pixels between pt1 and projection into Camera 1.
        err2: Distance in pixels between pt2 and projection into Camera 2.
        err_comb: Mean combined reprojection error (err1 + err2) / 2.
    """
    # Projection 1: K @ pt3D
    proj1 = K @ pt3D_cam1
    if abs(proj1[2]) < 1e-12:
        return float("inf"), float("inf"), float("inf")
    p1_px = np.array([proj1[0] / proj1[2], proj1[1] / proj1[2]])
    err1 = float(np.linalg.norm(np.array(pt1) - p1_px))

    # Projection 2: K @ (R @ pt3D + t)
    pt3D_cam2 = R @ pt3D_cam1 + t.ravel()
    proj2 = K @ pt3D_cam2
    if abs(proj2[2]) < 1e-12:
        return float("inf"), float("inf"), float("inf")
    p2_px = np.array([proj2[0] / proj2[2], proj2[1] / proj2[2]])
    err2 = float(np.linalg.norm(np.array(pt2) - p2_px))

    err_comb = float(0.5 * (err1 + err2))
    return err1, err2, err_comb


def triangulate_correspondence_batch(
    correspondences: List[Dict[str, Any]],
    P1: np.ndarray,
    P2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    img1_rgb: Optional[np.ndarray] = None,
    epsilon_w: float = 1e-7,
    min_parallax_deg: float = 2.0,
    max_reproj_err_px: float = 2.0,
) -> Tuple[List[TriangulatedPoint], Dict[str, Any]]:
    """
    Triangulates a collection of 2D correspondences and applies rigorous geometric filters.

    Args:
        correspondences: List of dicts, each with keys:
            'pt1': (x, y), 'pt2': (x, y), 'kp_idx1': int, 'kp_idx2': int,
            'track_id': Optional[int], 'track_length': int, 'observing_images': List[str]
        P1: 3x4 Camera 1 projection matrix K[I | 0].
        P2: 3x4 Camera 2 projection matrix K[R | t].
        K: 3x3 Camera intrinsic matrix.
        R: 3x3 Relative rotation matrix.
        t: 3x1 Relative translation vector.
        img1_rgb: Optional RGB image for color sampling at pt1.
        epsilon_w: Homogeneous scale near-zero rejection threshold.
        min_parallax_deg: Minimum viewing ray parallax angle in degrees.
        max_reproj_err_px: Maximum allowable combined reprojection error in pixels.

    Returns:
        points: Complete list of TriangulatedPoint objects (both accepted and rejected).
        filtering_stats: Dictionary containing attrition counts and distributions.
    """
    total_candidates = len(correspondences)
    points: List[TriangulatedPoint] = []

    # Attrition tracking
    pass_scale = 0
    pass_cheirality = 0
    pass_parallax = 0
    pass_reprojection = 0

    rejection_counts: Dict[str, int] = {
        "INVALID_HOMOGENEOUS_SCALE": 0,
        "NEGATIVE_DEPTH_CAM1": 0,
        "NEGATIVE_DEPTH_CAM2": 0,
        "NEGATIVE_DEPTH_BOTH": 0,
        "INSUFFICIENT_PARALLAX": 0,
        "HIGH_REPROJECTION_ERROR": 0,
    }

    # Sensitivity tracking for parallax
    parallax_sensitivity = {
        "ge_1deg": 0,
        "ge_2deg": 0,
        "ge_3deg": 0,
        "ge_5deg": 0,
    }

    img_h, img_w = (img1_rgb.shape[:2]) if img1_rgb is not None else (2272, 1704)

    for idx, corr in enumerate(correspondences):
        pt1 = corr["pt1"]
        pt2 = corr["pt2"]
        kp_idx1 = corr.get("kp_idx1", -1)
        kp_idx2 = corr.get("kp_idx2", -1)
        track_id = corr.get("track_id", None)
        track_length = corr.get("track_length", 2)
        observing_images = corr.get("observing_images", [])

        # Color sampling from img1
        color_r, color_g, color_b = 200, 200, 200
        if img1_rgb is not None:
            px_x = int(round(pt1[0]))
            px_y = int(round(pt1[1]))
            if 0 <= px_x < img_w and 0 <= px_y < img_h:
                # Expecting RGB image input
                color_r = int(img1_rgb[px_y, px_x, 0])
                color_g = int(img1_rgb[px_y, px_x, 1])
                color_b = int(img1_rgb[px_y, px_x, 2])

        p_obj = TriangulatedPoint(
            point_id=idx + 1,
            track_id=track_id,
            kp_idx1=kp_idx1,
            kp_idx2=kp_idx2,
            x1=float(pt1[0]),
            y1=float(pt1[1]),
            x2=float(pt2[0]),
            y2=float(pt2[1]),
            r=color_r,
            g=color_g,
            b=color_b,
            track_length=track_length,
            observing_images=observing_images,
        )

        # 1. DLT Triangulation
        X_homo, W = triangulate_point_dlt(pt1, pt2, P1, P2)
        p_obj.W = W

        # Filter 1: Homogeneous scale W
        if abs(W) < epsilon_w:
            p_obj.is_valid = False
            p_obj.rejection_reasons.append("INVALID_HOMOGENEOUS_SCALE")
            p_obj.primary_rejection_reason = "INVALID_HOMOGENEOUS_SCALE"
            rejection_counts["INVALID_HOMOGENEOUS_SCALE"] += 1
            points.append(p_obj)
            continue

        pass_scale += 1

        # Signed division
        pt3D_cam1 = X_homo[:3] / W
        p_obj.X = float(pt3D_cam1[0])
        p_obj.Y = float(pt3D_cam1[1])
        p_obj.Z = float(pt3D_cam1[2])

        # Transform to Camera 2 frame
        pt3D_cam2 = R @ pt3D_cam1 + t.ravel()
        p_obj.X2 = float(pt3D_cam2[0])
        p_obj.Y2 = float(pt3D_cam2[1])
        p_obj.Z2 = float(pt3D_cam2[2])

        Z1 = p_obj.Z
        Z2 = p_obj.Z2

        # Filter 2: Cheirality (Positive depth in both cameras)
        neg1 = Z1 <= 0
        neg2 = Z2 <= 0

        if neg1 and neg2:
            p_obj.is_valid = False
            p_obj.rejection_reasons.append("NEGATIVE_DEPTH_BOTH")
            p_obj.primary_rejection_reason = "NEGATIVE_DEPTH_BOTH"
            rejection_counts["NEGATIVE_DEPTH_BOTH"] += 1
            points.append(p_obj)
            continue
        elif neg1:
            p_obj.is_valid = False
            p_obj.rejection_reasons.append("NEGATIVE_DEPTH_CAM1")
            p_obj.primary_rejection_reason = "NEGATIVE_DEPTH_CAM1"
            rejection_counts["NEGATIVE_DEPTH_CAM1"] += 1
            points.append(p_obj)
            continue
        elif neg2:
            p_obj.is_valid = False
            p_obj.rejection_reasons.append("NEGATIVE_DEPTH_CAM2")
            p_obj.primary_rejection_reason = "NEGATIVE_DEPTH_CAM2"
            rejection_counts["NEGATIVE_DEPTH_CAM2"] += 1
            points.append(p_obj)
            continue

        pass_cheirality += 1

        # 3. Parallax Angle
        par_deg = compute_ray_parallax_angle(pt1, pt2, K, R)
        p_obj.parallax_deg = par_deg

        if par_deg >= 1.0:
            parallax_sensitivity["ge_1deg"] += 1
        if par_deg >= 2.0:
            parallax_sensitivity["ge_2deg"] += 1
        if par_deg >= 3.0:
            parallax_sensitivity["ge_3deg"] += 1
        if par_deg >= 5.0:
            parallax_sensitivity["ge_5deg"] += 1

        # Filter 3: Parallax Threshold
        if par_deg < min_parallax_deg:
            p_obj.is_valid = False
            p_obj.rejection_reasons.append("INSUFFICIENT_PARALLAX")
            p_obj.primary_rejection_reason = "INSUFFICIENT_PARALLAX"
            rejection_counts["INSUFFICIENT_PARALLAX"] += 1
            points.append(p_obj)
            continue

        pass_parallax += 1

        # 4. Reprojection Errors
        e1, e2, e_comb = compute_reprojection_errors(pt3D_cam1, pt1, pt2, K, R, t)
        p_obj.reproj_err1 = e1
        p_obj.reproj_err2 = e2
        p_obj.reproj_err_comb = e_comb

        # Filter 4: Maximum combined reprojection error
        if e_comb > max_reproj_err_px:
            p_obj.is_valid = False
            p_obj.rejection_reasons.append("HIGH_REPROJECTION_ERROR")
            p_obj.primary_rejection_reason = "HIGH_REPROJECTION_ERROR"
            rejection_counts["HIGH_REPROJECTION_ERROR"] += 1
            points.append(p_obj)
            continue

        pass_reprojection += 1
        p_obj.is_valid = True
        p_obj.primary_rejection_reason = "PASSED"
        points.append(p_obj)

    accepted_points = [p for p in points if p.is_valid]

    filtering_stats = {
        "total_candidates": total_candidates,
        "pass_scale": pass_scale,
        "pass_cheirality": pass_cheirality,
        "pass_parallax": pass_parallax,
        "pass_reprojection": pass_reprojection,
        "accepted_count": len(accepted_points),
        "rejected_count": total_candidates - len(accepted_points),
        "acceptance_ratio": round(len(accepted_points) / max(1, total_candidates), 4),
        "rejection_counts": rejection_counts,
        "parallax_sensitivity": parallax_sensitivity,
        "filtering_parameters": {
            "epsilon_w": epsilon_w,
            "min_parallax_deg": min_parallax_deg,
            "max_reproj_err_px": max_reproj_err_px,
        }
    }

    return points, filtering_stats


def export_point_cloud_ply(
    points: List[TriangulatedPoint],
    out_path: Path,
    only_valid: bool = True,
) -> int:
    """
    Exports 3D point cloud as a standards-compliant ASCII PLY file.
    Includes X, Y, Z coordinates and RGB color properties.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts_to_export = [p for p in points if p.is_valid] if only_valid else points
    N = len(pts_to_export)

    header = [
        "ply",
        "format ascii 1.0",
        "comment Created by Adaptive Multi-View 3D Reconstruction Pipeline",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]

    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(header) + "\n")
        for p in pts_to_export:
            f.write(f"{p.X:.6f} {p.Y:.6f} {p.Z:.6f} {p.r} {p.g} {p.b}\n")

    return N


def export_triangulation_csv(
    points: List[TriangulatedPoint],
    out_path: Path,
) -> None:
    """Exports comprehensive triangulation records to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [p.to_dict() for p in points]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def export_metadata_json(
    metadata: Dict[str, Any],
    out_path: Path,
) -> None:
    """Exports experimental configuration and reconstruction metadata to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
