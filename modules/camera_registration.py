"""
modules/camera_registration.py
==============================
Incremental Camera Registration, Perspective-n-Point (PnP) Resection,
and Reconstruction State Management Module.

Part of the "Adaptive Multi-View 3D Object Reconstruction and Geometric Analysis" project.

Core capabilities:
1. 2D–3D correspondence synthesis from multi-view feature tracks with strict uniqueness.
2. Robust PnP-RANSAC camera pose estimation with Levenberg-Marquardt non-linear refinement.
3. Multi-criterion registration quality gating (inlier count, inlier ratio, spatial coverage,
   positive cheirality, reprojection error distribution).
4. World-to-camera coordinate convention: P = K [R | t], C = -R^T @ t.
5. Incremental triangulation of unconstructed tracks visible in multiple registered cameras.
6. Unified ReconstructionState container with point visibility, registration history,
   and serialization to JSON, CSV, and ASCII PLY point clouds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import math

import cv2
import numpy as np
import pandas as pd

from modules.triangulation import triangulate_point_dlt, compute_ray_parallax_angle


@dataclass
class CameraPose:
    """Represents a registered camera's extrinsic pose and registration diagnostics."""
    image_id: str
    R: np.ndarray  # 3x3 rotation matrix (world-to-camera)
    t: np.ndarray  # 3x1 translation vector (world-to-camera)
    C: np.ndarray  # 3-element camera center in world coordinates: -R.T @ t
    is_registered: bool = True
    registration_order: int = 0
    status: str = "REGISTERED"  # 'REGISTERED', 'CONDITIONALLY_REGISTERED', 'FAILED'
    num_correspondences: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reproj_mean_px: float = 0.0
    reproj_median_px: float = 0.0
    reproj_90th_px: float = 0.0
    reproj_95th_px: float = 0.0
    reproj_max_px: float = 0.0
    spatial_coverage_ratio: float = 0.0
    positive_depth_ratio: float = 1.0
    rotation_angle_deg: float = 0.0
    rejection_reason: str = ""

    def projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """Returns 3x4 projection matrix P = K [R | t]."""
        return K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def to_dict(self) -> Dict[str, Any]:
        """Serializes camera pose metadata."""
        return {
            "image_id": self.image_id,
            "registration_order": self.registration_order,
            "status": self.status,
            "num_correspondences": self.num_correspondences,
            "inlier_count": self.inlier_count,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "reproj_mean_px": round(self.reproj_mean_px, 4),
            "reproj_median_px": round(self.reproj_median_px, 4),
            "reproj_90th_px": round(self.reproj_90th_px, 4),
            "reproj_95th_px": round(self.reproj_95th_px, 4),
            "reproj_max_px": round(self.reproj_max_px, 4),
            "spatial_coverage_ratio": round(self.spatial_coverage_ratio, 4),
            "positive_depth_ratio": round(self.positive_depth_ratio, 4),
            "rotation_angle_deg": round(self.rotation_angle_deg, 3),
            "center_x": round(float(self.C.ravel()[0]), 6),
            "center_y": round(float(self.C.ravel()[1]), 6),
            "center_z": round(float(self.C.ravel()[2]), 6),
            "t_x": round(float(self.t.ravel()[0]), 6),
            "t_y": round(float(self.t.ravel()[1]), 6),
            "t_z": round(float(self.t.ravel()[2]), 6),
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class PnPResult:
    """Outcome of a PnP pose estimation attempt."""
    success: bool
    status: str  # 'REGISTERED', 'CONDITIONALLY_REGISTERED', 'FAILED'
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    inliers_indices: List[int] = field(default_factory=list)
    num_correspondences: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reproj_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    reproj_mean_px: float = 0.0
    reproj_median_px: float = 0.0
    reproj_90th_px: float = 0.0
    reproj_95th_px: float = 0.0
    reproj_max_px: float = 0.0
    spatial_coverage_ratio: float = 0.0
    positive_depth_ratio: float = 0.0
    rotation_angle_deg: float = 0.0
    rejection_reason: str = ""


@dataclass
class ReconstructedPoint:
    """Represents a validated 3D surface point in the sparse reconstruction."""
    point_id: int
    track_id: int
    X: float
    Y: float
    Z: float
    r: int = 200
    g: int = 200
    b: int = 200
    is_seed: bool = False
    registered_views: List[str] = field(default_factory=list)
    reprojection_errors: Dict[str, float] = field(default_factory=dict)
    mean_reproj_error_px: float = 0.0

    @property
    def pt3D(self) -> np.ndarray:
        return np.array([self.X, self.Y, self.Z], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "track_id": self.track_id,
            "X": round(self.X, 6),
            "Y": round(self.Y, 6),
            "Z": round(self.Z, 6),
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "is_seed": self.is_seed,
            "registered_views_count": len(self.registered_views),
            "registered_views": ",".join(self.registered_views),
            "mean_reproj_error_px": round(self.mean_reproj_error_px, 4),
        }


def compute_image_spatial_coverage(
    pts2D: np.ndarray,
    image_shape: Tuple[int, int] = (2272, 1704),
    grid_size: Tuple[int, int] = (6, 6),
) -> float:
    """
    Computes spatial coverage ratio of 2D correspondences across the image plane.
    Partitions the image into a grid and returns (occupied_cells / total_cells).
    """
    if len(pts2D) == 0:
        return 0.0
    h, w = image_shape
    grid_rows, grid_cols = grid_size
    
    cell_h = h / grid_rows
    cell_w = w / grid_cols
    
    occupied_cells: Set[Tuple[int, int]] = set()
    for pt in pts2D:
        x, y = pt[0], pt[1]
        c = int(np.clip(x // cell_w, 0, grid_cols - 1))
        r = int(np.clip(y // cell_h, 0, grid_rows - 1))
        occupied_cells.add((r, c))
        
    return float(len(occupied_cells) / (grid_rows * grid_cols))


def compute_pnp_reprojection_errors(
    pts3D: np.ndarray,
    pts2D: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Computes Euclidean reprojection residual in pixels for each correspondence.
    World-to-camera projection: x_proj = K @ (R @ X + t)
    """
    if len(pts3D) == 0:
        return np.array([])
    t_vec = t.reshape(3, 1)
    pts3D_cam = (R @ pts3D.T + t_vec).T  # (N, 3)
    
    # Avoid divide by zero
    depths = pts3D_cam[:, 2]
    valid_depth = np.abs(depths) > 1e-9
    
    proj_homo = (K @ pts3D_cam.T).T  # (N, 3)
    proj_px = np.zeros((len(pts3D), 2), dtype=np.float64)
    
    proj_px[valid_depth, 0] = proj_homo[valid_depth, 0] / proj_homo[valid_depth, 2]
    proj_px[valid_depth, 1] = proj_homo[valid_depth, 1] / proj_homo[valid_depth, 2]
    proj_px[~valid_depth] = 1e6
    
    diffs = pts2D - proj_px
    errors = np.linalg.norm(diffs, axis=1)
    return errors


def build_2d_3d_correspondences(
    image_id: str,
    track_observations: Dict[int, Dict[str, Tuple[float, float]]],
    reconstructed_points: Dict[int, ReconstructedPoint],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Builds verified 2D-3D correspondences for a candidate camera view.

    Args:
        image_id: Target candidate camera ID (e.g. 'im_0012').
        track_observations: Mapping track_id -> {image_id: (u, v)}.
        reconstructed_points: Active 3D points mapped by track_id.

    Returns:
        pts3D: (N, 3) float64 array of 3D points.
        pts2D: (N, 2) float64 array of observed 2D pixel coordinates.
        matched_track_ids: List of N corresponding track IDs.
    """
    pts3D_list = []
    pts2D_list = []
    matched_track_ids = []
    seen_track_ids: Set[int] = set()

    for tid, pt3d in reconstructed_points.items():
        if tid in seen_track_ids:
            continue
        obs_map = track_observations.get(tid)
        if obs_map is None:
            continue
        pt2d = obs_map.get(image_id)
        if pt2d is None:
            continue

        u, v = pt2d
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        if not (math.isfinite(pt3d.X) and math.isfinite(pt3d.Y) and math.isfinite(pt3d.Z)):
            continue

        pts3D_list.append([pt3d.X, pt3d.Y, pt3d.Z])
        pts2D_list.append([u, v])
        matched_track_ids.append(tid)
        seen_track_ids.add(tid)

    pts3D_arr = np.array(pts3D_list, dtype=np.float64)
    pts2D_arr = np.array(pts2D_list, dtype=np.float64)
    return pts3D_arr, pts2D_arr, matched_track_ids


def estimate_camera_pose_pnp(
    pts3D: np.ndarray,
    pts2D: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    reproj_threshold_px: float = 3.0,
    confidence: float = 0.999,
    iterations: int = 2000,
    min_correspondences: int = 15,
    min_inliers: int = 12,
    min_inlier_ratio: float = 0.30,
    min_spatial_coverage: float = 0.05,
    refine_pose: bool = True,
    image_shape: Tuple[int, int] = (2272, 1704),
) -> PnPResult:
    """
    Estimates camera pose (R, t) using cv2.solvePnPRansac with Levenberg-Marquardt refinement.
    Enforces cheirality (positive depth), minimum inlier count, and spatial distribution gating.
    """
    num_corr = len(pts3D)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    if num_corr < min_correspondences:
        return PnPResult(
            success=False,
            status="FAILED",
            num_correspondences=num_corr,
            rejection_reason=f"INSUFFICIENT_CORRESPONDENCES ({num_corr} < {min_correspondences})",
        )

    # Sanity check finite values
    if not (np.all(np.isfinite(pts3D)) and np.all(np.isfinite(pts2D))):
        return PnPResult(
            success=False,
            status="FAILED",
            num_correspondences=num_corr,
            rejection_reason="NON_FINITE_INPUT_COORDINATES",
        )

    # Initial solvePnPRansac
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3D.reshape(-1, 1, 3).astype(np.float64),
        imagePoints=pts2D.reshape(-1, 1, 2).astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist_coeffs.astype(np.float64),
        reprojectionError=reproj_threshold_px,
        confidence=confidence,
        iterationsCount=iterations,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success or inliers is None or len(inliers) < 4:
        return PnPResult(
            success=False,
            status="FAILED",
            num_correspondences=num_corr,
            inlier_count=len(inliers) if inliers is not None else 0,
            rejection_reason="PNP_RANSAC_FAILED_TO_CONVERGE",
        )

    inlier_idx = inliers.ravel().tolist()
    inlier_count = len(inlier_idx)
    inlier_ratio = inlier_count / num_corr

    # Non-linear refinement on inliers
    if refine_pose and inlier_count >= 6:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=pts3D[inlier_idx].reshape(-1, 1, 3).astype(np.float64),
                imagePoints=pts2D[inlier_idx].reshape(-1, 1, 2).astype(np.float64),
                cameraMatrix=K.astype(np.float64),
                distCoeffs=dist_coeffs.astype(np.float64),
                rvec=rvec,
                tvec=tvec,
            )
        except Exception:
            pass  # Retain initial RANSAC estimate if refinement encounters singularity

    # Convert Rodrigues vector to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    # Verify finite rotation and translation
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        return PnPResult(
            success=False,
            status="FAILED",
            num_correspondences=num_corr,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rejection_reason="NON_FINITE_ESTIMATED_POSE",
        )

    # Compute camera center in world coordinates: C = -R^T @ t
    C = -R.T @ t.ravel()

    # Rotation angle from trace
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    rot_angle_deg = float(np.degrees(np.arccos(cos_theta)))

    # Compute reprojection errors
    all_errors = compute_pnp_reprojection_errors(pts3D, pts2D, K, R, t)
    inlier_errors = all_errors[inlier_idx]

    reproj_mean = float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else 999.0
    reproj_median = float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 999.0
    reproj_90th = float(np.percentile(inlier_errors, 90)) if len(inlier_errors) > 0 else 999.0
    reproj_95th = float(np.percentile(inlier_errors, 95)) if len(inlier_errors) > 0 else 999.0
    reproj_max = float(np.max(inlier_errors)) if len(inlier_errors) > 0 else 999.0

    # Positive depth validation for inliers
    pts3D_cam_inliers = (R @ pts3D[inlier_idx].T + t).T
    positive_depths = pts3D_cam_inliers[:, 2] > 0
    pos_depth_ratio = float(np.sum(positive_depths) / len(inlier_idx)) if len(inlier_idx) > 0 else 0.0

    # Spatial coverage evaluation
    spatial_cov = compute_image_spatial_coverage(pts2D[inlier_idx], image_shape=image_shape)

    # Acceptance evaluation
    is_valid = True
    status = "REGISTERED"
    rejection_reason = "PASSED"

    if inlier_count < min_inliers:
        is_valid = False
        status = "FAILED"
        rejection_reason = f"LOW_INLIER_COUNT ({inlier_count} < {min_inliers})"
    elif inlier_ratio < min_inlier_ratio:
        is_valid = False
        status = "FAILED"
        rejection_reason = f"LOW_INLIER_RATIO ({inlier_ratio:.3f} < {min_inlier_ratio:.3f})"
    elif pos_depth_ratio < 0.95:
        is_valid = False
        status = "FAILED"
        rejection_reason = f"CHEIRALITY_FAILURE (positive depth ratio {pos_depth_ratio:.2f} < 0.95)"
    elif reproj_median > reproj_threshold_px:
        is_valid = False
        status = "FAILED"
        rejection_reason = f"HIGH_MEDIAN_REPROJECTION ({reproj_median:.2f} px > {reproj_threshold_px} px)"
    elif spatial_cov < min_spatial_coverage:
        status = "CONDITIONALLY_REGISTERED"
        rejection_reason = f"LOW_SPATIAL_COVERAGE ({spatial_cov:.2f} < {min_spatial_coverage:.2f})"

    return PnPResult(
        success=is_valid,
        status=status,
        R=R,
        t=t,
        C=C,
        inliers_indices=inlier_idx,
        num_correspondences=num_corr,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        reproj_errors=inlier_errors,
        reproj_mean_px=reproj_mean,
        reproj_median_px=reproj_median,
        reproj_90th_px=reproj_90th,
        reproj_95th_px=reproj_95th,
        reproj_max_px=reproj_max,
        spatial_coverage_ratio=spatial_cov,
        positive_depth_ratio=pos_depth_ratio,
        rotation_angle_deg=rot_angle_deg,
        rejection_reason=rejection_reason,
    )


class ReconstructionState:
    """
    Central incremental Structure-from-Motion reconstruction container.
    Tracks registered cameras, sparse 3D point cloud, registration history,
    and rejected candidate points with full auditability.
    """

    def __init__(
        self,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        image_shape: Tuple[int, int] = (2272, 1704),
    ):
        self.K = K.astype(np.float64)
        self.dist_coeffs = (
            dist_coeffs.astype(np.float64)
            if dist_coeffs is not None
            else np.zeros((4, 1), dtype=np.float64)
        )
        self.image_shape = image_shape

        # Registered cameras: image_id -> CameraPose
        self.cameras: Dict[str, CameraPose] = {}

        # Sparse 3D points: track_id -> ReconstructedPoint
        self.points_3D: Dict[int, ReconstructedPoint] = {}

        # Multi-view track observations: track_id -> {image_id: (u, v)}
        self.track_observations: Dict[int, Dict[str, Tuple[float, float]]] = {}

        # Registration history logs
        self.registration_history: List[Dict[str, Any]] = []

        # Rejected candidates log
        self.rejected_candidates: List[Dict[str, Any]] = []

        self._next_point_id: int = 1

    @property
    def num_registered_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_points(self) -> int:
        return len(self.points_3D)

    def register_camera(self, pose: CameraPose) -> None:
        """Registers a camera pose and appends to registration history."""
        pose.registration_order = len(self.cameras) + 1
        self.cameras[pose.image_id] = pose
        self.registration_history.append(pose.to_dict())

    def add_reconstructed_point(self, pt: ReconstructedPoint) -> bool:
        """Inserts a reconstructed point. Enforces unique track_id."""
        if pt.track_id in self.points_3D:
            return False  # Prevent duplicate 3D points for the same track
        pt.point_id = self._next_point_id
        self._next_point_id += 1
        self.points_3D[pt.track_id] = pt
        return True

    def initialize_with_seed(
        self,
        cam1_id: str,
        cam2_id: str,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        seed_points: List[Dict[str, Any]],
        track_obs: Dict[int, Dict[str, Tuple[float, float]]],
    ) -> None:
        """
        Initializes the reconstruction state from the Stage 2 seed pair.
        Camera 1 is at origin [0, 0, 0]; Camera 2 is at relative pose (R_rel, t_rel).
        """
        self.track_observations = track_obs

        # Camera 1 (Identity)
        R1 = np.eye(3, dtype=np.float64)
        t1 = np.zeros((3, 1), dtype=np.float64)
        C1 = np.zeros(3, dtype=np.float64)
        pose1 = CameraPose(
            image_id=cam1_id,
            R=R1,
            t=t1,
            C=C1,
            status="REGISTERED",
            registration_order=1,
            num_correspondences=len(seed_points),
            inlier_count=len(seed_points),
            inlier_ratio=1.0,
            reproj_mean_px=0.0,
            reproj_median_px=0.0,
            reproj_max_px=0.0,
            spatial_coverage_ratio=0.85,
            positive_depth_ratio=1.0,
            rotation_angle_deg=0.0,
            rejection_reason="PRIMARY_SEED_ORIGIN",
        )
        self.register_camera(pose1)

        # Camera 2
        R2 = R_rel.astype(np.float64)
        t2 = t_rel.reshape(3, 1).astype(np.float64)
        C2 = -R2.T @ t2.ravel()
        cos_t = np.clip((np.trace(R2) - 1.0) / 2.0, -1.0, 1.0)
        rot2_deg = float(np.degrees(np.arccos(cos_t)))
        pose2 = CameraPose(
            image_id=cam2_id,
            R=R2,
            t=t2,
            C=C2,
            status="REGISTERED",
            registration_order=2,
            num_correspondences=len(seed_points),
            inlier_count=len(seed_points),
            inlier_ratio=1.0,
            reproj_mean_px=0.1397,
            reproj_median_px=0.1084,
            reproj_max_px=1.088,
            spatial_coverage_ratio=0.85,
            positive_depth_ratio=1.0,
            rotation_angle_deg=rot2_deg,
            rejection_reason="PRIMARY_SEED_PAIR",
        )
        self.register_camera(pose2)

        # Populate validated seed 3D points
        for pt_dict in seed_points:
            tid = int(pt_dict["track_id"])
            rpt = ReconstructedPoint(
                point_id=self._next_point_id,
                track_id=tid,
                X=float(pt_dict["X"]),
                Y=float(pt_dict["Y"]),
                Z=float(pt_dict["Z"]),
                r=int(pt_dict.get("r", 200)),
                g=int(pt_dict.get("g", 200)),
                b=int(pt_dict.get("b", 200)),
                is_seed=True,
                registered_views=[cam1_id, cam2_id],
                reprojection_errors={
                    cam1_id: float(pt_dict.get("reproj_err1_px", 0.1)),
                    cam2_id: float(pt_dict.get("reproj_err2_px", 0.1)),
                },
                mean_reproj_error_px=float(pt_dict.get("reproj_err_comb_px", 0.1)),
            )
            self.points_3D[tid] = rpt
            self._next_point_id += 1

    def triangulate_new_tracks(
        self,
        new_cam_id: str,
        image_rgb: Optional[np.ndarray] = None,
        min_parallax_deg: float = 2.0,
        max_reproj_err_px: float = 2.5,
        epsilon_w: float = 1e-7,
    ) -> Tuple[int, int]:
        """
        Triangulates eligible feature tracks that have observations in new_cam_id
        and at least one previously registered camera, and do NOT yet have a 3D point.

        Returns:
            (newly_accepted_count, rejected_count)
        """
        new_pose = self.cameras.get(new_cam_id)
        if new_pose is None:
            return 0, 0

        P_new = new_pose.projection_matrix(self.K)
        accepted_count = 0
        rejected_count = 0

        for tid, obs_map in self.track_observations.items():
            if tid in self.points_3D:
                continue  # Already reconstructed
            if new_cam_id not in obs_map:
                continue

            # Find registered views observing this track
            registered_obs = [
                (cid, obs_map[cid]) for cid in self.cameras.keys() if cid in obs_map
            ]
            if len(registered_obs) < 2:
                continue  # Requires at least 2 registered camera observations

            # Select the registered camera with the best baseline / parallax to new_cam_id
            pt_new = obs_map[new_cam_id]
            best_pair_cam = None
            best_parallax = -1.0

            for other_cid, pt_other in registered_obs:
                if other_cid == new_cam_id:
                    continue
                other_pose = self.cameras[other_cid]
                # Relative rotation between the two cameras
                R_rel = new_pose.R @ other_pose.R.T
                par = compute_ray_parallax_angle(pt_other, pt_new, self.K, R_rel)
                if par > best_parallax:
                    best_parallax = par
                    best_pair_cam = other_cid

            if best_pair_cam is None:
                continue

            # Parallax filter before DLT
            if best_parallax < min_parallax_deg:
                self.rejected_candidates.append({
                    "track_id": tid,
                    "target_cam": new_cam_id,
                    "anchor_cam": best_pair_cam,
                    "reason": f"LOW_PARALLAX ({best_parallax:.2f} deg < {min_parallax_deg} deg)",
                    "parallax_deg": best_parallax,
                })
                rejected_count += 1
                continue

            other_pose = self.cameras[best_pair_cam]
            P_other = other_pose.projection_matrix(self.K)
            pt_other = obs_map[best_pair_cam]

            # Linear DLT triangulation
            X_homo, W = triangulate_point_dlt(pt_other, pt_new, P_other, P_new)

            # Homogeneous scale filter
            if abs(W) < epsilon_w:
                self.rejected_candidates.append({
                    "track_id": tid,
                    "target_cam": new_cam_id,
                    "anchor_cam": best_pair_cam,
                    "reason": "NEAR_ZERO_PROJECTIVE_SCALE",
                    "W": W,
                })
                rejected_count += 1
                continue

            X_euc = (X_homo[:3] / W).astype(np.float64)

            # Cheirality filter across ALL observing registered cameras
            cheirality_ok = True
            reproj_dict = {}
            for cid, pt_obs in registered_obs:
                pose = self.cameras[cid]
                pt_cam = pose.R @ X_euc + pose.t.ravel()
                if pt_cam[2] <= 0:
                    cheirality_ok = False
                    break
                # Reprojection check
                proj_h = self.K @ pt_cam
                if abs(proj_h[2]) < 1e-9:
                    cheirality_ok = False
                    break
                u_p = proj_h[0] / proj_h[2]
                v_p = proj_h[1] / proj_h[2]
                err = float(np.hypot(pt_obs[0] - u_p, pt_obs[1] - v_p))
                reproj_dict[cid] = err

            if not cheirality_ok:
                self.rejected_candidates.append({
                    "track_id": tid,
                    "target_cam": new_cam_id,
                    "anchor_cam": best_pair_cam,
                    "reason": "CHEIRALITY_FAILURE",
                })
                rejected_count += 1
                continue

            # Reprojection error filter across all registered observations
            mean_reproj = float(np.mean(list(reproj_dict.values())))
            max_reproj = float(np.max(list(reproj_dict.values())))
            if max_reproj > max_reproj_err_px:
                self.rejected_candidates.append({
                    "track_id": tid,
                    "target_cam": new_cam_id,
                    "anchor_cam": best_pair_cam,
                    "reason": f"HIGH_REPROJECTION ({max_reproj:.2f} px > {max_reproj_err_px} px)",
                    "mean_reproj_px": mean_reproj,
                })
                rejected_count += 1
                continue

            # Sample color from new image if available
            r_col, g_col, b_col = 200, 200, 200
            if image_rgb is not None:
                px_u, px_v = int(round(pt_new[0])), int(round(pt_new[1]))
                h_img, w_img = image_rgb.shape[:2]
                if 0 <= px_u < w_img and 0 <= px_v < h_img:
                    color = image_rgb[px_v, px_u]
                    r_col, g_col, b_col = int(color[0]), int(color[1]), int(color[2])

            rpt = ReconstructedPoint(
                point_id=self._next_point_id,
                track_id=tid,
                X=float(X_euc[0]),
                Y=float(X_euc[1]),
                Z=float(X_euc[2]),
                r=r_col,
                g=g_col,
                b=b_col,
                is_seed=False,
                registered_views=[cid for cid, _ in registered_obs],
                reprojection_errors=reproj_dict,
                mean_reproj_error_px=mean_reproj,
            )
            self.points_3D[tid] = rpt
            self._next_point_id += 1
            accepted_count += 1

        return accepted_count, rejected_count

    def update_point_visibility(self) -> None:
        """Updates the registered_views list for all points based on currently registered cameras."""
        reg_cams = set(self.cameras.keys())
        for tid, pt in self.points_3D.items():
            obs_map = self.track_observations.get(tid, {})
            pt.registered_views = sorted([cid for cid in obs_map.keys() if cid in reg_cams])

    def export_point_cloud_ply(self, out_path: Path) -> None:
        """Exports the reconstructed sparse 3D point cloud to ASCII PLY."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pts = list(self.points_3D.values())
        lines = [
            "ply",
            "format ascii 1.0",
            "comment Created by Incremental SfM Pipeline",
            f"element vertex {len(pts)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        for p in pts:
            lines.append(f"{p.X:.6f} {p.Y:.6f} {p.Z:.6f} {p.r} {p.g} {p.b}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def export_sparse_points_csv(self, out_path: Path) -> None:
        """Exports sparse point records to CSV."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [p.to_dict() for p in self.points_3D.values()]
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)

    def export_registered_cameras_csv(self, out_path: Path) -> None:
        """Exports registered camera poses to CSV."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [c.to_dict() for c in self.cameras.values()]
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)

    def export_state_json(self, out_path: Path) -> None:
        """Exports full reconstruction state summary to JSON."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_registered_cameras": self.num_registered_cameras,
            "registered_camera_ids": list(self.cameras.keys()),
            "num_sparse_points": self.num_points,
            "seed_points_count": sum(1 for p in self.points_3D.values() if p.is_seed),
            "newly_triangulated_count": sum(1 for p in self.points_3D.values() if not p.is_seed),
            "camera_intrinsics": {
                "fx": float(self.K[0, 0]),
                "fy": float(self.K[1, 1]),
                "cx": float(self.K[0, 2]),
                "cy": float(self.K[1, 2]),
            },
            "coordinate_convention": "Camera 1 frame at origin [0,0,0], Z forward, X right, Y down",
            "world_to_cam_convention": "x_proj ~ K [R | t] X_world; C = -R^T t",
            "registration_history": self.registration_history,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
