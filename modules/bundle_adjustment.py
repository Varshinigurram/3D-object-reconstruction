"""
modules/bundle_adjustment.py
============================
Sparse Bundle Adjustment and Global Reconstruction Refinement Module.

Project: "Adaptive Multi-View 3D Object Reconstruction and Geometric Analysis"
Dataset: THU-MVS Dog_RGB (72 Views)

Mathematical Formulations & Conventions:
----------------------------------------
- Camera Coordinate System:
    X_camera = R @ X_world + t
  where R = cv2.Rodrigues(rvec)[0], t in R^3.
- Camera Center:
    C = -R.T @ t
- Perspective Projection (Zero Distortion, Fixed Nominal Intrinsics K):
    u_proj = f * (X_c / Z_c) + c_x
    v_proj = f * (Y_c / Z_c) + c_y
- Reprojection Residual:
    r_{2k}   = u_proj - u_obs
    r_{2k+1} = v_proj - v_obs
- Gauge Freedom (7 DOF: 3 translation, 3 rotation, 1 scale):
    To resolve gauge ambiguity in monocular reconstruction:
    - Origin Reference: Camera 'im_0010' is fixed at R = I, t = 0 (6 DOF).
    - Scale Gauge: The unit baseline between 'im_0010' and 'im_0011' is preserved:
      Option A ('fixed_first_and_baseline_constraint'): Camera 'im_0010' is fixed,
      and an explicit baseline length constraint ||t_{im_0011} - t_{im_0010}|| = 1.0
      is enforced via a weighted residual constraint.
      Option B ('fixed_seed_pair'): Both seed cameras 'im_0010' and 'im_0011'
      are held fixed, freezing the initial 7-DOF gauge rigidly.
- Sparse Jacobian Structure:
    Each 2D observation depends on at most 1 camera (6 parameters) and 1 3D point (3 parameters).
    jac_sparsity is constructed as a scipy.sparse.csr_matrix with at most 9 non-zero entries per row.
"""

import sys
import time
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BAConfig:
    """Configuration parameters for Bundle Adjustment."""
    loss: str = "huber"                      # Robust loss: 'linear', 'huber', 'soft_l1', 'cauchy', 'arctan'
    f_scale: float = 2.0                    # Robust loss scale parameter (in pixels)
    max_nfev: int = 50                      # Maximum number of objective function evaluations
    ftol: float = 1e-6                      # Cost change tolerance
    xtol: float = 1e-6                      # Step norm tolerance
    gtol: float = 1e-6                      # Gradient norm / first-order optimality tolerance
    gauge_strategy: str = "fixed_seed_pair" # 'fixed_seed_pair', 'fixed_first_and_baseline_constraint', 'fixed_first_camera'
    baseline_weight: float = 1000.0         # Penalty weight for baseline length constraint (if used)
    min_positive_depth: float = 0.05        # Minimum allowed point depth in camera frame (Z > 0)
    outlier_threshold_px: float = 5.0       # Residual threshold for diagnostic flagging / rejection
    verbose: int = 2                        # Verbosity level (0, 1, 2)


@dataclass
class BAObservation:
    """Single 2D feature observation in the BA observation graph."""
    obs_id: int
    camera_id: str
    cam_idx: int                            # Index into the camera parameter array
    point_id: int
    point_idx: int                          # Index into the point parameter array
    track_id: int
    u_obs: float
    v_obs: float


@dataclass
class BAResult:
    """Summary and details of the Bundle Adjustment optimization."""
    success: bool
    status: int
    message: str
    nfev: int
    nit: int
    initial_cost: float
    final_cost: float
    initial_mean_reproj_px: float
    final_mean_reproj_px: float
    initial_median_reproj_px: float
    final_median_reproj_px: float
    initial_p90_reproj_px: float
    final_p90_reproj_px: float
    initial_p95_reproj_px: float
    final_p95_reproj_px: float
    initial_max_reproj_px: float
    final_max_reproj_px: float
    num_cameras: int
    num_optimized_cameras: int
    num_points: int
    num_optimized_points: int
    num_observations: int
    runtime_sec: float
    gauge_strategy: str
    initial_loop_closure_gap: float
    final_loop_closure_gap: float
    optimized_camera_poses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimized_points: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    per_camera_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    outlier_observations: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Mathematical Helpers
# =============================================================================

def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Converts 3-vector Rodrigues angle-axis to 3x3 orthonormal rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Converts 3x3 orthonormal rotation matrix to 3-vector Rodrigues angle-axis."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def compute_camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Computes camera optical center C in world coordinates:
      C = -R^T @ t
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    C = -R.T @ t
    return C.ravel()


def check_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float, float]:
    """
    Verifies rotation matrix orthogonality and determinant:
      ortho_err = ||R^T R - I||_F
      det_err   = |det(R) - 1.0|
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(np.linalg.det(R))
    det_err = float(abs(det - 1.0))
    is_valid = (ortho_err < tol) and (det_err < tol)
    return is_valid, ortho_err, det


# =============================================================================
# Vectorized Projection & Residual Computation
# =============================================================================

def project_points_vectorized(
    points_3d: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    K: np.ndarray,
    cam_indices: np.ndarray,
    point_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized projection of 3D points into cameras.

    Args:
        points_3d: Shape (N_points, 3) world coordinates.
        rvecs: Shape (N_cams, 3) Rodrigues rotation vectors.
        tvecs: Shape (N_cams, 3) translation vectors.
        K: Shape (3, 3) camera calibration matrix.
        cam_indices: Shape (N_obs,) int array mapping each observation to its camera.
        point_indices: Shape (N_obs,) int array mapping each observation to its 3D point.

    Returns:
        projected_2d: Shape (N_obs, 2) predicted pixel coordinates (u, v).
        depths: Shape (N_obs,) point depths along the camera optical axis (Z_camera).
    """
    # 1. Precompute rotation matrices for all cameras
    n_cams = len(rvecs)
    R_mats = np.zeros((n_cams, 3, 3), dtype=np.float64)
    for i in range(n_cams):
        R_mats[i] = rodrigues_to_matrix(rvecs[i])

    # 2. Gather camera extrinsics and point coordinates per observation
    R_obs = R_mats[cam_indices]       # Shape: (N_obs, 3, 3)
    t_obs = tvecs[cam_indices]        # Shape: (N_obs, 3)
    pts_obs = points_3d[point_indices] # Shape: (N_obs, 3)

    # 3. Transform to camera frame: X_c = R @ X_w + t
    # Using np.einsum for fast vectorized matrix-vector multiplication
    X_c = np.einsum("nij,nj->ni", R_obs, pts_obs) + t_obs  # Shape: (N_obs, 3)

    Z_c = X_c[:, 2]
    # Guard against division by zero or negative depths in numerical differentiation
    safe_Z = np.where(np.abs(Z_c) < 1e-7, 1e-7 * np.sign(Z_c + 1e-9), Z_c)

    # 4. Project using nominal camera matrix K
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u_proj = fx * (X_c[:, 0] / safe_Z) + cx
    v_proj = fy * (X_c[:, 1] / safe_Z) + cy

    projected_2d = np.column_stack([u_proj, v_proj])
    return projected_2d, Z_c


def pack_optimization_params(
    free_cam_rvecs: np.ndarray,
    free_cam_tvecs: np.ndarray,
    points_3d: np.ndarray,
) -> np.ndarray:
    """Packs camera parameters and 3D point coordinates into a single 1D parameter vector."""
    cam_params = np.hstack([free_cam_rvecs, free_cam_tvecs]).ravel()
    point_params = points_3d.ravel()
    return np.hstack([cam_params, point_params])


def unpack_optimization_params(
    params: np.ndarray,
    n_free_cams: int,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpacks the 1D parameter vector into camera rvecs, tvecs, and 3D points."""
    cam_block_size = n_free_cams * 6
    cam_params = params[:cam_block_size].reshape(n_free_cams, 6)
    points_3d = params[cam_block_size:].reshape(n_points, 3)

    free_rvecs = cam_params[:, :3]
    free_tvecs = cam_params[:, 3:6]
    return free_rvecs, free_tvecs, points_3d


def assemble_all_camera_poses(
    free_rvecs: np.ndarray,
    free_tvecs: np.ndarray,
    free_cam_indices: List[int],
    fixed_cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    total_cams: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combines fixed camera poses and free camera poses into full (total_cams, 3) arrays."""
    all_rvecs = np.zeros((total_cams, 3), dtype=np.float64)
    all_tvecs = np.zeros((total_cams, 3), dtype=np.float64)

    for local_idx, global_idx in enumerate(free_cam_indices):
        all_rvecs[global_idx] = free_rvecs[local_idx]
        all_tvecs[global_idx] = free_tvecs[local_idx]

    for global_idx, (rvec, tvec) in fixed_cam_poses.items():
        all_rvecs[global_idx] = rvec
        all_tvecs[global_idx] = tvec

    return all_rvecs, all_tvecs


def compute_residuals(
    params: np.ndarray,
    n_free_cams: int,
    n_points: int,
    free_cam_indices: List[int],
    fixed_cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    total_cams: int,
    cam_indices: np.ndarray,
    point_indices: np.ndarray,
    observed_2d: np.ndarray,
    K: np.ndarray,
    gauge_strategy: str = "fixed_seed_pair",
    seed_cam1_idx: int = 0,
    seed_cam2_idx: int = 1,
    target_baseline: float = 1.0,
    baseline_weight: float = 1000.0,
) -> np.ndarray:
    """
    Computes the reprojection residuals across all observations (and optional gauge constraint).
    """
    # 1. Unpack parameters
    free_rvecs, free_tvecs, points_3d = unpack_optimization_params(params, n_free_cams, n_points)

    # 2. Assemble all camera poses
    all_rvecs, all_tvecs = assemble_all_camera_poses(
        free_rvecs, free_tvecs, free_cam_indices, fixed_cam_poses, total_cams
    )

    # 3. Vectorized projection
    projected_2d, depths = project_points_vectorized(
        points_3d, all_rvecs, all_tvecs, K, cam_indices, point_indices
    )

    # 4. Reprojection error: predicted - observed
    reproj_diff = (projected_2d - observed_2d).ravel()

    # 5. Optional gauge constraint residual
    if gauge_strategy == "fixed_first_and_baseline_constraint":
        t1 = all_tvecs[seed_cam1_idx]
        t2 = all_tvecs[seed_cam2_idx]
        current_baseline = float(np.linalg.norm(t2 - t1))
        baseline_residual = baseline_weight * (current_baseline - target_baseline)
        return np.append(reproj_diff, baseline_residual)

    return reproj_diff


# =============================================================================
# Sparse Jacobian Sparsity Pattern Builder
# =============================================================================

def build_jacobian_sparsity(
    n_free_cams: int,
    n_points: int,
    free_cam_map: Dict[int, int],          # Global cam_idx -> local free_cam_idx
    cam_indices: np.ndarray,               # Global cam_idx per observation
    point_indices: np.ndarray,             # Point_idx per observation
    gauge_strategy: str = "fixed_seed_pair",
    seed_cam2_global_idx: int = 1,
) -> sp.csr_matrix:
    """
    Constructs the exact sparsity pattern (binary jac_sparsity) for SciPy least_squares.

    Rows: 2 * N_obs (plus 1 if baseline constraint enabled).
    Cols: 6 * N_free_cams + 3 * N_points.

    Each 2D observation (u, v) depends on:
    - 6 parameters of the camera (if that camera is free)
    - 3 parameters of the 3D point
    """
    n_obs = len(cam_indices)
    n_residuals = 2 * n_obs
    n_params = n_free_cams * 6 + n_points * 3
    extra_row = (gauge_strategy == "fixed_first_and_baseline_constraint")
    total_rows = n_residuals + (1 if extra_row else 0)

    rows: List[int] = []
    cols: List[int] = []

    pt_col_offset = n_free_cams * 6

    for obs_idx in range(n_obs):
        g_cam_idx = int(cam_indices[obs_idx])
        pt_idx = int(point_indices[obs_idx])

        r_u = 2 * obs_idx
        r_v = 2 * obs_idx + 1

        # Camera dependency
        if g_cam_idx in free_cam_map:
            l_cam_idx = free_cam_map[g_cam_idx]
            cam_start_col = l_cam_idx * 6
            for c_offset in range(6):
                c_col = cam_start_col + c_offset
                rows.extend([r_u, r_v])
                cols.extend([c_col, c_col])

        # Point dependency
        pt_start_col = pt_col_offset + pt_idx * 3
        for p_offset in range(3):
            p_col = pt_start_col + p_offset
            rows.extend([r_u, r_v])
            cols.extend([p_col, p_col])

    # Baseline constraint dependency
    if extra_row and (seed_cam2_global_idx in free_cam_map):
        b_row = n_residuals
        l_cam_idx = free_cam_map[seed_cam2_global_idx]
        # Baseline depends on tvec of seed_cam2 (indices 3, 4, 5 of that camera)
        for t_offset in [3, 4, 5]:
            rows.append(b_row)
            cols.append(l_cam_idx * 6 + t_offset)

    data = np.ones(len(rows), dtype=bool)
    A_sparse = sp.csr_matrix((data, (rows, cols)), shape=(total_rows, n_params), dtype=bool)
    return A_sparse


# =============================================================================
# Observation Graph Construction & Validation
# =============================================================================

class ObservationGraph:
    """
    Constructs, indexes and validates the complete camera-point observation graph
    prior to Bundle Adjustment.
    """
    def __init__(
        self,
        registered_cameras: List[str],
        camera_poses: Dict[str, Dict[str, Any]],
        sparse_points: pd.DataFrame,
        track_observations: Dict[int, Dict[str, Tuple[float, float]]],
        image_shape: Tuple[int, int] = (2272, 1704),  # (H, W)
    ):
        self.camera_ids: List[str] = list(registered_cameras)
        self.cam_id_to_idx: Dict[str, int] = {cid: idx for idx, cid in enumerate(self.camera_ids)}
        self.idx_to_cam_id: Dict[int, str] = {idx: cid for idx, cid in enumerate(self.camera_ids)}

        self.camera_poses = camera_poses
        self.sparse_points = sparse_points
        self.track_obs = track_observations
        self.image_shape = image_shape  # (H, W)

        self.observations: List[BAObservation] = []
        self.point_ids: List[int] = []
        self.point_id_to_idx: Dict[int, int] = {}
        self.track_id_to_point_idx: Dict[int, int] = {}

        self.initial_points_3d: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.point_rgb: np.ndarray = np.empty((0, 3), dtype=np.uint8)
        self.point_metadata: List[Dict[str, Any]] = []

        self.rejection_log: List[Dict[str, Any]] = []

    def build_graph(self) -> None:
        """Assembles and verifies all valid observations."""
        h, w = self.image_shape

        # Index 3D points
        n_pts = len(self.sparse_points)
        self.initial_points_3d = np.zeros((n_pts, 3), dtype=np.float64)
        self.point_rgb = np.zeros((n_pts, 3), dtype=np.uint8)

        for p_idx, (_, row) in enumerate(self.sparse_points.iterrows()):
            pid = int(row["point_id"])
            tid = int(row["track_id"])
            self.point_ids.append(pid)
            self.point_id_to_idx[pid] = p_idx
            self.track_id_to_point_idx[tid] = p_idx
            self.initial_points_3d[p_idx] = [float(row["X"]), float(row["Y"]), float(row["Z"])]

            r_val = int(row.get("r", 128))
            g_val = int(row.get("g", 128))
            b_val = int(row.get("b", 128))
            self.point_rgb[p_idx] = [r_val, g_val, b_val]

            self.point_metadata.append({
                "point_id": pid,
                "track_id": tid,
                "is_seed": bool(row.get("is_seed", False)),
                "registered_views": str(row.get("registered_views", "")),
            })

        # Process observations from feature tracks
        obs_counter = 0
        duplicate_check: Set[Tuple[int, int]] = set()

        for p_idx, meta in enumerate(self.point_metadata):
            tid = meta["track_id"]
            pid = meta["point_id"]

            if tid not in self.track_obs:
                self.rejection_log.append({
                    "reason": "TRACK_NOT_IN_DATABASE",
                    "point_id": pid,
                    "track_id": tid,
                    "camera_id": "NONE",
                })
                continue

            track_dict = self.track_obs[tid]
            reg_views_str = meta["registered_views"]
            reg_views_set = set(v.strip() for v in reg_views_str.split(",") if v.strip())

            for cam_id, (u, v) in track_dict.items():
                # 1. Camera registered?
                if cam_id not in self.cam_id_to_idx:
                    continue

                cam_idx = self.cam_id_to_idx[cam_id]

                # 2. View in registered views for this point?
                if cam_id not in reg_views_set:
                    # Observation exists in track but was not part of the validated SfM point views
                    # We only include observations from cameras that registered this point
                    continue

                # 3. Finite coordinates?
                if not (np.isfinite(u) and np.isfinite(v)):
                    self.rejection_log.append({
                        "reason": "NON_FINITE_COORDINATES",
                        "point_id": pid,
                        "track_id": tid,
                        "camera_id": cam_id,
                        "coords": (u, v),
                    })
                    continue

                # 4. In image bounds?
                if not (0.0 <= u < w and 0.0 <= v < h):
                    self.rejection_log.append({
                        "reason": "OUT_OF_IMAGE_BOUNDS",
                        "point_id": pid,
                        "track_id": tid,
                        "camera_id": cam_id,
                        "coords": (u, v),
                    })
                    continue

                # 5. Duplicate check
                pair_key = (cam_idx, p_idx)
                if pair_key in duplicate_check:
                    self.rejection_log.append({
                        "reason": "DUPLICATE_CAMERA_POINT_OBSERVATION",
                        "point_id": pid,
                        "track_id": tid,
                        "camera_id": cam_id,
                    })
                    continue
                duplicate_check.add(pair_key)

                obs = BAObservation(
                    obs_id=obs_counter,
                    camera_id=cam_id,
                    cam_idx=cam_idx,
                    point_id=pid,
                    point_idx=p_idx,
                    track_id=tid,
                    u_obs=float(u),
                    v_obs=float(v),
                )
                self.observations.append(obs)
                obs_counter += 1

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts synchronized arrays for vectorized residual computation:
        - cam_indices: Shape (N_obs,)
        - point_indices: Shape (N_obs,)
        - observed_2d: Shape (N_obs, 2)
        """
        n_obs = len(self.observations)
        cam_indices = np.empty(n_obs, dtype=np.int32)
        point_indices = np.empty(n_obs, dtype=np.int32)
        observed_2d = np.empty((n_obs, 2), dtype=np.float64)

        for i, obs in enumerate(self.observations):
            cam_indices[i] = obs.cam_idx
            point_indices[i] = obs.point_idx
            observed_2d[i, 0] = obs.u_obs
            observed_2d[i, 1] = obs.v_obs

        return cam_indices, point_indices, observed_2d


# =============================================================================
# Optimization Execution Function
# =============================================================================

def run_sparse_bundle_adjustment(
    obs_graph: ObservationGraph,
    K: np.ndarray,
    config: Optional[BAConfig] = None,
) -> BAResult:
    """
    Executes fixed-intrinsics Sparse Bundle Adjustment using SciPy least_squares.
    """
    if config is None:
        config = BAConfig()

    t_start = time.time()
    n_total_cams = len(obs_graph.camera_ids)
    n_points = len(obs_graph.initial_points_3d)

    cam_indices, point_indices, observed_2d = obs_graph.get_arrays()
    n_obs = len(cam_indices)

    # 1. Determine Gauge Fixing & Camera Partition
    # seed pair: im_0010 (idx 0), im_0011 (idx 1)
    seed_cam1_id = "im_0010"
    seed_cam2_id = "im_0011"
    seed_cam1_idx = obs_graph.cam_id_to_idx[seed_cam1_id]
    seed_cam2_idx = obs_graph.cam_id_to_idx[seed_cam2_id]

    fixed_cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    free_cam_indices: List[int] = []

    # Read initial poses
    init_rvecs = np.zeros((n_total_cams, 3), dtype=np.float64)
    init_tvecs = np.zeros((n_total_cams, 3), dtype=np.float64)

    for cid, c_idx in obs_graph.cam_id_to_idx.items():
        pose = obs_graph.camera_poses[cid]
        R = np.array(pose["R_matrix"], dtype=np.float64)
        t = np.array(pose["t_vector"], dtype=np.float64).ravel()
        init_rvecs[c_idx] = matrix_to_rodrigues(R)
        init_tvecs[c_idx] = t

    target_baseline = float(np.linalg.norm(init_tvecs[seed_cam2_idx] - init_tvecs[seed_cam1_idx]))

    if config.gauge_strategy == "fixed_seed_pair":
        # Fix both im_0010 and im_0011
        fixed_cam_poses[seed_cam1_idx] = (init_rvecs[seed_cam1_idx].copy(), init_tvecs[seed_cam1_idx].copy())
        fixed_cam_poses[seed_cam2_idx] = (init_rvecs[seed_cam2_idx].copy(), init_tvecs[seed_cam2_idx].copy())
        free_cam_indices = [i for i in range(n_total_cams) if i not in (seed_cam1_idx, seed_cam2_idx)]
    elif config.gauge_strategy == "fixed_first_and_baseline_constraint":
        # Fix im_0010 only, optimize im_0011 with soft baseline constraint
        fixed_cam_poses[seed_cam1_idx] = (init_rvecs[seed_cam1_idx].copy(), init_tvecs[seed_cam1_idx].copy())
        free_cam_indices = [i for i in range(n_total_cams) if i != seed_cam1_idx]
    elif config.gauge_strategy == "fixed_first_camera":
        # Fix im_0010 only
        fixed_cam_poses[seed_cam1_idx] = (init_rvecs[seed_cam1_idx].copy(), init_tvecs[seed_cam1_idx].copy())
        free_cam_indices = [i for i in range(n_total_cams) if i != seed_cam1_idx]
    else:
        raise ValueError(f"Unknown gauge strategy: {config.gauge_strategy}")

    n_free_cams = len(free_cam_indices)
    free_cam_map = {global_idx: local_idx for local_idx, global_idx in enumerate(free_cam_indices)}

    # Initial parameter vectors
    free_init_rvecs = init_rvecs[free_cam_indices].copy()
    free_init_tvecs = init_tvecs[free_cam_indices].copy()
    points_3d_init = obs_graph.initial_points_3d.copy()

    x0 = pack_optimization_params(free_init_rvecs, free_init_tvecs, points_3d_init)

    # Initial evaluation
    initial_res = compute_residuals(
        x0,
        n_free_cams,
        n_points,
        free_cam_indices,
        fixed_cam_poses,
        n_total_cams,
        cam_indices,
        point_indices,
        observed_2d,
        K,
        gauge_strategy=config.gauge_strategy,
        seed_cam1_idx=seed_cam1_idx,
        seed_cam2_idx=seed_cam2_idx,
        target_baseline=target_baseline,
        baseline_weight=config.baseline_weight,
    )
    initial_cost = 0.5 * float(np.sum(initial_res[: 2 * n_obs] ** 2))
    initial_reproj_magnitudes = np.hypot(initial_res[: 2 * n_obs:2], initial_res[1: 2 * n_obs:2])
    init_mean_err = float(np.mean(initial_reproj_magnitudes))
    init_med_err = float(np.median(initial_reproj_magnitudes))
    init_p90_err = float(np.percentile(initial_reproj_magnitudes, 90))
    init_p95_err = float(np.percentile(initial_reproj_magnitudes, 95))
    init_max_err = float(np.max(initial_reproj_magnitudes))

    # Initial loop-closure gap between im_0072 and im_0001 (if present)
    has_loop_cams = ("im_0072" in obs_graph.cam_id_to_idx) and ("im_0001" in obs_graph.cam_id_to_idx)
    if has_loop_cams:
        c72_idx = obs_graph.cam_id_to_idx["im_0072"]
        c01_idx = obs_graph.cam_id_to_idx["im_0001"]
        c72_center_init = compute_camera_center(rodrigues_to_matrix(init_rvecs[c72_idx]), init_tvecs[c72_idx])
        c01_center_init = compute_camera_center(rodrigues_to_matrix(init_rvecs[c01_idx]), init_tvecs[c01_idx])
        init_lc_gap = float(np.linalg.norm(c72_center_init - c01_center_init))
    else:
        init_lc_gap = 0.0

    # 2. Build Sparse Jacobian Sparsity Pattern
    A_sparse = build_jacobian_sparsity(
        n_free_cams,
        n_points,
        free_cam_map,
        cam_indices,
        point_indices,
        gauge_strategy=config.gauge_strategy,
        seed_cam2_global_idx=seed_cam2_idx,
    )

    # 3. Objective Function for least_squares
    def objective_fn(x: np.ndarray) -> np.ndarray:
        return compute_residuals(
            x,
            n_free_cams,
            n_points,
            free_cam_indices,
            fixed_cam_poses,
            n_total_cams,
            cam_indices,
            point_indices,
            observed_2d,
            K,
            gauge_strategy=config.gauge_strategy,
            seed_cam1_idx=seed_cam1_idx,
            seed_cam2_idx=seed_cam2_idx,
            target_baseline=target_baseline,
            baseline_weight=config.baseline_weight,
        )

    # 4. Execute Optimization
    opt_res = least_squares(
        objective_fn,
        x0,
        jac_sparsity=A_sparse,
        method="trf",
        loss=config.loss,
        f_scale=config.f_scale,
        max_nfev=config.max_nfev,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        verbose=config.verbose,
    )

    t_end = time.time()
    runtime = t_end - t_start

    # 5. Extract Optimized Parameters
    opt_free_rvecs, opt_free_tvecs, opt_points_3d = unpack_optimization_params(
        opt_res.x, n_free_cams, n_points
    )
    all_opt_rvecs, all_opt_tvecs = assemble_all_camera_poses(
        opt_free_rvecs, opt_free_tvecs, free_cam_indices, fixed_cam_poses, n_total_cams
    )

    # Final residuals
    final_res = objective_fn(opt_res.x)
    final_cost = 0.5 * float(np.sum(final_res[: 2 * n_obs] ** 2))
    final_reproj_magnitudes = np.hypot(final_res[: 2 * n_obs:2], final_res[1: 2 * n_obs:2])
    fin_mean_err = float(np.mean(final_reproj_magnitudes))
    fin_med_err = float(np.median(final_reproj_magnitudes))
    fin_p90_err = float(np.percentile(final_reproj_magnitudes, 90))
    fin_p95_err = float(np.percentile(final_reproj_magnitudes, 95))
    fin_max_err = float(np.max(final_reproj_magnitudes))

    # Final loop-closure gap
    if has_loop_cams:
        c72_center_fin = compute_camera_center(rodrigues_to_matrix(all_opt_rvecs[c72_idx]), all_opt_tvecs[c72_idx])
        c01_center_fin = compute_camera_center(rodrigues_to_matrix(all_opt_rvecs[c01_idx]), all_opt_tvecs[c01_idx])
        fin_lc_gap = float(np.linalg.norm(c72_center_fin - c01_center_fin))
    else:
        fin_lc_gap = 0.0

    # 6. Format Optimized Camera Dictionary
    opt_cam_poses: Dict[str, Dict[str, Any]] = {}
    cam_rows = []

    for cid, c_idx in obs_graph.cam_id_to_idx.items():
        R_opt = rodrigues_to_matrix(all_opt_rvecs[c_idx])
        t_opt = all_opt_tvecs[c_idx]
        C_opt = compute_camera_center(R_opt, t_opt)
        is_ortho, ortho_err, det = check_rotation_matrix(R_opt)

        # Per-camera residuals
        mask_cam = (cam_indices == c_idx)
        cam_res_mags = final_reproj_magnitudes[mask_cam]
        cam_n_obs = len(cam_res_mags)

        if cam_n_obs > 0:
            c_mean_err = float(np.mean(cam_res_mags))
            c_med_err = float(np.median(cam_res_mags))
            c_p90_err = float(np.percentile(cam_res_mags, 90))
            c_p95_err = float(np.percentile(cam_res_mags, 95))
            c_max_err = float(np.max(cam_res_mags))
        else:
            c_mean_err = c_med_err = c_p90_err = c_p95_err = c_max_err = 0.0

        orig_pose = obs_graph.camera_poses[cid]
        opt_cam_poses[cid] = {
            "image_id": cid,
            "registration_order": orig_pose.get("registration_order", 0),
            "status": "OPTIMIZED",
            "is_gauge_fixed": (c_idx in fixed_cam_poses),
            "num_observations": cam_n_obs,
            "reproj_mean_px": round(c_mean_err, 4),
            "reproj_median_px": round(c_med_err, 4),
            "reproj_90th_px": round(c_p90_err, 4),
            "reproj_95th_px": round(c_p95_err, 4),
            "reproj_max_px": round(c_max_err, 4),
            "rotation_orthogonality_error": ortho_err,
            "rotation_determinant": det,
            "center_x": round(float(C_opt[0]), 6),
            "center_y": round(float(C_opt[1]), 6),
            "center_z": round(float(C_opt[2]), 6),
            "t_x": round(float(t_opt[0]), 6),
            "t_y": round(float(t_opt[1]), 6),
            "t_z": round(float(t_opt[2]), 6),
            "R_matrix": R_opt.tolist(),
            "t_vector": t_opt.tolist(),
            "camera_center": C_opt.tolist(),
        }

        cam_rows.append({
            "image_id": cid,
            "registration_order": orig_pose.get("registration_order", 0),
            "is_gauge_fixed": (c_idx in fixed_cam_poses),
            "num_observations": cam_n_obs,
            "reproj_mean_px": c_mean_err,
            "reproj_median_px": c_med_err,
            "reproj_90th_px": c_p90_err,
            "reproj_95th_px": c_p95_err,
            "reproj_max_px": c_max_err,
            "center_x": float(C_opt[0]),
            "center_y": float(C_opt[1]),
            "center_z": float(C_opt[2]),
            "t_x": float(t_opt[0]),
            "t_y": float(t_opt[1]),
            "t_z": float(t_opt[2]),
        })

    df_cameras = pd.DataFrame(cam_rows).sort_values("registration_order").reset_index(drop=True)

    # 7. Format Optimized Points
    opt_points_dict: Dict[int, Dict[str, Any]] = {}
    point_rows = []

    # Map observations to points for per-point error
    pt_obs_errors: Dict[int, List[float]] = {p_idx: [] for p_idx in range(n_points)}
    for o_idx in range(n_obs):
        pt_obs_errors[point_indices[o_idx]].append(final_reproj_magnitudes[o_idx])

    for p_idx in range(n_points):
        meta = obs_graph.point_metadata[p_idx]
        pid = meta["point_id"]
        tid = meta["track_id"]
        xyz = opt_points_3d[p_idx]
        rgb = obs_graph.point_rgb[p_idx]

        errors = pt_obs_errors[p_idx]
        p_mean_err = float(np.mean(errors)) if errors else 0.0

        p_info = {
            "point_id": pid,
            "track_id": tid,
            "X": round(float(xyz[0]), 6),
            "Y": round(float(xyz[1]), 6),
            "Z": round(float(xyz[2]), 6),
            "r": int(rgb[0]),
            "g": int(rgb[1]),
            "b": int(rgb[2]),
            "is_seed": meta["is_seed"],
            "registered_views_count": len(errors),
            "registered_views": meta["registered_views"],
            "mean_reproj_error_px": round(p_mean_err, 4),
        }
        opt_points_dict[pid] = p_info
        point_rows.append(p_info)

    df_points = pd.DataFrame(point_rows)

    # 8. Outlier observations table
    outlier_list = []
    for o_idx in range(n_obs):
        res_mag = final_reproj_magnitudes[o_idx]
        if res_mag > config.outlier_threshold_px:
            obs = obs_graph.observations[o_idx]
            outlier_list.append({
                "obs_id": obs.obs_id,
                "camera_id": obs.camera_id,
                "point_id": obs.point_id,
                "track_id": obs.track_id,
                "u_obs": obs.u_obs,
                "v_obs": obs.v_obs,
                "residual_px": round(float(res_mag), 4),
                "rejection_reason": f"EXCEEDS_THRESHOLD_{config.outlier_threshold_px}PX",
            })
    df_outliers = pd.DataFrame(outlier_list)

    return BAResult(
        success=bool(opt_res.success),
        status=int(opt_res.status),
        message=str(opt_res.message),
        nfev=int(opt_res.nfev),
        nit=int(getattr(opt_res, "nit", opt_res.nfev)),
        initial_cost=initial_cost,
        final_cost=final_cost,
        initial_mean_reproj_px=init_mean_err,
        final_mean_reproj_px=fin_mean_err,
        initial_median_reproj_px=init_med_err,
        final_median_reproj_px=fin_med_err,
        initial_p90_reproj_px=init_p90_err,
        final_p90_reproj_px=fin_p90_err,
        initial_p95_reproj_px=init_p95_err,
        final_p95_reproj_px=fin_p95_err,
        initial_max_reproj_px=init_max_err,
        final_max_reproj_px=fin_max_err,
        num_cameras=n_total_cams,
        num_optimized_cameras=n_free_cams,
        num_points=n_points,
        num_optimized_points=n_points,
        num_observations=n_obs,
        runtime_sec=round(runtime, 2),
        gauge_strategy=config.gauge_strategy,
        initial_loop_closure_gap=round(init_lc_gap, 4),
        final_loop_closure_gap=round(fin_lc_gap, 4),
        optimized_camera_poses=opt_cam_poses,
        optimized_points=opt_points_dict,
        per_camera_metrics=df_cameras,
        outlier_observations=df_outliers,
    )


# =============================================================================
# State Export & Artifact Generation
# =============================================================================

def export_bundle_adjusted_reconstruction(
    ba_result: BAResult,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Exports the complete bundle adjusted reconstruction to outputs/bundle_adjustment/:
    - bundle_adjusted_state.json
    - optimized_camera_poses/ (*.json)
    - optimized_cameras.csv
    - optimized_points.csv
    - optimized_sparse_reconstruction.ply
    - optimization_history.json
    - optimization_outliers.csv
    - bundle_adjustment_metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    poses_dir = output_dir / "optimized_camera_poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # 1. Individual Camera Poses JSON
    for cid, pose_dict in ba_result.optimized_camera_poses.items():
        pose_path = poses_dir / f"{cid}.json"
        with open(pose_path, "w", encoding="utf-8") as f:
            json.dump(pose_dict, f, indent=2)
    paths["camera_poses_dir"] = poses_dir

    # 2. Optimized Cameras CSV
    cams_csv_path = output_dir / "optimized_cameras.csv"
    ba_result.per_camera_metrics.to_csv(cams_csv_path, index=False)
    paths["optimized_cameras_csv"] = cams_csv_path

    # 3. Optimized Points CSV
    pts_csv_path = output_dir / "optimized_points.csv"
    df_pts = pd.DataFrame(list(ba_result.optimized_points.values()))
    df_pts.to_csv(pts_csv_path, index=False)
    paths["optimized_points_csv"] = pts_csv_path

    # 4. Outliers CSV
    outliers_csv_path = output_dir / "optimization_outliers.csv"
    ba_result.outlier_observations.to_csv(outliers_csv_path, index=False)
    paths["optimization_outliers_csv"] = outliers_csv_path

    # 5. Optimized Sparse Reconstruction PLY
    ply_path = output_dir / "optimized_sparse_reconstruction.ply"
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Stage 4 Sparse Bundle Adjustment Refined Point Cloud\n")
        f.write(f"element vertex {len(df_pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for _, row in df_pts.iterrows():
            f.write(f"{row['X']:.6f} {row['Y']:.6f} {row['Z']:.6f} {int(row['r'])} {int(row['g'])} {int(row['b'])}\n")
    paths["optimized_ply"] = ply_path

    # 6. Optimization History JSON
    history_json_path = output_dir / "optimization_history.json"
    history_dict = {
        "status": ba_result.status,
        "success": ba_result.success,
        "message": ba_result.message,
        "nfev": ba_result.nfev,
        "nit": ba_result.nit,
        "runtime_sec": ba_result.runtime_sec,
        "initial_cost": ba_result.initial_cost,
        "final_cost": ba_result.final_cost,
        "reprojection_improvement_pct": round(100.0 * (1.0 - ba_result.final_mean_reproj_px / (ba_result.initial_mean_reproj_px + 1e-12)), 2),
        "initial_errors_px": {
            "mean": ba_result.initial_mean_reproj_px,
            "median": ba_result.initial_median_reproj_px,
            "p90": ba_result.initial_p90_reproj_px,
            "p95": ba_result.initial_p95_reproj_px,
            "max": ba_result.initial_max_reproj_px,
        },
        "final_errors_px": {
            "mean": ba_result.final_mean_reproj_px,
            "median": ba_result.final_median_reproj_px,
            "p90": ba_result.final_p90_reproj_px,
            "p95": ba_result.final_p95_reproj_px,
            "max": ba_result.final_max_reproj_px,
        },
        "initial_loop_closure_gap": ba_result.initial_loop_closure_gap,
        "final_loop_closure_gap": ba_result.final_loop_closure_gap,
    }
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    paths["optimization_history_json"] = history_json_path

    # 7. Bundle Adjusted State JSON
    state_json_path = output_dir / "bundle_adjusted_state.json"
    state_dict = {
        "dataset": "THU-MVS Dog_RGB",
        "num_registered_cameras": ba_result.num_cameras,
        "num_optimized_cameras": ba_result.num_optimized_cameras,
        "num_sparse_points": ba_result.num_points,
        "num_observations": ba_result.num_observations,
        "gauge_strategy": ba_result.gauge_strategy,
        "optimizer_status": ba_result.message,
        "reprojection_error_mean_px": ba_result.final_mean_reproj_px,
        "reprojection_error_median_px": ba_result.final_median_reproj_px,
        "reprojection_error_p90_px": ba_result.final_p90_reproj_px,
        "reprojection_error_p95_px": ba_result.final_p95_reproj_px,
        "reprojection_error_max_px": ba_result.final_max_reproj_px,
        "loop_closure_gap_units": ba_result.final_loop_closure_gap,
        "optimization_runtime_sec": ba_result.runtime_sec,
    }
    with open(state_json_path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2)
    paths["bundle_adjusted_state_json"] = state_json_path

    # 8. Bundle Adjustment Metadata JSON
    metadata_json_path = output_dir / "bundle_adjustment_metadata.json"
    metadata_dict = {
        "stage": "Stage 4 Sparse Bundle Adjustment and Global Reconstruction Refinement",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coordinate_convention": "X_cam = R @ X_world + t, C = -R.T @ t",
        "origin_reference": "im_0010 (R=I, t=0)",
        "scale_reference": "im_0010 vs im_0011 baseline = 1.0",
        "gauge_strategy": ba_result.gauge_strategy,
        "solver": "scipy.optimize.least_squares (method='trf', loss='huber', f_scale=2.0)",
        "total_cameras": ba_result.num_cameras,
        "total_points": ba_result.num_points,
        "total_observations": ba_result.num_observations,
        "outlier_count": len(ba_result.outlier_observations),
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)
    paths["bundle_adjustment_metadata_json"] = metadata_json_path

    return paths
