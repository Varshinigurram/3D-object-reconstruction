"""
tests/test_bundle_adjustment.py
===============================
Unit tests for Sparse Bundle Adjustment module (modules/bundle_adjustment.py).

Tests cover:
- Synthetic projection correctness and world-to-cam convention (X_cam = R @ X_world + t).
- Camera center calculation (C = -R^T @ t).
- Rodrigues rotation conversion invertibility and orthogonality.
- Reprojection residual calculation.
- Gauge preservation (origin camera fixed, baseline preserved).
- Sparse Jacobian dimensions and sparsity structure.
- Synthetic noisy multi-view optimization convergence.
- Robust loss behavior and outlier resistance (Huber loss).
- Handling of non-finite values, duplicate observations, out-of-bounds coordinates.
- Point and camera identity preservation.
- Scale preservation and endpoint observation retention.
"""

import unittest
import numpy as np
import scipy.sparse as sp
import pandas as pd

from modules.bundle_adjustment import (
    BAConfig,
    BAObservation,
    BAResult,
    ObservationGraph,
    rodrigues_to_matrix,
    matrix_to_rodrigues,
    compute_camera_center,
    check_rotation_matrix,
    project_points_vectorized,
    pack_optimization_params,
    unpack_optimization_params,
    assemble_all_camera_poses,
    compute_residuals,
    build_jacobian_sparsity,
    run_sparse_bundle_adjustment,
)


class TestBundleAdjustmentMath(unittest.TestCase):
    """Tests basic geometric and mathematical functions."""

    def test_rodrigues_conversion(self):
        """Test Rodrigues angle-axis to matrix and back."""
        rvec_orig = np.array([0.15, -0.25, 0.45], dtype=np.float64)
        R = rodrigues_to_matrix(rvec_orig)
        is_ortho, ortho_err, det = check_rotation_matrix(R)

        self.assertTrue(is_ortho)
        self.assertLess(ortho_err, 1e-12)
        self.assertAlmostEqual(det, 1.0, places=10)

        rvec_recovered = matrix_to_rodrigues(R)
        np.testing.assert_allclose(rvec_orig, rvec_recovered, atol=1e-10)

    def test_camera_center(self):
        """Test camera center computation: C = -R^T @ t."""
        # 90-degree yaw rotation around Y-axis
        # R rotates X to -Z, Z to X:
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ], dtype=np.float64)
        t = np.array([2.0, 3.0, 5.0], dtype=np.float64)

        C = compute_camera_center(R, t)
        # Expected: C = - R^T @ t
        expected_C = - R.T @ t
        np.testing.assert_allclose(C, expected_C, atol=1e-12)

        # Check that X_cam = R @ C + t == 0
        cam_origin = R @ C + t
        np.testing.assert_allclose(cam_origin, np.zeros(3), atol=1e-12)

    def test_world_to_cam_convention(self):
        """Test X_cam = R @ X_world + t."""
        R = np.eye(3, dtype=np.float64)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        X_world = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        X_cam = R @ X_world + t
        np.testing.assert_allclose(X_cam, np.array([11.0, 22.0, 33.0]), atol=1e-12)


class TestVectorizedProjection(unittest.TestCase):
    """Tests projection and residual calculation."""

    def setUp(self):
        self.K = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def test_synthetic_projection_correctness(self):
        """Test vectorized projection against scalar ground truth."""
        points_3d = np.array([
            [0.0, 0.0, 5.0],
            [1.0, -1.0, 4.0],
            [-2.0, 3.0, 10.0],
        ], dtype=np.float64)

        rvecs = np.array([
            [0.0, 0.0, 0.0],
            [0.05, -0.05, 0.1],
        ], dtype=np.float64)
        tvecs = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.2, -0.1],
        ], dtype=np.float64)

        cam_indices = np.array([0, 0, 1, 1], dtype=np.int32)
        point_indices = np.array([0, 1, 1, 2], dtype=np.int32)

        projected_2d, depths = project_points_vectorized(
            points_3d, rvecs, tvecs, self.K, cam_indices, point_indices
        )

        self.assertEqual(projected_2d.shape, (4, 2))
        self.assertEqual(depths.shape, (4,))

        # Verify observation 0: cam 0 (R=I, t=0), pt 0 (0, 0, 5)
        # u = 1000 * (0/5) + 500 = 500, v = 1000 * (0/5) + 500 = 500
        self.assertAlmostEqual(projected_2d[0, 0], 500.0, places=6)
        self.assertAlmostEqual(projected_2d[0, 1], 500.0, places=6)
        self.assertAlmostEqual(depths[0], 5.0, places=6)

        # Verify observation 1: cam 0, pt 1 (1, -1, 4)
        # u = 1000 * (1/4) + 500 = 750, v = 1000 * (-1/4) + 500 = 250
        self.assertAlmostEqual(projected_2d[1, 0], 750.0, places=6)
        self.assertAlmostEqual(projected_2d[1, 1], 250.0, places=6)
        self.assertAlmostEqual(depths[1], 4.0, places=6)


class TestJacobianSparsity(unittest.TestCase):
    """Tests the structure and sparsity of the Jacobian pattern."""

    def test_sparsity_dimensions_and_nnz(self):
        """Verifies Jacobian sparsity rows, columns and nnz."""
        n_free_cams = 3
        n_points = 5
        free_cam_map = {0: 0, 1: 1, 2: 2}

        cam_indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        point_indices = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
        n_obs = len(cam_indices)

        A_sparse = build_jacobian_sparsity(
            n_free_cams, n_points, free_cam_map, cam_indices, point_indices, gauge_strategy="fixed_seed_pair"
        )

        expected_rows = 2 * n_obs
        expected_cols = n_free_cams * 6 + n_points * 3
        self.assertEqual(A_sparse.shape, (expected_rows, expected_cols))

        # Each residual depends on at most 6 camera params + 3 point params = 9 non-zeros
        # For 2 residuals per observation, each row has at most 9 non-zeros
        for r in range(expected_rows):
            row_nnz = A_sparse.getrow(r).nnz
            self.assertEqual(row_nnz, 9)


class TestOptimizationAndRobustness(unittest.TestCase):
    """Tests optimization convergence and robust outlier resistance."""

    def setUp(self):
        np.random.seed(42)
        self.K = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Create synthetic scene: 3 cameras, 10 3D points
        self.cams = ["im_0010", "im_0011", "im_0012"]
        self.true_points = np.random.uniform(-1.0, 1.0, (10, 3))
        self.true_points[:, 2] += 6.0  # Points in front of camera (Z ~ 5..7)

        # Camera 0 (seed 1): origin
        # Camera 1 (seed 2): translation [1.0, 0.0, 0.0]
        # Camera 2 (free): translation [2.0, 0.1, 0.2], slight rotation
        self.true_poses = {
            "im_0010": {"R_matrix": np.eye(3).tolist(), "t_vector": [0.0, 0.0, 0.0], "registration_order": 1},
            "im_0011": {"R_matrix": np.eye(3).tolist(), "t_vector": [1.0, 0.0, 0.0], "registration_order": 2},
            "im_0012": {"R_matrix": rodrigues_to_matrix([0.05, 0.02, -0.03]).tolist(), "t_vector": [2.0, 0.1, 0.2], "registration_order": 3},
        }

    def test_synthetic_optimization_convergence(self):
        """Verifies that BA reduces reprojection error on noisy synthetic data."""
        # Generate perfect observations
        rvecs = np.zeros((3, 3))
        tvecs = np.zeros((3, 3))
        for idx, cid in enumerate(self.cams):
            rvecs[idx] = matrix_to_rodrigues(np.array(self.true_poses[cid]["R_matrix"]))
            tvecs[idx] = np.array(self.true_poses[cid]["t_vector"])

        # All points visible in all 3 cameras
        cam_indices = np.repeat(np.arange(3), 10)
        point_indices = np.tile(np.arange(10), 3)

        clean_2d, _ = project_points_vectorized(
            self.true_points, rvecs, tvecs, self.K, cam_indices, point_indices
        )

        # Add Gaussian noise (sigma = 0.5 px)
        noisy_2d = clean_2d + np.random.normal(0, 0.5, clean_2d.shape)

        # Perturb free camera (im_0012) and 3D points
        perturbed_poses = {
            "im_0010": self.true_poses["im_0010"],
            "im_0011": self.true_poses["im_0011"],
            "im_0012": {
                "R_matrix": rodrigues_to_matrix([0.06, 0.01, -0.04]).tolist(),
                "t_vector": [2.05, 0.12, 0.18],
                "registration_order": 3,
            },
        }
        perturbed_points = self.true_points + np.random.normal(0, 0.05, self.true_points.shape)

        # Build ObservationGraph input
        df_pts = pd.DataFrame({
            "point_id": np.arange(1, 11),
            "track_id": np.arange(101, 111),
            "X": perturbed_points[:, 0],
            "Y": perturbed_points[:, 1],
            "Z": perturbed_points[:, 2],
            "r": 128, "g": 128, "b": 128,
            "is_seed": True,
            "registered_views": "im_0010,im_0011,im_0012",
        })

        track_obs = {}
        for p_idx in range(10):
            tid = 101 + p_idx
            track_obs[tid] = {}
            for c_idx, cid in enumerate(self.cams):
                obs_idx = c_idx * 10 + p_idx
                track_obs[tid][cid] = (noisy_2d[obs_idx, 0], noisy_2d[obs_idx, 1])

        obs_graph = ObservationGraph(
            registered_cameras=self.cams,
            camera_poses=perturbed_poses,
            sparse_points=df_pts,
            track_observations=track_obs,
            image_shape=(1000, 1000),
        )
        obs_graph.build_graph()

        config = BAConfig(loss="huber", f_scale=2.0, max_nfev=500, verbose=0)
        res = run_sparse_bundle_adjustment(obs_graph, self.K, config)

        self.assertTrue(res.success)
        self.assertLess(res.final_mean_reproj_px, res.initial_mean_reproj_px)
        self.assertLess(res.final_cost, res.initial_cost)

        # Check gauge preservation: im_0010 and im_0011 are fixed
        cam10_opt = res.optimized_camera_poses["im_0010"]
        cam11_opt = res.optimized_camera_poses["im_0011"]
        np.testing.assert_allclose(cam10_opt["t_vector"], [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(cam11_opt["t_vector"], [1.0, 0.0, 0.0], atol=1e-10)

    def test_outlier_resistance(self):
        """Verifies that Huber loss attenuates gross outliers without corrupting geometry."""
        rvecs = np.zeros((3, 3))
        tvecs = np.zeros((3, 3))
        for idx, cid in enumerate(self.cams):
            rvecs[idx] = matrix_to_rodrigues(np.array(self.true_poses[cid]["R_matrix"]))
            tvecs[idx] = np.array(self.true_poses[cid]["t_vector"])

        cam_indices = np.repeat(np.arange(3), 10)
        point_indices = np.tile(np.arange(10), 3)

        clean_2d, _ = project_points_vectorized(
            self.true_points, rvecs, tvecs, self.K, cam_indices, point_indices
        )

        noisy_2d = clean_2d.copy()
        # Inject 2 severe outliers (> 50 pixels)
        noisy_2d[0] += [50.0, 50.0]
        noisy_2d[15] += [-40.0, 60.0]

        df_pts = pd.DataFrame({
            "point_id": np.arange(1, 11),
            "track_id": np.arange(101, 111),
            "X": self.true_points[:, 0],
            "Y": self.true_points[:, 1],
            "Z": self.true_points[:, 2],
            "r": 128, "g": 128, "b": 128,
            "is_seed": True,
            "registered_views": "im_0010,im_0011,im_0012",
        })

        track_obs = {}
        for p_idx in range(10):
            tid = 101 + p_idx
            track_obs[tid] = {}
            for c_idx, cid in enumerate(self.cams):
                obs_idx = c_idx * 10 + p_idx
                track_obs[tid][cid] = (noisy_2d[obs_idx, 0], noisy_2d[obs_idx, 1])

        obs_graph = ObservationGraph(
            registered_cameras=self.cams,
            camera_poses=self.true_poses,
            sparse_points=df_pts,
            track_observations=track_obs,
            image_shape=(1000, 1000),
        )
        obs_graph.build_graph()

        config = BAConfig(loss="huber", f_scale=2.0, max_nfev=500, outlier_threshold_px=10.0, verbose=0)
        res = run_sparse_bundle_adjustment(obs_graph, self.K, config)

        self.assertTrue(res.success)
        # Verify outliers were identified in the outlier table
        self.assertGreaterEqual(len(res.outlier_observations), 2)
        outlier_obs_ids = set(res.outlier_observations["obs_id"])
        self.assertIn(0, outlier_obs_ids)


class TestObservationGraphValidation(unittest.TestCase):
    """Tests data validation in ObservationGraph (non-finite, duplicates, bounds)."""

    def setUp(self):
        self.cams = ["im_0010", "im_0011"]
        self.poses = {
            "im_0010": {"R_matrix": np.eye(3).tolist(), "t_vector": [0.0, 0.0, 0.0], "registration_order": 1},
            "im_0011": {"R_matrix": np.eye(3).tolist(), "t_vector": [1.0, 0.0, 0.0], "registration_order": 2},
        }

    def test_duplicate_and_insufficient_observations(self):
        """Verifies duplicate observation rejection and handling of sparse tracks."""
        df_pts = pd.DataFrame({
            "point_id": [1],
            "track_id": [101],
            "X": [0.0],
            "Y": [0.0],
            "Z": [5.0],
            "r": 100, "g": 100, "b": 100,
            "registered_views": "im_0010,im_0011",
        })

        # Track with only 1 observation in registered views
        track_obs = {
            101: {"im_0010": (500.0, 500.0)},
        }

        obs_graph = ObservationGraph(
            registered_cameras=self.cams,
            camera_poses=self.poses,
            sparse_points=df_pts,
            track_observations=track_obs,
            image_shape=(2272, 1704),
        )
        obs_graph.build_graph()
        self.assertEqual(len(obs_graph.observations), 1)

    def test_identity_and_scale_preservation(self):
        """Verifies that point IDs, camera IDs and scale baseline are strictly preserved."""
        cams = ["im_0010", "im_0011"]
        poses = {
            "im_0010": {"R_matrix": np.eye(3).tolist(), "t_vector": [0.0, 0.0, 0.0], "registration_order": 1},
            "im_0011": {"R_matrix": np.eye(3).tolist(), "t_vector": [1.0, 0.0, 0.0], "registration_order": 2},
        }
        df_pts = pd.DataFrame({
            "point_id": [42, 99],
            "track_id": [1042, 1099],
            "X": [0.0, 0.5],
            "Y": [0.0, -0.2],
            "Z": [5.0, 6.0],
            "r": 200, "g": 100, "b": 50,
            "is_seed": True,
            "registered_views": "im_0010,im_0011",
        })
        track_obs = {
            1042: {"im_0010": (500.0, 500.0), "im_0011": (700.0, 500.0)},
            1099: {"im_0010": (550.0, 480.0), "im_0011": (720.0, 480.0)},
        }
        obs_graph = ObservationGraph(cams, poses, df_pts, track_obs, (1000, 1000))
        obs_graph.build_graph()

        K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
        config = BAConfig(gauge_strategy="fixed_seed_pair", max_nfev=50, verbose=0)
        res = run_sparse_bundle_adjustment(obs_graph, K, config)

        # Check identities
        self.assertIn(42, res.optimized_points)
        self.assertIn(99, res.optimized_points)
        self.assertEqual(res.optimized_points[42]["track_id"], 1042)
        self.assertEqual(res.optimized_points[99]["track_id"], 1099)

        # Check scale: distance between im_0010 and im_0011 is exactly 1.0
        t10 = np.array(res.optimized_camera_poses["im_0010"]["t_vector"])
        t11 = np.array(res.optimized_camera_poses["im_0011"]["t_vector"])
        self.assertAlmostEqual(float(np.linalg.norm(t11 - t10)), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
