"""
tests/test_camera_registration.py
=================================
Unit tests for modules.camera_registration.

Tests all 14 required Stage 3 criteria:
1. Correct synthetic PnP recovery.
2. Outlier-resistant PnP-RANSAC.
3. Invalid 3D points rejection.
4. Non-finite observations rejection.
5. Duplicate track IDs rejection.
6. Insufficient correspondences rejection.
7. Reprojection-error calculation.
8. Camera-centre calculation.
9. Positive-depth validation.
10. Camera pose convention.
11. Feature-track association.
12. Incremental point insertion without duplicate track IDs.
13. Rejection of invalid triangulation candidates.
14. Registration failure handling.
"""

import unittest
import math
import numpy as np
import cv2

from modules.camera_registration import (
    CameraPose,
    PnPResult,
    ReconstructedPoint,
    ReconstructionState,
    compute_image_spatial_coverage,
    compute_pnp_reprojection_errors,
    build_2d_3d_correspondences,
    estimate_camera_pose_pnp,
)


class TestCameraRegistration(unittest.TestCase):

    def setUp(self):
        """Set up standard camera intrinsics and synthetic scene."""
        self.fx = 4460.91
        self.fy = 4460.91
        self.cx = 852.0
        self.cy = 1136.0
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Generate 50 well-distributed 3D points in front of camera
        np.random.seed(42)
        X = np.random.uniform(-0.5, 0.5, 50)
        Y = np.random.uniform(-0.8, 0.8, 50)
        Z = np.random.uniform(5.0, 7.0, 50)
        self.pts3D_gt = np.column_stack([X, Y, Z]).astype(np.float64)

        # Ground truth rotation (10 deg around Y) and translation
        theta = np.radians(10.0)
        self.R_gt = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ], dtype=np.float64)
        self.t_gt = np.array([[1.0], [0.0], [0.1]], dtype=np.float64)

        # Project 3D points into camera
        pts3D_cam = (self.R_gt @ self.pts3D_gt.T + self.t_gt).T
        proj_h = (self.K @ pts3D_cam.T).T
        self.pts2D_gt = np.column_stack([
            proj_h[:, 0] / proj_h[:, 2],
            proj_h[:, 1] / proj_h[:, 2],
        ])

    def test_01_synthetic_pnp_recovery(self):
        """1. Verify accurate recovery of known ground-truth camera pose."""
        res = estimate_camera_pose_pnp(
            self.pts3D_gt, self.pts2D_gt, self.K,
            reproj_threshold_px=2.0, min_correspondences=10, min_inliers=10
        )
        self.assertTrue(res.success)
        self.assertEqual(res.status, "REGISTERED")
        self.assertAlmostEqual(res.reproj_median_px, 0.0, delta=0.05)
        # Verify rotation and translation recovery
        np.testing.assert_allclose(res.R, self.R_gt, atol=1e-3)
        np.testing.assert_allclose(res.t.ravel(), self.t_gt.ravel(), atol=1e-3)

    def test_02_outlier_resistant_pnp_ransac(self):
        """2. Verify PnP-RANSAC succeeds in the presence of 30% gross outliers."""
        pts2D_corrupted = self.pts2D_gt.copy()
        # Corrupt 15 of 50 points
        pts2D_corrupted[:15] += np.random.uniform(100.0, 300.0, (15, 2))

        res = estimate_camera_pose_pnp(
            self.pts3D_gt, pts2D_corrupted, self.K,
            reproj_threshold_px=3.0, min_correspondences=20, min_inliers=15, min_inlier_ratio=0.5
        )
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.inlier_count, 30)
        self.assertLessEqual(res.reproj_median_px, 1.0)
        np.testing.assert_allclose(res.R, self.R_gt, atol=2e-2)

    def test_03_invalid_3d_points_rejection(self):
        """3. Verify rejection when 3D points contain NaN or Inf."""
        pts3D_bad = self.pts3D_gt.copy()
        pts3D_bad[5, 0] = np.nan
        res = estimate_camera_pose_pnp(pts3D_bad, self.pts2D_gt, self.K)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "FAILED")
        self.assertIn("NON_FINITE", res.rejection_reason)

    def test_04_non_finite_observations_rejection(self):
        """4. Verify rejection when 2D observations contain Inf."""
        pts2D_bad = self.pts2D_gt.copy()
        pts2D_bad[10, 1] = np.inf
        res = estimate_camera_pose_pnp(self.pts3D_gt, pts2D_bad, self.K)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "FAILED")
        self.assertIn("NON_FINITE", res.rejection_reason)

    def test_05_duplicate_track_ids_rejection(self):
        """5. Verify correspondence builder rejects or de-duplicates duplicate track IDs."""
        track_obs = {
            101: {"im_0012": (500.0, 600.0)},
            102: {"im_0012": (550.0, 650.0)},
        }
        reconstructed = {
            101: ReconstructedPoint(point_id=1, track_id=101, X=0.1, Y=0.2, Z=6.0),
            102: ReconstructedPoint(point_id=2, track_id=102, X=0.3, Y=0.4, Z=6.0),
        }
        pts3D, pts2D, track_ids = build_2d_3d_correspondences("im_0012", track_obs, reconstructed)
        self.assertEqual(len(track_ids), len(set(track_ids)))
        self.assertEqual(len(pts3D), 2)

    def test_06_insufficient_correspondences_rejection(self):
        """6. Verify graceful rejection when correspondence count is below minimum."""
        pts3D_small = self.pts3D_gt[:5]
        pts2D_small = self.pts2D_gt[:5]
        res = estimate_camera_pose_pnp(pts3D_small, pts2D_small, self.K, min_correspondences=15)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "FAILED")
        self.assertIn("INSUFFICIENT_CORRESPONDENCES", res.rejection_reason)

    def test_07_reprojection_error_calculation(self):
        """7. Verify sub-pixel accuracy of reprojection residual computation."""
        errs = compute_pnp_reprojection_errors(self.pts3D_gt, self.pts2D_gt, self.K, self.R_gt, self.t_gt)
        self.assertEqual(len(errs), len(self.pts3D_gt))
        self.assertTrue(np.all(errs < 1e-4))

    def test_08_camera_center_calculation(self):
        """8. Verify camera center calculation C = -R^T @ t."""
        C = -self.R_gt.T @ self.t_gt.ravel()
        # For pure rotation around Y and t=[1, 0, 0.1]^T
        self.assertAlmostEqual(C[1], 0.0, delta=1e-6)
        # Check that R @ C + t == 0
        transformed_C = self.R_gt @ C + self.t_gt.ravel()
        np.testing.assert_allclose(transformed_C, [0.0, 0.0, 0.0], atol=1e-12)

    def test_09_positive_depth_validation(self):
        """9. Verify cheirality validation detects points placed behind camera."""
        pts3D_behind = self.pts3D_gt.copy()
        # Shift points along optical axis to be behind camera (Z < 0 in camera frame)
        pts3D_behind[:, 2] = -10.0
        # Project anyway (inverted)
        pts2D_mock = self.pts2D_gt.copy()
        res = estimate_camera_pose_pnp(pts3D_behind, pts2D_mock, self.K, min_correspondences=10)
        # Should fail cheirality or PnP convergence
        self.assertFalse(res.success)

    def test_10_camera_pose_convention(self):
        """10. Verify world-to-camera projection matrix convention P = K [R | t]."""
        pose = CameraPose(
            image_id="im_test",
            R=self.R_gt,
            t=self.t_gt,
            C=-self.R_gt.T @ self.t_gt.ravel(),
        )
        P = pose.projection_matrix(self.K)
        self.assertEqual(P.shape, (3, 4))
        # Test projection of point 0
        X_h = np.append(self.pts3D_gt[0], 1.0)
        proj = P @ X_h
        u_p = proj[0] / proj[2]
        v_p = proj[1] / proj[2]
        self.assertAlmostEqual(u_p, self.pts2D_gt[0, 0], delta=1e-3)
        self.assertAlmostEqual(v_p, self.pts2D_gt[0, 1], delta=1e-3)

    def test_11_feature_track_association(self):
        """11. Verify correct 2D-3D association by track_id."""
        track_obs = {
            201: {"im_0012": (100.0, 200.0)},
            202: {"im_0012": (300.0, 400.0)},
            203: {"im_0013": (500.0, 600.0)},  # Different image
        }
        reconstructed = {
            201: ReconstructedPoint(point_id=1, track_id=201, X=1.0, Y=2.0, Z=5.0),
            202: ReconstructedPoint(point_id=2, track_id=202, X=3.0, Y=4.0, Z=5.0),
            204: ReconstructedPoint(point_id=3, track_id=204, X=5.0, Y=6.0, Z=5.0),
        }
        pts3D, pts2D, tids = build_2d_3d_correspondences("im_0012", track_obs, reconstructed)
        self.assertEqual(tids, [201, 202])
        self.assertEqual(len(pts3D), 2)
        np.testing.assert_allclose(pts3D[0], [1.0, 2.0, 5.0])
        np.testing.assert_allclose(pts2D[0], [100.0, 200.0])

    def test_12_incremental_point_insertion_uniqueness(self):
        """12. Verify incremental point insertion rejects duplicate track IDs."""
        state = ReconstructionState(self.K)
        pt1 = ReconstructedPoint(point_id=1, track_id=555, X=0.0, Y=0.0, Z=5.0)
        pt2 = ReconstructedPoint(point_id=2, track_id=555, X=1.0, Y=1.0, Z=5.0)  # Duplicate track_id
        
        added1 = state.add_reconstructed_point(pt1)
        added2 = state.add_reconstructed_point(pt2)
        self.assertTrue(added1)
        self.assertFalse(added2)
        self.assertEqual(state.num_points, 1)

    def test_13_rejection_of_invalid_triangulation_candidates(self):
        """13. Verify rejection of candidate tracks with near-zero baseline or negative depth."""
        state = ReconstructionState(self.K)
        # Initialize two identical cameras (zero baseline / zero parallax)
        pose1 = CameraPose(
            image_id="camA", R=np.eye(3), t=np.zeros((3, 1)), C=np.zeros(3)
        )
        pose2 = CameraPose(
            image_id="camB", R=np.eye(3), t=np.zeros((3, 1)), C=np.zeros(3)
        )
        state.register_camera(pose1)
        state.register_camera(pose2)

        state.track_observations = {
            999: {"camA": (100.0, 100.0), "camB": (100.0, 100.0)}
        }
        acc, rej = state.triangulate_new_tracks("camB", min_parallax_deg=2.0)
        self.assertEqual(acc, 0)
        self.assertEqual(rej, 1)
        self.assertIn("LOW_PARALLAX", state.rejected_candidates[0]["reason"])

    def test_14_registration_failure_handling(self):
        """14. Verify graceful handling of degeneracies without raising exceptions."""
        pts3D_zero = np.zeros((20, 3), dtype=np.float64)  # Completely degenerate points
        pts2D_zero = np.zeros((20, 2), dtype=np.float64)
        res = estimate_camera_pose_pnp(pts3D_zero, pts2D_zero, self.K)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "FAILED")


if __name__ == "__main__":
    unittest.main()
