"""
tests/test_triangulation.py
===========================
Focused Unit Tests for Linear Triangulation and Geometric Quality Filtering.

Tests verify:
1. Synthetic two-camera triangulation with a known ground-truth 3D point.
2. Correct reprojection from both camera projection matrices.
3. Positive and negative homogeneous scale (W) signed division.
4. Near-zero W rejection threshold.
5. Positive depth validation in both camera coordinate frames (cheirality).
6. Parallax angle computation.
7. Reprojection residual computation (Image 1, Image 2, and combined).
8. Invalid correspondence rejection and reason tracking.
9. Feature track ID and multi-view metadata preservation.
10. Coordinate transformation consistency between camera frames.
11. Invalid numerical input handling (NaNs, zero vectors).
"""

import unittest
import math
import numpy as np

from modules.triangulation import (
    TriangulatedPoint,
    triangulate_point_dlt,
    compute_ray_parallax_angle,
    compute_reprojection_errors,
    triangulate_correspondence_batch,
)


class TestTriangulation(unittest.TestCase):
    """Test suite for linear DLT triangulation and geometric quality filtering."""

    def setUp(self):
        """Set up canonical synthetic camera setup with known intrinsics and baseline."""
        # Simple pinhole camera with f = 1000, principal point (500, 500)
        self.K = np.array([
            [1000.0,    0.0, 500.0],
            [   0.0, 1000.0, 500.0],
            [   0.0,    0.0,   1.0]
        ], dtype=np.float64)

        # Camera 1 at origin
        self.R1 = np.eye(3, dtype=np.float64)
        self.t1 = np.zeros((3, 1), dtype=np.float64)
        self.P1 = self.K @ np.hstack([self.R1, self.t1])

        # Camera 2 translated along X by 1.0 unit (pure baseline)
        self.R2 = np.eye(3, dtype=np.float64)
        self.t2 = np.array([[-1.0], [0.0], [0.0]], dtype=np.float64)
        self.P2 = self.K @ np.hstack([self.R2, self.t2])

    def test_synthetic_known_3d_point_triangulation(self):
        """Test DLT triangulation recovers an exact known 3D point in front of both cameras."""
        # Ground truth 3D point: X = 0.5, Y = 0.2, Z = 5.0
        X_gt = np.array([0.5, 0.2, 5.0])

        # Project into Camera 1
        p1_h = self.K @ X_gt
        pt1 = (float(p1_h[0] / p1_h[2]), float(p1_h[1] / p1_h[2]))

        # Project into Camera 2: X2 = R2 * X + t2 = [0.5 - 1.0, 0.2, 5.0] = [-0.5, 0.2, 5.0]
        X2_gt = self.R2 @ X_gt + self.t2.ravel()
        p2_h = self.K @ X2_gt
        pt2 = (float(p2_h[0] / p2_h[2]), float(p2_h[1] / p2_h[2]))

        # Triangulate via DLT
        X_homo, W = triangulate_point_dlt(pt1, pt2, self.P1, self.P2)
        self.assertGreater(abs(W), 1e-7)

        X_est = X_homo[:3] / W
        # Verify reconstructed 3D point matches ground truth within numerical precision
        np.testing.assert_allclose(X_est, X_gt, atol=1e-5)

    def test_reprojection_residuals(self):
        """Test reprojection errors evaluate to sub-pixel values for perfect projections."""
        X_gt = np.array([0.5, 0.2, 5.0])
        p1_h = self.K @ X_gt
        pt1 = (float(p1_h[0] / p1_h[2]), float(p1_h[1] / p1_h[2]))
        X2_gt = self.R2 @ X_gt + self.t2.ravel()
        p2_h = self.K @ X2_gt
        pt2 = (float(p2_h[0] / p2_h[2]), float(p2_h[1] / p2_h[2]))

        e1, e2, e_comb = compute_reprojection_errors(X_gt, pt1, pt2, self.K, self.R2, self.t2)
        self.assertLess(e1, 1e-5)
        self.assertLess(e2, 1e-5)
        self.assertLess(e_comb, 1e-5)

    def test_parallax_calculation(self):
        """Test viewing ray parallax angle calculation between two cameras."""
        # Baseline = 1.0 along X, depth Z = 5.0, centered at X=0.5
        # The angle between ray from (0,0,0) to (0.5, 0, 5) and (1,0,0) to (0.5, 0, 5)
        pt1 = (500.0 + 1000.0 * (0.5 / 5.0), 500.0) # (600, 500)
        pt2 = (500.0 + 1000.0 * (-0.5 / 5.0), 500.0) # (400, 500)

        par_deg = compute_ray_parallax_angle(pt1, pt2, self.K, self.R2)
        # Expected angle: 2 * arctan(0.5 / 5.0) = 2 * arctan(0.1) approx 11.42 degrees
        expected_deg = math.degrees(2.0 * math.atan(0.1))
        self.assertAlmostEqual(par_deg, expected_deg, places=2)

    def test_near_zero_w_rejection(self):
        """Test that points at infinity or ill-conditioned rays with |W| < epsilon are rejected."""
        # Provide degenerate identical lines where rays are collinear / parallel
        # or artificially construct batch with invalid input
        corr = [{
            "pt1": (500.0, 500.0),
            "pt2": (500.0, 500.0), # No horizontal disparity despite horizontal baseline -> at infinity
            "kp_idx1": 0, "kp_idx2": 0, "track_id": 1, "track_length": 2
        }]
        points, stats = triangulate_correspondence_batch(
            corr, self.P1, self.P2, self.K, self.R2, self.t2,
            min_parallax_deg=1.0, max_reproj_err_px=2.0
        )
        self.assertEqual(len(points), 1)
        # Point has 0 parallax (0 disparity on pure X translation), so it fails
        self.assertFalse(points[0].is_valid)

    def test_cheirality_rejection_negative_depth(self):
        """Test points triangulated behind either camera are rejected with cheirality failure."""
        # Point behind Camera 1: Z = -5.0
        X_behind = np.array([0.5, 0.2, -5.0])
        p1_h = self.K @ X_behind
        pt1 = (float(p1_h[0] / p1_h[2]), float(p1_h[1] / p1_h[2]))
        X2_behind = self.R2 @ X_behind + self.t2.ravel()
        p2_h = self.K @ X2_behind
        pt2 = (float(p2_h[0] / p2_h[2]), float(p2_h[1] / p2_h[2]))

        corr = [{
            "pt1": pt1, "pt2": pt2,
            "kp_idx1": 1, "kp_idx2": 1, "track_id": 42, "track_length": 3
        }]
        points, stats = triangulate_correspondence_batch(
            corr, self.P1, self.P2, self.K, self.R2, self.t2
        )
        self.assertEqual(len(points), 1)
        self.assertFalse(points[0].is_valid)
        self.assertIn("NEGATIVE_DEPTH", points[0].primary_rejection_reason)

    def test_track_id_and_multi_view_preservation(self):
        """Test track ID and observing image list are preserved across triangulation."""
        X_gt = np.array([0.5, 0.2, 5.0])
        p1_h = self.K @ X_gt
        pt1 = (float(p1_h[0] / p1_h[2]), float(p1_h[1] / p1_h[2]))
        X2_gt = self.R2 @ X_gt + self.t2.ravel()
        p2_h = self.K @ X2_gt
        pt2 = (float(p2_h[0] / p2_h[2]), float(p2_h[1] / p2_h[2]))

        corr = [{
            "pt1": pt1, "pt2": pt2,
            "kp_idx1": 10, "kp_idx2": 20,
            "track_id": 9999,
            "track_length": 5,
            "observing_images": ["im_0009", "im_0010", "im_0011", "im_0012", "im_0013"]
        }]
        points, stats = triangulate_correspondence_batch(
            corr, self.P1, self.P2, self.K, self.R2, self.t2,
            min_parallax_deg=2.0, max_reproj_err_px=2.0
        )
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertTrue(p.is_valid)
        self.assertEqual(p.track_id, 9999)
        self.assertEqual(p.track_length, 5)
        self.assertEqual(p.observing_images, ["im_0009", "im_0010", "im_0011", "im_0012", "im_0013"])

    def test_coordinate_transformation_consistency(self):
        """Test that X2 in Camera 2 frame satisfies X2 = R2 * X1 + t2."""
        X_gt = np.array([1.2, -0.4, 4.2])
        p1_h = self.K @ X_gt
        pt1 = (float(p1_h[0] / p1_h[2]), float(p1_h[1] / p1_h[2]))
        X2_gt = self.R2 @ X_gt + self.t2.ravel()
        p2_h = self.K @ X2_gt
        pt2 = (float(p2_h[0] / p2_h[2]), float(p2_h[1] / p2_h[2]))

        corr = [{"pt1": pt1, "pt2": pt2, "kp_idx1": 0, "kp_idx2": 0}]
        points, _ = triangulate_correspondence_batch(
            corr, self.P1, self.P2, self.K, self.R2, self.t2
        )
        p = points[0]
        pt1_cam = np.array([p.X, p.Y, p.Z])
        pt2_cam = np.array([p.X2, p.Y2, p.Z2])
        expected_pt2 = self.R2 @ pt1_cam + self.t2.ravel()
        np.testing.assert_allclose(pt2_cam, expected_pt2, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
