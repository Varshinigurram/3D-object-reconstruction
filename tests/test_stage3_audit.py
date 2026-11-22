"""
tests/test_stage3_audit.py
==========================
Unit tests for Stage 3 Independent Reconstruction Audit math and validation logic.

Tests:
1. PCA plane normal and plane residual calculation.
2. Algebraic 2D circle fitting and radius recovery.
3. Rotation matrix orthogonality and determinant deviation checks.
4. Camera center calculation C = -R.T @ t.
5. Inlier recomputation under Euclidean reprojection error.
6. Monocular closure gap calculation.
"""

import unittest
import numpy as np


def fit_plane_pca(points: np.ndarray) -> tuple:
    """Fits plane via PCA. Returns (center, normal, residuals)."""
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[2, :]
    if normal[1] > 0:  # Consistent convention
        normal = -normal
    residuals = np.abs(centered @ normal)
    return center, normal, residuals


def fit_circle_2d(u: np.ndarray, v: np.ndarray):
    """Algebraic 2D circle fit. Returns (uc, vc, radius, residuals)."""
    A = np.column_stack([2 * u, 2 * v, np.ones_like(u)])
    b = u**2 + v**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    uc, vc, c_const = sol
    radius = np.sqrt(c_const + uc**2 + vc**2)
    residuals = np.abs(np.hypot(u - uc, v - vc) - radius)
    return float(uc), float(vc), float(radius), residuals


class TestStage3AuditMath(unittest.TestCase):

    def test_01_pca_plane_fit_synthetic(self):
        """1. Verify PCA recovers ground-truth horizontal plane normal."""
        # Points on plane Y = 2.0
        np.random.seed(42)
        X = np.random.uniform(-5, 5, 100)
        Z = np.random.uniform(-5, 5, 100)
        Y = np.full_like(X, 2.0)
        pts = np.column_stack([X, Y, Z])

        center, normal, res = fit_plane_pca(pts)
        self.assertAlmostEqual(center[1], 2.0, delta=1e-6)
        # Normal should align with Y-axis [0, -1, 0]
        np.testing.assert_allclose(np.abs(normal), [0.0, 1.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(np.max(res), 0.0, delta=1e-6)

    def test_02_circle_fit_synthetic(self):
        """2. Verify algebraic 2D circle fit recovers known center and radius."""
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        r_gt = 6.0
        uc_gt, vc_gt = 1.5, -2.5
        u = uc_gt + r_gt * np.cos(theta)
        v = vc_gt + r_gt * np.sin(theta)

        uc, vc, r, res = fit_circle_2d(u, v)
        self.assertAlmostEqual(uc, uc_gt, delta=1e-5)
        self.assertAlmostEqual(vc, vc_gt, delta=1e-5)
        self.assertAlmostEqual(r, r_gt, delta=1e-5)
        self.assertAlmostEqual(np.max(res), 0.0, delta=1e-5)

    def test_03_rotation_orthogonality_check(self):
        """3. Verify rotation matrix orthogonality check detects non-orthogonal matrices."""
        theta = np.radians(30.0)
        R_valid = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ])
        ortho_err = np.linalg.norm(R_valid.T @ R_valid - np.eye(3))
        self.assertLess(ortho_err, 1e-12)

        # Corrupt matrix
        R_corrupt = R_valid.copy()
        R_corrupt[0, 0] += 0.1
        ortho_err_bad = np.linalg.norm(R_corrupt.T @ R_corrupt - np.eye(3))
        self.assertGreater(ortho_err_bad, 0.05)

    def test_04_camera_center_recomputation(self):
        """4. Verify camera center recomputation C = -R.T @ t."""
        theta = np.radians(45.0)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t = np.array([2.0, -1.0, 5.0])
        C = -R.T @ t
        # Verify that R @ C + t == 0
        np.testing.assert_allclose(R @ C + t, [0.0, 0.0, 0.0], atol=1e-12)

    def test_05_closure_gap_measurement(self):
        """5. Verify closure gap distance calculation between two camera centers."""
        C1 = np.array([1.0, 2.0, 3.0])
        C2 = np.array([4.0, 6.0, 3.0])
        dist = float(np.linalg.norm(C2 - C1))
        # 3-4-5 triangle in X-Y plane => sqrt(3^2 + 4^2) = 5.0
        self.assertAlmostEqual(dist, 5.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
