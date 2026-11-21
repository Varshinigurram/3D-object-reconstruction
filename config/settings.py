"""Centralized Project Configuration and Path Settings.

This module defines standardized directory paths and baseline configuration
parameters for the Adaptive Multi-View 3D Object Reconstruction project.
"""

from pathlib import Path
from typing import Dict, Any


# -----------------------------------------------------------------------------
# Base Directories
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_INPUT_DIR = DATA_DIR / "input"
DATA_SELECTED_DIR = DATA_DIR / "selected"
DATA_REFERENCE_DIR = DATA_DIR / "reference"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Source & Package Directories
CONFIG_DIR = PROJECT_ROOT / "config"
MODULES_DIR = PROJECT_ROOT / "modules"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TESTS_DIR = PROJECT_ROOT / "tests"

# Output Directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_POINTCLOUDS_DIR = OUTPUTS_DIR / "pointclouds"
OUTPUTS_MESHES_DIR = OUTPUTS_DIR / "meshes"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"

# -----------------------------------------------------------------------------
# Supported Formats & File Extensions
# -----------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
POINTCLOUD_EXTENSIONS = {".ply", ".pcd", ".xyz"}
MESH_EXTENSIONS = {".ply", ".obj", ".stl"}

# -----------------------------------------------------------------------------
# Default Processing Parameters (Placeholders for Future Pipeline Stages)
# -----------------------------------------------------------------------------
DEFAULT_FEATURE_CONFIG: Dict[str, Any] = {
    "detector": "SIFT",            # e.g., 'SIFT', 'ORB'
    "n_features": 2000,
    "ratio_test_threshold": 0.75,
    "ransac_reproj_threshold": 3.0,
}

DEFAULT_RECONSTRUCTION_CONFIG: Dict[str, Any] = {
    "min_triangulation_angle_deg": 3.0,
    "max_reprojection_error_px": 2.5,
    "depth_truncation": 5.0,
}

DEFAULT_POINTCLOUD_FILTER_CONFIG: Dict[str, Any] = {
    "voxel_size": 0.02,
    "nb_neighbors": 20,
    "std_ratio": 2.0,
}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_all_directories() -> list[Path]:
    """Returns a list of all primary project directories."""
    return [
        DATA_INPUT_DIR,
        DATA_SELECTED_DIR,
        DATA_REFERENCE_DIR,
        DATA_PROCESSED_DIR,
        CONFIG_DIR,
        MODULES_DIR,
        VISUALIZATION_DIR,
        EXPERIMENTS_DIR,
        TESTS_DIR,
        OUTPUTS_POINTCLOUDS_DIR,
        OUTPUTS_MESHES_DIR,
        OUTPUTS_REPORTS_DIR,
    ]


def ensure_directories_exist() -> None:
    """Creates any missing project directories safely."""
    for directory in get_all_directories():
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    ensure_directories_exist()
    print("All project directories verified/created successfully.")
