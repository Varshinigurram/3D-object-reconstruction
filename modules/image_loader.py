"""Module for discovering, safely loading, and hashing image files.

Part of the Adaptive Multi-View 3D Object Reconstruction pipeline.
Designed for modular, robust dataset inspection and quality assessment.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np


# Standard supported image extensions
DEFAULT_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def discover_image_files(
    directory: Path | str,
    supported_extensions: set[str] = DEFAULT_SUPPORTED_EXTENSIONS,
) -> List[Path]:
    """Recursively discover all image files with supported extensions in a directory.

    Args:
        directory: Root path to search.
        supported_extensions: Set of allowed lowercase file extensions.

    Returns:
        List of Path objects for all discovered image files, sorted alphabetically.
    """
    root_path = Path(directory)
    if not root_path.exists() or not root_path.is_dir():
        return []

    image_files: List[Path] = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            image_files.append(file_path)

    return sorted(image_files)


def compute_file_hash(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 hash of a file's raw binary content.

    Used to detect exact duplicates across the dataset without altering images.

    Args:
        file_path: Path to the target file.
        chunk_size: Block size in bytes for reading the file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_image_safely(
    file_path: Path | str,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Safely loads an image file and extracts basic geometric/file metadata.

    Handles unreadable or corrupted files gracefully without raising unhandled exceptions.

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple of (image_bgr_array or None, metadata_dict):
            metadata_dict keys:
                - readable (bool): True if loaded successfully.
                - width (int or None): Image width in pixels.
                - height (int or None): Image height in pixels.
                - channels (int or None): Number of color channels.
                - file_size_kb (float): File size in kilobytes.
                - extension (str): File extension in lowercase.
                - error_message (str or None): Error message if loading failed.
    """
    path = Path(file_path)
    metadata: Dict[str, Any] = {
        "readable": False,
        "width": None,
        "height": None,
        "channels": None,
        "file_size_kb": 0.0,
        "extension": path.suffix.lower(),
        "error_message": None,
    }

    try:
        # File existence & size
        if not path.is_file():
            metadata["error_message"] = "File does not exist or is not a regular file."
            return None, metadata

        file_size_bytes = path.stat().st_size
        metadata["file_size_kb"] = round(file_size_bytes / 1024.0, 2)

        if file_size_bytes == 0:
            metadata["error_message"] = "Empty file (0 bytes)."
            return None, metadata

        # Use cv2.imdecode to avoid path encoding issues on Windows
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            metadata["error_message"] = "OpenCV failed to decode image (corrupt or unsupported format)."
            return None, metadata

        height, width = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1

        metadata["readable"] = True
        metadata["width"] = int(width)
        metadata["height"] = int(height)
        metadata["channels"] = int(channels)
        return img, metadata

    except Exception as exc:
        metadata["error_message"] = f"Unexpected read error: {type(exc).__name__}: {str(exc)}"
        return None, metadata
