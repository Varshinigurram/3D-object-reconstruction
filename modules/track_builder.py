"""
modules/track_builder.py
========================
Multi-View Feature Track Construction and Consistency Verification Module.

Part of the "Adaptive Multi-View 3D Object Reconstruction and Geometric Analysis" project.

Core capabilities:
1. Feature observation representation with sub-pixel coordinates and SIFT metadata.
2. Pairwise matching graph management with geometric verification (RANSAC F-matrix).
3. Disjoint-Set / Union-Find track construction with strict physical conflict checks:
   - At most one observation per image in any valid track.
   - Explicit rejection of duplicate-image conflicts.
   - Cycle detection: distinguishing consistent reinforcing cycles from inconsistent cycles.
   - Invalid keypoint index protection.
4. Comprehensive multi-view track statistics (length distributions, per-image coverage).
5. Visual diagnostic generators for multi-view track verification across sequence images.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import math

import cv2
import numpy as np
import pandas as pd

from modules.feature_matching import (
    detect_sift_features,
    match_descriptors,
    estimate_fundamental_matrix,
)


@dataclass
class FeatureObservation:
    """Represents a single 2D feature observation in an image."""
    image_id: str
    keypoint_idx: int
    u: float
    v: float
    size: float = 0.0
    angle: float = 0.0
    response: float = 0.0
    octave: int = 0
    descriptor: Optional[np.ndarray] = None

    @property
    def pt(self) -> Tuple[float, float]:
        """Returns (u, v) pixel coordinate tuple."""
        return (self.u, self.v)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes observation metadata to dictionary."""
        return {
            "image_id": self.image_id,
            "keypoint_idx": self.keypoint_idx,
            "u": round(self.u, 2),
            "v": round(self.v, 2),
            "size": round(self.size, 2),
            "angle": round(self.angle, 2),
            "response": round(self.response, 4),
        }


@dataclass
class PairwiseMatchEdge:
    """Represents a pairwise matching edge between two images in the matching graph."""
    image_id1: str
    image_id2: str
    num_raw_matches: int = 0
    num_good_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    status: str = "PENDING"  # 'VERIFIED', 'REJECTED'
    rejection_reason: str = ""
    inlier_pairs: List[Tuple[int, int]] = field(default_factory=list)  # (kp1_idx, kp2_idx)
    F_matrix: Optional[np.ndarray] = None

    @property
    def pair_key(self) -> Tuple[str, str]:
        """Canonical sorted pair key."""
        return (self.image_id1, self.image_id2)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes edge metadata to dictionary."""
        return {
            "image_1": self.image_id1,
            "image_2": self.image_id2,
            "raw_matches": self.num_raw_matches,
            "good_matches": self.num_good_matches,
            "inliers": self.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "status": self.status,
            "rejection_reason": self.rejection_reason,
        }


class FeatureTrack:
    """Represents a candidate physical surface point observed across multiple views."""

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        # Strict mapping: image_id -> FeatureObservation (at most ONE observation per image)
        self.observations: Dict[str, FeatureObservation] = {}
        # History of verified pairwise edges that merged into this track: (img1, kp1, img2, kp2)
        self.contributing_edges: List[Tuple[str, int, str, int]] = []

    @property
    def length(self) -> int:
        """Number of distinct image views observing this track."""
        return len(self.observations)

    @property
    def image_ids(self) -> List[str]:
        """Sorted list of image IDs observing this track."""
        return sorted(self.observations.keys())

    def contains_image(self, image_id: str) -> bool:
        """Checks if track already contains an observation for the specified image."""
        return image_id in self.observations

    def add_observation(self, obs: FeatureObservation) -> bool:
        """Adds observation if image_id is not already present. Returns False on conflict."""
        if obs.image_id in self.observations:
            return False
        self.observations[obs.image_id] = obs
        return True

    def get_observation(self, image_id: str) -> Optional[FeatureObservation]:
        """Returns observation for given image_id, or None."""
        return self.observations.get(image_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes track summary."""
        return {
            "track_id": self.track_id,
            "track_length": self.length,
            "images": ",".join(self.image_ids),
            "first_image": self.image_ids[0] if self.observations else "",
            "last_image": self.image_ids[-1] if self.observations else "",
            "observations_count": len(self.observations),
        }


class TrackBuilder:
    """Graph-based feature track constructor with strict multi-view conflict resolution."""

    def __init__(
        self,
        min_inliers_per_edge: int = 30,
        min_inlier_ratio: float = 0.35,
        ransac_threshold_px: float = 2.0,
        ransac_confidence: float = 0.999,
    ):
        self.min_inliers = min_inliers_per_edge
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_threshold = ransac_threshold_px
        self.ransac_confidence = ransac_confidence

        # Image feature cache: image_id -> {"keypoints": [...], "descriptors": [...], "shape": (h, w)}
        self.image_features: Dict[str, Dict[str, Any]] = {}

        # Pairwise matching graph: (image_id1, image_id2) -> PairwiseMatchEdge
        self.pairwise_edges: Dict[Tuple[str, str], PairwiseMatchEdge] = {}

        # Node to Track mapping: (image_id, keypoint_idx) -> track_id
        self.node_to_track_id: Dict[Tuple[str, int], int] = {}

        # Active tracks: track_id -> FeatureTrack
        self.tracks: Dict[int, FeatureTrack] = {}
        self._next_track_id: int = 1

        # Conflict & verification statistics
        self.stats = {
            "edges_evaluated": 0,
            "edges_verified": 0,
            "edges_rejected": 0,
            "unions_attempted": 0,
            "unions_successful": 0,
            "unions_rejected_duplicate_image": 0,
            "unions_rejected_invalid_keypoint": 0,
            "unions_rejected_self_image": 0,
            "consistent_cycles_detected": 0,
            "inconsistent_cycles_detected": 0,
        }

    def register_image_features(
        self,
        image_id: str,
        keypoints: List[cv2.KeyPoint],
        descriptors: Optional[np.ndarray],
        image_shape: Tuple[int, int] = (2272, 1704),
    ) -> None:
        """Caches keypoints, descriptors, and dimensions for an image view."""
        self.image_features[image_id] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "shape": image_shape,  # (height, width)
        }

    def match_pair(
        self,
        image_id1: str,
        image_id2: str,
        ratio_threshold: float = 0.75,
    ) -> PairwiseMatchEdge:
        """Performs SIFT matching and RANSAC epipolar verification between two cached views."""
        edge = PairwiseMatchEdge(image_id1=image_id1, image_id2=image_id2)
        self.stats["edges_evaluated"] += 1

        feat1 = self.image_features.get(image_id1)
        feat2 = self.image_features.get(image_id2)

        if feat1 is None or feat2 is None:
            edge.status = "REJECTED"
            edge.rejection_reason = "MISSING_IMAGE_FEATURES"
            self.stats["edges_rejected"] += 1
            self.pairwise_edges[(image_id1, image_id2)] = edge
            return edge

        kp1, desc1 = feat1["keypoints"], feat1["descriptors"]
        kp2, desc2 = feat2["keypoints"], feat2["descriptors"]

        if desc1 is None or desc2 is None or len(desc1) < 8 or len(desc2) < 8:
            edge.status = "REJECTED"
            edge.rejection_reason = "INSUFFICIENT_KEYPOINTS"
            self.stats["edges_rejected"] += 1
            self.pairwise_edges[(image_id1, image_id2)] = edge
            return edge

        # 1. Lowe's ratio test matching
        raw_cnt, good_matches = match_descriptors(desc1, desc2, ratio_threshold=ratio_threshold)
        edge.num_raw_matches = raw_cnt
        edge.num_good_matches = len(good_matches)

        if len(good_matches) < 8:
            edge.status = "REJECTED"
            edge.rejection_reason = "INSUFFICIENT_RATIO_MATCHES"
            self.stats["edges_rejected"] += 1
            self.pairwise_edges[(image_id1, image_id2)] = edge
            return edge

        # 2. RANSAC Fundamental Matrix geometric verification
        inl_cnt, inl_ratio, F, inl_mask, status = estimate_fundamental_matrix(
            kp1, kp2, good_matches,
            reproj_threshold=self.ransac_threshold,
            confidence=self.ransac_confidence,
        )

        edge.num_inliers = inl_cnt
        edge.inlier_ratio = inl_ratio
        edge.F_matrix = F

        if status != "COMPUTED" or F is None:
            edge.status = "REJECTED"
            edge.rejection_reason = f"GEOMETRIC_FAILURE: {status}"
            self.stats["edges_rejected"] += 1
            self.pairwise_edges[(image_id1, image_id2)] = edge
            return edge

        # Threshold checks
        if inl_cnt < self.min_inliers:
            edge.status = "REJECTED"
            edge.rejection_reason = f"LOW_INLIER_COUNT ({inl_cnt} < {self.min_inliers})"
            self.stats["edges_rejected"] += 1
            self.pairwise_edges[(image_id1, image_id2)] = edge
            return edge

        if inl_ratio < self.min_inlier_ratio:
            edge.status = "REJECTED"
            edge.rejection_reason = f"LOW_INLIER_RATIO ({inl_ratio:.3f} < {self.min_inlier_ratio:.3f})"
            self.stats["edges_rejected"] += 1
            self.pairwise_edges[(image_id1, image_id2)] = edge
            return edge

        # Edge verified! Extract inlier keypoint indices
        inlier_indices = []
        for idx, is_inl in enumerate(inl_mask):
            if is_inl:
                m = good_matches[idx]
                inlier_indices.append((m.queryIdx, m.trainIdx))

        edge.inlier_pairs = inlier_indices
        edge.status = "VERIFIED"
        edge.rejection_reason = "PASSED_GEOMETRIC_VERIFICATION"
        self.stats["edges_verified"] += 1
        self.pairwise_edges[(image_id1, image_id2)] = edge
        return edge

    def _create_observation(self, image_id: str, kp_idx: int) -> Optional[FeatureObservation]:
        """Creates a FeatureObservation from cached keypoint metadata."""
        feat = self.image_features.get(image_id)
        if feat is None:
            return None
        keypoints = feat["keypoints"]
        descriptors = feat["descriptors"]
        if kp_idx < 0 or kp_idx >= len(keypoints):
            return None

        kp = keypoints[kp_idx]
        desc = descriptors[kp_idx] if descriptors is not None else None
        return FeatureObservation(
            image_id=image_id,
            keypoint_idx=kp_idx,
            u=float(kp.pt[0]),
            v=float(kp.pt[1]),
            size=float(kp.size),
            angle=float(kp.angle),
            response=float(kp.response),
            octave=int(kp.octave),
            descriptor=desc,
        )

    def add_verified_correspondence(
        self,
        image_id1: str,
        kp_idx1: int,
        image_id2: str,
        kp_idx2: int,
    ) -> bool:
        """
        Attempts to merge a pairwise correspondence into a multi-view track.
        Enforces strict single-observation-per-image constraint and detects conflicts.

        Returns:
            True if correspondence was successfully integrated into a track or reinforced a cycle.
            False if union was rejected due to conflicts or invalid inputs.
        """
        self.stats["unions_attempted"] += 1

        # Check self-image matching
        if image_id1 == image_id2:
            self.stats["unions_rejected_self_image"] += 1
            return False

        # Validate keypoint indices
        feat1 = self.image_features.get(image_id1)
        feat2 = self.image_features.get(image_id2)
        if feat1 is None or feat2 is None:
            self.stats["unions_rejected_invalid_keypoint"] += 1
            return False

        if kp_idx1 < 0 or kp_idx1 >= len(feat1["keypoints"]) or \
           kp_idx2 < 0 or kp_idx2 >= len(feat2["keypoints"]):
            self.stats["unions_rejected_invalid_keypoint"] += 1
            return False

        node1 = (image_id1, kp_idx1)
        node2 = (image_id2, kp_idx2)

        track_id1 = self.node_to_track_id.get(node1)
        track_id2 = self.node_to_track_id.get(node2)

        # CASE 1: Neither node is currently in any track
        if track_id1 is None and track_id2 is None:
            obs1 = self._create_observation(image_id1, kp_idx1)
            obs2 = self._create_observation(image_id2, kp_idx2)
            if obs1 is None or obs2 is None:
                self.stats["unions_rejected_invalid_keypoint"] += 1
                return False

            new_track = FeatureTrack(track_id=self._next_track_id)
            self._next_track_id += 1
            new_track.add_observation(obs1)
            new_track.add_observation(obs2)
            new_track.contributing_edges.append((image_id1, kp_idx1, image_id2, kp_idx2))

            self.tracks[new_track.track_id] = new_track
            self.node_to_track_id[node1] = new_track.track_id
            self.node_to_track_id[node2] = new_track.track_id
            self.stats["unions_successful"] += 1
            return True

        # CASE 2: node1 is in a track, node2 is unassigned
        elif track_id1 is not None and track_id2 is None:
            track1 = self.tracks[track_id1]
            # Check if track1 already contains an observation for image_id2
            if track1.contains_image(image_id2):
                # Duplicate-image conflict! Reject union!
                self.stats["unions_rejected_duplicate_image"] += 1
                self.stats["inconsistent_cycles_detected"] += 1
                return False

            obs2 = self._create_observation(image_id2, kp_idx2)
            if obs2 is None:
                self.stats["unions_rejected_invalid_keypoint"] += 1
                return False

            track1.add_observation(obs2)
            track1.contributing_edges.append((image_id1, kp_idx1, image_id2, kp_idx2))
            self.node_to_track_id[node2] = track_id1
            self.stats["unions_successful"] += 1
            return True

        # CASE 3: node2 is in a track, node1 is unassigned
        elif track_id1 is None and track_id2 is not None:
            track2 = self.tracks[track_id2]
            # Check if track2 already contains an observation for image_id1
            if track2.contains_image(image_id1):
                # Duplicate-image conflict! Reject union!
                self.stats["unions_rejected_duplicate_image"] += 1
                self.stats["inconsistent_cycles_detected"] += 1
                return False

            obs1 = self._create_observation(image_id1, kp_idx1)
            if obs1 is None:
                self.stats["unions_rejected_invalid_keypoint"] += 1
                return False

            track2.add_observation(obs1)
            track2.contributing_edges.append((image_id1, kp_idx1, image_id2, kp_idx2))
            self.node_to_track_id[node1] = track_id2
            self.stats["unions_successful"] += 1
            return True

        # CASE 4: Both nodes are already in tracks
        else:
            # Subcase 4A: Both nodes belong to the exact SAME track
            if track_id1 == track_id2:
                # Consistent reinforcing cycle!
                track1 = self.tracks[track_id1]
                track1.contributing_edges.append((image_id1, kp_idx1, image_id2, kp_idx2))
                self.stats["consistent_cycles_detected"] += 1
                return True

            # Subcase 4B: Nodes belong to DIFFERENT tracks -> Attempt merge
            track1 = self.tracks[track_id1]
            track2 = self.tracks[track_id2]

            # Check for image overlap between the two candidate tracks
            images1 = set(track1.observations.keys())
            images2 = set(track2.observations.keys())
            common_images = images1.intersection(images2)

            if len(common_images) > 0:
                # Merging would create multiple observations for common_images!
                # Reject union to preserve physical 3D-to-2D single projection property.
                self.stats["unions_rejected_duplicate_image"] += 1
                self.stats["inconsistent_cycles_detected"] += 1
                return False

            # No overlap -> Safe to merge! Merge smaller track into larger track
            if track1.length < track2.length:
                target_track = track2
                source_track = track1
                target_id = track_id2
                source_id = track_id1
            else:
                target_track = track1
                source_track = track2
                target_id = track_id1
                source_id = track_id2

            for img_id, obs in source_track.observations.items():
                target_track.add_observation(obs)
                self.node_to_track_id[(img_id, obs.keypoint_idx)] = target_id

            target_track.contributing_edges.extend(source_track.contributing_edges)
            target_track.contributing_edges.append((image_id1, kp_idx1, image_id2, kp_idx2))

            # Delete merged source track
            del self.tracks[source_id]
            self.stats["unions_successful"] += 1
            return True

    def build_tracks_from_edges(
        self,
        edge_keys: Optional[List[Tuple[str, str]]] = None,
        sort_by_inliers: bool = True,
    ) -> None:
        """
        Builds feature tracks across all verified pairwise edges in the matching graph.
        
        Args:
            edge_keys: Optional ordered list of pairwise keys to process. If None,
                       all verified edges in self.pairwise_edges are processed.
            sort_by_inliers: If True, edges are processed in descending order of inliers
                             so the highest confidence connections form the initial track backbone.
        """
        if edge_keys is None:
            candidate_edges = [
                e for e in self.pairwise_edges.values() if e.status == "VERIFIED"
            ]
        else:
            candidate_edges = [
                self.pairwise_edges[k] for k in edge_keys
                if k in self.pairwise_edges and self.pairwise_edges[k].status == "VERIFIED"
            ]

        if sort_by_inliers:
            candidate_edges.sort(key=lambda e: e.num_inliers, reverse=True)

        for edge in candidate_edges:
            for kp1_idx, kp2_idx in edge.inlier_pairs:
                self.add_verified_correspondence(
                    edge.image_id1, kp1_idx,
                    edge.image_id2, kp2_idx,
                )

    def get_track_statistics(self) -> Dict[str, Any]:
        """Calculates comprehensive track quality metrics and length distributions."""
        tracks_list = list(self.tracks.values())
        total_tracks = len(tracks_list)
        if total_tracks == 0:
            return {
                "total_tracks": 0,
                "length_distribution": {},
                "tracks_len_ge_2": 0,
                "tracks_len_ge_3": 0,
                "tracks_len_ge_4": 0,
                "tracks_len_ge_5": 0,
                "max_length": 0,
                "mean_length": 0.0,
                "median_length": 0.0,
                "per_image_observations": {},
                "conflict_stats": self.stats,
            }

        lengths = [t.length for t in tracks_list]
        max_len = max(lengths)
        mean_len = float(np.mean(lengths))
        med_len = float(np.median(lengths))

        # Length distribution histogram
        length_dist: Dict[int, int] = {}
        for l in lengths:
            length_dist[l] = length_dist.get(l, 0) + 1

        ge_2 = sum(1 for l in lengths if l >= 2)
        ge_3 = sum(1 for l in lengths if l >= 3)
        ge_4 = sum(1 for l in lengths if l >= 4)
        ge_5 = sum(1 for l in lengths if l >= 5)

        # Per-image observation counts
        per_img_obs: Dict[str, int] = {}
        for t in tracks_list:
            for img_id in t.image_ids:
                per_img_obs[img_id] = per_img_obs.get(img_id, 0) + 1

        return {
            "total_tracks": total_tracks,
            "length_distribution": dict(sorted(length_dist.items())),
            "tracks_len_ge_2": ge_2,
            "tracks_len_ge_3": ge_3,
            "tracks_len_ge_4": ge_4,
            "tracks_len_ge_5": ge_5,
            "max_length": max_len,
            "mean_length": round(mean_len, 2),
            "median_length": round(med_len, 2),
            "per_image_observations": dict(sorted(per_img_obs.items())),
            "conflict_stats": self.stats,
        }

    def export_tracks_dataframe(self) -> pd.DataFrame:
        """Exports track summary records as a pandas DataFrame."""
        rows = []
        for t in sorted(self.tracks.values(), key=lambda x: x.length, reverse=True):
            rows.append({
                "track_id": t.track_id,
                "track_length": t.length,
                "images_count": len(t.image_ids),
                "images_list": ",".join(t.image_ids),
                "first_image": t.image_ids[0],
                "last_image": t.image_ids[-1],
                "span": f"{t.image_ids[0]} -> {t.image_ids[-1]}",
            })
        return pd.DataFrame(rows)


def render_track_multiview_vis(
    track: FeatureTrack,
    image_dir: Path,
    patch_size: int = 120,
    max_views: int = 8,
) -> Optional[np.ndarray]:
    """
    Renders visual cropped patches with crosshairs for a track across its observed views.
    Enables visual inspection of feature track consistency across camera viewpoints.
    """
    if track.length < 2:
        return None

    obs_list = list(track.observations.values())
    # Subsample if length exceeds max_views
    if len(obs_list) > max_views:
        step = max(1, len(obs_list) // max_views)
        obs_to_show = obs_list[::step][:max_views]
    else:
        obs_to_show = obs_list

    num_views = len(obs_to_show)
    card_w = 200
    card_h = 240
    padding = 10
    total_w = num_views * (card_w + padding) + padding
    total_h = card_h + 80

    canvas = np.full((total_h, total_w, 3), 32, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (total_w, 50), (20, 20, 20), -1)
    title = f"TRACK #{track.track_id:05d} | LENGTH = {track.length} VIEWS | SPAN: {track.image_ids[0]} -> {track.image_ids[-1]}"
    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    half_patch = patch_size // 2

    for v_idx, obs in enumerate(obs_to_show):
        img_path = image_dir / f"{obs.image_id}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        cx, cy = int(round(obs.u)), int(round(obs.v))

        # Crop patch safely
        x1 = max(0, cx - half_patch)
        y1 = max(0, cy - half_patch)
        x2 = min(w, cx + half_patch)
        y2 = min(h, cy + half_patch)

        patch = img[y1:y2, x1:x2].copy()
        patch_resized = cv2.resize(patch, (card_w - 20, card_w - 20))

        # Draw crosshair at center of resized patch
        pc_x = (card_w - 20) // 2
        pc_y = (card_w - 20) // 2
        cv2.drawMarker(patch_resized, (pc_x, pc_y), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.circle(patch_resized, (pc_x, pc_y), 8, (0, 255, 255), 1)

        # Place patch on canvas
        start_x = padding + v_idx * (card_w + padding)
        start_y = 65
        cv2.rectangle(canvas, (start_x, start_y), (start_x + card_w, start_y + card_h), (48, 48, 48), -1)
        cv2.rectangle(canvas, (start_x, start_y), (start_x + card_w, start_y + card_h), (80, 80, 80), 1)

        canvas[start_y + 10 : start_y + 10 + (card_w - 20), start_x + 10 : start_x + 10 + (card_w - 20)] = patch_resized

        # Caption
        cv2.putText(canvas, obs.image_id, (start_x + 14, start_y + card_h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(canvas, f"({obs.u:.1f}, {obs.v:.1f})", (start_x + 14, start_y + card_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return canvas


def render_track_length_histogram_vis(
    stats: Dict[str, Any],
    out_path: Path,
) -> np.ndarray:
    """Renders a visualization of track length distribution and sequence coverage."""
    canvas_w = 900
    canvas_h = 500
    canvas = np.full((canvas_h, canvas_w, 3), 28, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (canvas_w, 55), (18, 18, 18), -1)
    cv2.putText(
        canvas, "MULTI-VIEW FEATURE TRACK LENGTH DISTRIBUTION & METRICS",
        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )

    length_dist = stats["length_distribution"]
    if not length_dist:
        return canvas

    # Panel 1: Bar chart for track lengths 2 to 12+
    px, py, pw, ph = 40, 85, 480, 370
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (40, 40, 40), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (70, 70, 70), 1)
    cv2.putText(canvas, "Track Count by Track Length", (px + 15, py + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Group lengths >= 10 into 10+
    display_counts: Dict[str, int] = {}
    for l in range(2, 10):
        display_counts[str(l)] = length_dist.get(l, 0)
    display_counts["10+"] = sum(v for k, v in length_dist.items() if k >= 10)

    max_count = max(display_counts.values()) if display_counts.values() else 1
    num_bars = len(display_counts)
    bar_width = int((pw - 60) / num_bars) - 8

    for idx, (label, count) in enumerate(display_counts.items()):
        bx = px + 40 + idx * (bar_width + 8)
        bar_h = int((count / max_count) * (ph - 80))
        by = py + ph - 40 - bar_h

        # Gradient bar
        cv2.rectangle(canvas, (bx, by), (bx + bar_width, py + ph - 40), (0, 180, 240), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_width, py + ph - 40), (0, 220, 255), 1)

        # Label and count text
        cv2.putText(canvas, str(count), (bx - 4, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(canvas, label, (bx + bar_width // 4, py + ph - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.putText(canvas, "Track Length (Views)", (px + pw // 3, py + ph - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

    # Panel 2: Summary Stats and Quality Table
    tx, ty, tw, th = 550, 85, 310, 370
    cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (40, 40, 40), -1)
    cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (70, 70, 70), 1)
    cv2.putText(canvas, "Track Quality Summary", (tx + 15, ty + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

    stat_lines = [
        ("Total Tracks:", f"{stats['total_tracks']:,}"),
        ("Tracks (Length >= 2):", f"{stats['tracks_len_ge_2']:,} (100.0%)"),
        ("Tracks (Length >= 3):", f"{stats['tracks_len_ge_3']:,} ({stats['tracks_len_ge_3']/max(1, stats['total_tracks'])*100:.1f}%)"),
        ("Tracks (Length >= 4):", f"{stats['tracks_len_ge_4']:,} ({stats['tracks_len_ge_4']/max(1, stats['total_tracks'])*100:.1f}%)"),
        ("Tracks (Length >= 5):", f"{stats['tracks_len_ge_5']:,} ({stats['tracks_len_ge_5']/max(1, stats['total_tracks'])*100:.1f}%)"),
        ("Max Track Length:", f"{stats['max_length']} views"),
        ("Mean Track Length:", f"{stats['mean_length']:.2f} views"),
        ("Median Track Length:", f"{stats['median_length']:.2f} views"),
        ("Successful Unions:", f"{stats['conflict_stats']['unions_successful']:,}"),
        ("Duplicate Conflicts:", f"{stats['conflict_stats']['unions_rejected_duplicate_image']:,} (REJECTED)"),
        ("Consistent Cycles:", f"{stats['conflict_stats']['consistent_cycles_detected']:,}"),
        ("Inconsistent Cycles:", f"{stats['conflict_stats']['inconsistent_cycles_detected']:,}"),
    ]

    for s_idx, (lbl, val) in enumerate(stat_lines):
        line_y = ty + 56 + s_idx * 25
        cv2.putText(canvas, lbl, (tx + 12, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        # Highlight conflicts in orange/red
        color = (0, 160, 255) if "Duplicate" in lbl or "Inconsistent" in lbl else (255, 255, 255)
        cv2.putText(canvas, val, (tx + 175, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return canvas


def render_sequence_coverage_vis(
    per_image_obs: Dict[str, int],
    out_path: Path,
) -> np.ndarray:
    """Renders a sequence track observation coverage plot across all 72 images."""
    canvas_w = 1100
    canvas_h = 380
    canvas = np.full((canvas_h, canvas_w, 3), 26, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas_w, 50), (18, 18, 18), -1)
    cv2.putText(
        canvas, "PER-IMAGE TRACK OBSERVATION COVERAGE (ALL 72 NORMALIZED VIEWS)",
        (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )

    if not per_image_obs:
        return canvas

    img_ids = list(per_image_obs.keys())
    counts = list(per_image_obs.values())
    max_c = max(counts) if counts else 1
    min_c = min(counts) if counts else 0

    px, py, pw, ph = 40, 75, 1020, 260
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (36, 36, 36), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (65, 65, 65), 1)

    bar_step = pw / len(img_ids)
    for idx, (img_id, c) in enumerate(zip(img_ids, counts)):
        bx = int(px + idx * bar_step)
        bar_w = max(1, int(bar_step - 2))
        bar_h = int((c / max_c) * (ph - 50))
        by = py + ph - 30 - bar_h

        # Color: cyan normally, highlight min/low in orange
        color = (0, 140, 255) if c < min_c * 1.5 else (0, 220, 180)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, py + ph - 30), color, -1)

        # Label every 5th image
        if (idx + 1) % 5 == 0 or idx == 0 or (idx + 1) == len(img_ids):
            label = f"{idx+1}"
            cv2.putText(canvas, label, (bx - 4, py + ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

    # Stats overlay
    mean_obs = np.mean(counts)
    med_obs = np.median(counts)
    cv2.putText(
        canvas, f"Total Observed: {sum(counts):,} | Min: {min_c} ({img_ids[np.argmin(counts)]}) | Max: {max_c} ({img_ids[np.argmax(counts)]}) | Mean: {mean_obs:.1f} | Med: {med_obs:.1f}",
        (px + 15, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return canvas
