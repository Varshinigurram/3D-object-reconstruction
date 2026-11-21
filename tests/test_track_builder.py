"""
tests/test_track_builder.py
===========================
Focused Unit Tests for Multi-View Feature Track Construction.

Tests verify:
1. Track merging across multiple images.
2. Strict duplicate-image conflict rejection.
3. Inconsistent cycle detection and rejection.
4. Consistent cycle reinforcement.
5. Rejected pairwise edge handling.
6. Invalid keypoint index protection.
7. Multi-image track span and length integrity.
"""

import unittest
import numpy as np
import cv2

from modules.track_builder import (
    FeatureObservation,
    PairwiseMatchEdge,
    FeatureTrack,
    TrackBuilder,
)


class TestTrackBuilder(unittest.TestCase):
    """Test suite for TrackBuilder disjoint-set association and conflict resolution."""

    def setUp(self):
        """Set up synthetic keypoints and TrackBuilder instance for testing."""
        self.builder = TrackBuilder(min_inliers_per_edge=5, min_inlier_ratio=0.3)

        # Register synthetic image features for 4 images (A, B, C, D)
        # Each image has 5 keypoints
        for img_id in ["im_A", "im_B", "im_C", "im_D"]:
            kps = [cv2.KeyPoint(x=float(i * 10), y=float(i * 20), size=1.0) for i in range(5)]
            descs = np.random.rand(5, 128).astype(np.float32)
            self.builder.register_image_features(img_id, kps, descs, (2272, 1704))

    def test_basic_track_merging(self):
        """Test standard pairwise correspondence creates a length-2 track."""
        success = self.builder.add_verified_correspondence("im_A", 0, "im_B", 0)
        self.assertTrue(success)

        # Exactly 1 track created
        self.assertEqual(len(self.builder.tracks), 1)
        track = list(self.builder.tracks.values())[0]
        self.assertEqual(track.length, 2)
        self.assertIn("im_A", track.observations)
        self.assertIn("im_B", track.observations)
        self.assertEqual(track.observations["im_A"].keypoint_idx, 0)
        self.assertEqual(track.observations["im_B"].keypoint_idx, 0)

    def test_multi_image_track_spanning(self):
        """Test chaining correspondences A0-B0 and B0-C0 creates a length-3 track."""
        self.builder.add_verified_correspondence("im_A", 0, "im_B", 0)
        self.builder.add_verified_correspondence("im_B", 0, "im_C", 0)

        # Still exactly 1 track, now spanning 3 images
        self.assertEqual(len(self.builder.tracks), 1)
        track = list(self.builder.tracks.values())[0]
        self.assertEqual(track.length, 3)
        self.assertEqual(track.image_ids, ["im_A", "im_B", "im_C"])

        # Chain across 4th image
        self.builder.add_verified_correspondence("im_C", 0, "im_D", 0)
        self.assertEqual(track.length, 4)
        self.assertEqual(track.image_ids, ["im_A", "im_B", "im_C", "im_D"])

    def test_duplicate_image_conflict_rejection_single_node(self):
        """
        Test that linking an existing track with a different keypoint from an already-observed image
        is strictly rejected as a duplicate-image conflict.
        """
        # Create track containing (im_A, 0) and (im_B, 0)
        self.builder.add_verified_correspondence("im_A", 0, "im_B", 0)

        # Attempt to link (im_A, 0) with (im_B, 1) -> Conflict! im_B already observed at kp 0
        success = self.builder.add_verified_correspondence("im_A", 0, "im_B", 1)
        self.assertFalse(success)

        # Track should still only have 2 observations
        track = list(self.builder.tracks.values())[0]
        self.assertEqual(track.length, 2)
        self.assertEqual(track.observations["im_B"].keypoint_idx, 0)
        self.assertEqual(self.builder.stats["unions_rejected_duplicate_image"], 1)

    def test_duplicate_image_conflict_rejection_between_tracks(self):
        """
        Test that merging two separate tracks that observe a common image is rejected.
        Track 1: (im_A, 0) - (im_B, 0)
        Track 2: (im_B, 1) - (im_C, 1)
        Attempting to link (im_A, 0) with (im_C, 1) would merge Track 1 and Track 2,
        which would create two observations in im_B (kp 0 and kp 1). Must be rejected!
        """
        self.builder.add_verified_correspondence("im_A", 0, "im_B", 0)
        self.builder.add_verified_correspondence("im_B", 1, "im_C", 1)

        self.assertEqual(len(self.builder.tracks), 2)

        # Attempt to merge Track 1 and Track 2 via A0 - C1
        success = self.builder.add_verified_correspondence("im_A", 0, "im_C", 1)
        self.assertFalse(success)

        # Both tracks remain separate; conflict recorded
        self.assertEqual(len(self.builder.tracks), 2)
        self.assertEqual(self.builder.stats["unions_rejected_duplicate_image"], 1)
        self.assertEqual(self.builder.stats["inconsistent_cycles_detected"], 1)

    def test_consistent_cycle_reinforcement(self):
        """
        Test that closing a triangle A0-B0, B0-C0, and C0-A0 is recognized as a
        consistent reinforcing cycle (not rejected, does not duplicate).
        """
        self.builder.add_verified_correspondence("im_A", 0, "im_B", 0)
        self.builder.add_verified_correspondence("im_B", 0, "im_C", 0)

        initial_unions = self.builder.stats["unions_successful"]

        # Close the loop with C0 - A0
        success = self.builder.add_verified_correspondence("im_C", 0, "im_A", 0)
        self.assertTrue(success)

        # Track count remains 1, length remains 3
        self.assertEqual(len(self.builder.tracks), 1)
        track = list(self.builder.tracks.values())[0]
        self.assertEqual(track.length, 3)

        # Consistent cycle counted
        self.assertEqual(self.builder.stats["consistent_cycles_detected"], 1)
        # Contributing edge recorded
        self.assertEqual(len(track.contributing_edges), 3)

    def test_invalid_keypoint_index_protection(self):
        """Test that out-of-range keypoint indices are rejected safely without crashes."""
        # im_A only has keypoints 0..4
        success = self.builder.add_verified_correspondence("im_A", 99, "im_B", 0)
        self.assertFalse(success)
        self.assertEqual(self.builder.stats["unions_rejected_invalid_keypoint"], 1)

        # Negative index
        success_neg = self.builder.add_verified_correspondence("im_A", -1, "im_B", 0)
        self.assertFalse(success_neg)

        # Unknown image
        success_unknown = self.builder.add_verified_correspondence("im_UNKNOWN", 0, "im_B", 0)
        self.assertFalse(success_unknown)

        # No tracks created
        self.assertEqual(len(self.builder.tracks), 0)

    def test_self_image_matching_rejection(self):
        """Test that matching keypoints within the same image is rejected."""
        success = self.builder.add_verified_correspondence("im_A", 0, "im_A", 1)
        self.assertFalse(success)
        self.assertEqual(self.builder.stats["unions_rejected_self_image"], 1)
        self.assertEqual(len(self.builder.tracks), 0)

    def test_rejected_pairwise_edge_handling(self):
        """Test that an edge with low inliers is classified as REJECTED and not merged into tracks."""
        edge = PairwiseMatchEdge(
            image_id1="im_A",
            image_id2="im_B",
            num_raw_matches=10,
            num_good_matches=10,
            num_inliers=2,  # below threshold 5
            inlier_ratio=0.2,  # below threshold 0.3
            status="REJECTED",
            rejection_reason="LOW_INLIER_COUNT",
            inlier_pairs=[(0, 0), (1, 1)]
        )
        self.builder.pairwise_edges[("im_A", "im_B")] = edge

        # Attempt to build tracks from edges
        self.builder.build_tracks_from_edges()

        # Rejected edge must NOT be merged
        self.assertEqual(len(self.builder.tracks), 0)


if __name__ == "__main__":
    unittest.main()
