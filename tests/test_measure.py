"""Tests for measuring a masked object from board-scaled depth."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from svmetrology.depth import DepthSetup, ScaledDepth, ScaleFit
from svmetrology.detection import find_grid
from svmetrology.measure import back_project, bounding_rectangle, corner_depths, measure, save_measurements
from svmetrology.pose import BoardPose

# The longest chord across the 4x11 grid's dot centres joins dots 6 pitches
# apart in x and 10 in y.
DOT_HULL_DIAMETER_PITCHES = math.hypot(6, 10)
# The dots span 10 pitches down the rows and, with the shifted rows, 7 across.
DOT_HULL_SIDES_PITCHES = (10, 7)


@pytest.fixture(scope="module")
def dot_hull_mask(board_photo: np.ndarray, board_photo_setup: DepthSetup) -> np.ndarray:
    """A mask over the convex hull of the dots, drawn on the photo before undistortion."""
    centres = find_grid(
        cv2.cvtColor(board_photo, cv2.COLOR_BGR2GRAY), board_photo_setup.board, board_photo_setup.detection
    )
    assert centres is not None
    mask = np.zeros(board_photo.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(centres).astype(np.int32), 1)
    return mask > 0


def _scaled(depth_cm: np.ndarray, setup: DepthSetup, pose: BoardPose) -> ScaledDepth:
    return ScaledDepth(
        image=np.zeros((*depth_cm.shape, 3), np.uint8),
        depth_cm=depth_cm,
        camera_matrix=setup.calibration.camera_matrix,
        centres=np.zeros((0, 2)),
        pose=pose,
        fit=ScaleFit(scale=1.0, board_error=0.0),
    )


# -- Pieces -----------------------------------------------------------------------


def test_back_projection_inverts_projection():
    k = np.array([[1000.0, 0.0, 400.0], [0.0, 1010.0, 300.0], [0.0, 0.0, 1.0]])
    points = np.array([[1.0, -2.0, 15.0], [-3.0, 0.5, 20.0]])
    pixels = (k @ (points / points[:, 2:]).T).T[:, :2]
    assert back_project(pixels, points[:, 2], k) == pytest.approx(points)


def test_corner_depth_comes_from_inside_the_outline():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    depth = np.full((100, 100), 15.0)
    depth[20:26, 20:80] = 40.0  # the background bleeding in along one edge
    assert corner_depths(np.array([[50, 20], [20, 20]]), mask, depth, margin_px=10) == pytest.approx([15.0, 15.0])


def test_a_mask_thinner_than_the_margin_is_refused():
    mask = np.zeros((100, 100), bool)
    mask[50:54, 10:90] = True
    with pytest.raises(ValueError, match="edge margin"):
        corner_depths(np.array([[10, 50]]), mask, np.ones((100, 100)), margin_px=5)


def _tilted(flat: np.ndarray) -> np.ndarray:
    """Place 2D points on a plane tilted about two axes, 40 cm from the camera."""
    rotation = cv2.Rodrigues(np.array([0.5, -0.3, 0.8]))[0]
    return (rotation @ np.column_stack([flat, np.zeros(len(flat))]).T).T + [3.0, -2.0, 40.0]


def _sides(box: np.ndarray) -> list[float]:
    """A rectangle's two side lengths, the longer first."""
    return sorted(np.linalg.norm(box[:2] - box[1:3], axis=1), reverse=True)


def test_the_rectangle_around_a_tilted_rectangle_is_itself():
    corners = np.array([[0.0, 0.0], [31.39, 0.0], [31.39, 3.35], [0.0, 3.35]])
    assert _sides(bounding_rectangle(_tilted(corners))) == pytest.approx([31.39, 3.35], rel=1e-4)


def test_rectangle_around_another_shape_is_the_box_it_fits_in():
    # An elongated hexagon: 12 cm between its points, 4 cm across its flats.
    hexagon = np.array([[0.0, 0.0], [8.0, 0.0], [10.0, 2.0], [8.0, 4.0], [0.0, 4.0], [-2.0, 2.0]])
    assert _sides(bounding_rectangle(_tilted(hexagon))) == pytest.approx([12.0, 4.0], rel=1e-4)


# -- Whole measurement ------------------------------------------------------------


def test_measures_the_dot_hull_with_exact_depth(
    dot_hull_mask: np.ndarray, board_photo_setup: DepthSetup, board_pose: BoardPose, board_plane_depth: np.ndarray
):
    result = measure(
        dot_hull_mask, _scaled(board_plane_depth, board_photo_setup, board_pose), board_photo_setup.calibration, 0
    )
    pitch = board_photo_setup.board.pitch_cm
    assert result.longest_cm == pytest.approx(DOT_HULL_DIAMETER_PITCHES * pitch, rel=5e-3)
    # The rectangle bounds the rasterised outline from outside, so at this
    # photo's size it carries a pixel or two of the mask's edge that the chord,
    # running corner to corner, does not.
    sides = (DOT_HULL_SIDES_PITCHES[0] * pitch, DOT_HULL_SIDES_PITCHES[1] * pitch)
    assert (result.length_cm, result.width_cm) == pytest.approx(sides, rel=1e-2)


def test_the_edge_margin_keeps_background_depth_out_of_the_length(
    dot_hull_mask: np.ndarray, board_photo_setup: DepthSetup, board_pose: BoardPose, board_plane_depth: np.ndarray
):
    undistorted = board_photo_setup.calibration.undistort(dot_hull_mask.astype(np.uint8) * 255) > 127
    band = undistorted & ~(cv2.erode(undistorted.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    bled = np.where(band, board_plane_depth * 1.5, board_plane_depth).astype(np.float32)
    depth = _scaled(bled, board_photo_setup, board_pose)
    expected = DOT_HULL_DIAMETER_PITCHES * board_photo_setup.board.pitch_cm

    assert measure(dot_hull_mask, depth, board_photo_setup.calibration, 0).longest_cm != pytest.approx(
        expected, rel=0.05
    )
    assert measure(dot_hull_mask, depth, board_photo_setup.calibration, 10).longest_cm == pytest.approx(
        expected, rel=1e-2
    )


# -- Report -----------------------------------------------------------------------


def test_report_states_its_pixel_frame(tmp_path: Path):
    out = tmp_path / "nested" / "measurements.yaml"
    results = [
        {"image": "a.jpg", "length_cm": 8.3, "width_cm": 5.81, "longest_distance_cm": 9.68, "component_count": 1}
    ]
    save_measurements(out, results, unpaired=["stray.png"], failed=[{"image": "b.jpg", "reason": "no board"}])

    loaded = yaml.safe_load(out.read_text())
    assert loaded["pixel_frame"] == "undistorted"
    assert loaded["results"] == results
    assert loaded["unpaired"] == ["stray.png"]
    assert loaded["failed"][0]["reason"] == "no board"
