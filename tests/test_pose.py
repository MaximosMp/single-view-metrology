"""Tests for the board pose and its plane depth."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from svmetrology.board import CircleGrid
from svmetrology.pose import NO_DISTORTION, solve_board_pose

BOARD = CircleGrid(4, 11, 0.83)
K = np.array([[1370.0, 0.0, 930.0], [0.0, 1372.0, 716.0], [0.0, 0.0, 1.0]])
RVEC = np.array([0.3, -0.2, 0.1])
TVEC = np.array([-3.0, -4.0, 18.0])


def _image_and_depth(board_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project board-frame points, returning their pixels and true camera-frame depths."""
    pixels, _ = cv2.projectPoints(board_points, RVEC, TVEC, K, NO_DISTORTION)
    rotation, _ = cv2.Rodrigues(RVEC)
    depth = (rotation @ board_points.T + TVEC[:, None])[2]
    return pixels.reshape(-1, 2).astype(np.float32), depth


def test_recovers_the_depth_of_every_dot():
    centres, depth = _image_and_depth(BOARD.object_points().astype(np.float64))
    pose = solve_board_pose(centres, BOARD, K)
    assert pose.plane_depth(centres) == pytest.approx(depth, abs=1e-4)
    assert pose.reprojection_px < 1e-3


def test_extends_the_plane_beyond_the_dots():
    centres, _ = _image_and_depth(BOARD.object_points().astype(np.float64))
    off_board = np.array([[15.0, -6.0, 0.0], [-8.0, 20.0, 0.0]])
    pixels, depth = _image_and_depth(off_board)
    assert solve_board_pose(centres, BOARD, K).plane_depth(pixels) == pytest.approx(depth, rel=1e-4)


def test_reports_how_far_noisy_dots_sit_from_the_pose():
    centres, _ = _image_and_depth(BOARD.object_points().astype(np.float64))
    noisy = centres + np.random.default_rng(0).normal(0.0, 0.5, centres.shape).astype(np.float32)
    assert 0.2 < solve_board_pose(noisy, BOARD, K).reprojection_px < 0.8
