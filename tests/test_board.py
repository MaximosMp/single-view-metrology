"""Tests for the board geometry."""

from __future__ import annotations

import numpy as np
import pytest

from svmetrology.board import CircleGrid


def test_one_point_per_dot_in_float32():
    points = CircleGrid(4, 11, 1.0).object_points()
    assert points.shape == (44, 3)
    assert points.dtype == np.float32


def test_the_board_is_flat():
    assert np.all(CircleGrid(4, 11, 0.83).object_points()[:, 2] == 0.0)


def test_odd_rows_shift_by_one_pitch():
    # The shift is the asymmetry that makes the grid's orientation unambiguous.
    points = CircleGrid(4, 11, 1.0).object_points()
    assert points[4, 0] - points[0, 0] == pytest.approx(1.0)
    assert points[8, 0] == pytest.approx(points[0, 0])


def test_dots_in_a_row_sit_two_pitches_apart():
    points = CircleGrid(4, 11, 0.5).object_points()
    assert np.allclose(np.diff(points[:4, 0]), 1.0)
    assert np.allclose(points[:4, 1], 0.0)


def test_points_scale_with_the_pitch():
    assert np.allclose(CircleGrid(4, 11, 2.0).object_points(), 2.0 * CircleGrid(4, 11, 1.0).object_points())


def test_shape_is_what_opencv_expects():
    assert CircleGrid(4, 11, 1.0).shape == (4, 11)


def test_shipped_board_is_the_4x11_grid(board: CircleGrid):
    assert board.shape == (4, 11)
    assert board.pitch_cm > 0
