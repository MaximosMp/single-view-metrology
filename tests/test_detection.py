"""Tests for finding the circle grid."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from svmetrology.board import CircleGrid
from svmetrology.calibration import CalibrationConfig
from svmetrology.detection import find_grid


def _drawn_board(
    board: CircleGrid, pitch_px: float, origin: tuple[float, float], size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the board's dots, black on white, and return the image and their centres."""
    centres = board.object_points()[:, :2] / board.pitch_cm * pitch_px + origin
    image = np.full(size[::-1], 255, np.uint8)
    for x, y in centres:
        # shift=4 places the centre to a sixteenth of a pixel.
        cv2.circle(image, (round(x * 16), round(y * 16)), round(pitch_px * 0.3 * 16), 0, -1, cv2.LINE_AA, shift=4)
    return image, centres


def test_finds_the_grid_in_a_real_photo(
    board_gray: np.ndarray, board: CircleGrid, calibration_config: CalibrationConfig
):
    centres = find_grid(board_gray, board, calibration_config.detection)
    assert centres is not None, "the 4x11 grid must be found in the fixture photo"
    assert centres.shape == (44, 2)
    assert centres.dtype == np.float32
    height, width = board_gray.shape
    assert np.all((centres >= 0) & (centres < [width, height]))


@pytest.mark.parametrize("pitch_px", [20, 45, 110])
def test_finds_a_drawn_board_at_any_size_to_a_fraction_of_a_pixel(
    pitch_px: int, board: CircleGrid, calibration_config: CalibrationConfig
):
    image, drawn = _drawn_board(board, pitch_px, origin=(150.0, 100.0), size=(1600, 1400))
    centres = find_grid(image, board, calibration_config.detection)
    assert centres is not None
    assert np.abs(centres - drawn).max() < 0.2


def test_misses_dots_outside_every_size_tried(board: CircleGrid, calibration_config: CalibrationConfig):
    image, _ = _drawn_board(board, 45, origin=(150.0, 100.0), size=(1600, 1400))
    far_too_small = replace(calibration_config.detection, dot_sizes=[0.001])
    assert find_grid(image, board, far_too_small) is None


def test_the_sizes_tried_leave_no_gaps(calibration_config: CalibrationConfig):
    settings = calibration_config.detection
    steps = np.divide(settings.dot_sizes[:-1], settings.dot_sizes[1:])
    assert np.all(steps > 1), "largest first"
    assert np.all(settings.size_tolerance**2 >= steps), "neighbouring tries must overlap"


def test_reports_no_grid_in_a_blank_image(board: CircleGrid, calibration_config: CalibrationConfig):
    assert find_grid(np.full((400, 600), 255, dtype=np.uint8), board, calibration_config.detection) is None
