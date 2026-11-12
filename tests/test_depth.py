"""Tests for board-scaled metric depth, with a stand-in for the network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from svmetrology.calibration import CameraCalibration
from svmetrology.depth import DepthSetup, MetricDepth, board_pixels, fit_scale, scaled_depth
from svmetrology.pose import BoardPose

DEPTH_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "depth.yaml"


class _StandInModel:
    """Returns a fixed depth map, recording that it was asked."""

    def __init__(self, depth_cm: np.ndarray) -> None:
        self.depth_cm = depth_cm
        self.calls = 0

    def predict(self, image: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:  # noqa: ARG002
        self.calls += 1
        return self.depth_cm


# -- Model input ------------------------------------------------------------------


def test_the_network_sees_a_reduced_rgb_photo_and_matching_intrinsics():
    seen = {}

    def network(image: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
        seen["image"], seen["k"] = image, camera_matrix.numpy()
        return torch.full(image.shape[-2:], 0.5)  # metres

    # Built without __init__, which would download the real weights.
    model = MetricDepth.__new__(MetricDepth)
    model.process_res, model.network = 200, network

    photo = np.zeros((300, 400, 3), np.uint8)
    photo[..., 0] = 255  # blue, in OpenCV's BGR order
    k = np.array([[1000.0, 0.0, 199.5], [0.0, 1010.0, 149.5], [0.0, 0.0, 1.0]])
    depth_cm = model.predict(photo, k)

    assert tuple(seen["image"].shape) == (3, 150, 200)
    assert seen["image"][2].min() == 255  # blue arrives as the last channel: RGB
    assert seen["image"][0].max() == 0
    # Half the size: focal lengths halve, and the centre of a 400 x 300 photo
    # stays the centre of the 200 x 150 one.
    assert seen["k"] == pytest.approx(np.array([[500.0, 0.0, 99.5], [0.0, 505.0, 74.5], [0.0, 0.0, 1.0]]))
    assert k[0, 0] == 1000.0  # the caller's matrix is untouched
    assert depth_cm.shape == (300, 400)
    assert depth_cm == pytest.approx(50.0)


# -- Scale fit --------------------------------------------------------------------


def test_recovers_the_scale_of_a_correctly_shaped_depth(board_pose: BoardPose, board_plane_depth: np.ndarray):
    true = board_plane_depth
    pixels = np.column_stack([np.arange(400, 800), np.full(400, 600)])
    fit = fit_scale(true * 2.5, board_pose, pixels)
    assert fit.scale == pytest.approx(0.4)
    assert fit.board_error == pytest.approx(0.0, abs=1e-6)


def test_a_glare_spot_does_not_move_the_scale(board_pose: BoardPose, board_plane_depth: np.ndarray):
    predicted = board_plane_depth * 2.5
    pixels = np.column_stack([np.arange(400, 800), np.full(400, 600)])
    predicted[600, 400:440] *= 3.0  # a tenth of the board wildly off
    assert fit_scale(predicted, board_pose, pixels).scale == pytest.approx(0.4)


def test_board_pixels_fill_the_hull_of_the_dots():
    centres = np.array([[10, 10], [30, 10], [30, 20], [10, 20]], dtype=np.float32)
    pixels = board_pixels(centres, (40, 40))
    assert len(pixels) == 21 * 11
    assert pixels[:, 0].min() == 10 and pixels[:, 0].max() == 30


# -- Whole chain ------------------------------------------------------------------


def test_scales_depth_onto_the_board(
    board_photo: np.ndarray, board_photo_setup: DepthSetup, board_plane_depth: np.ndarray
):
    true = board_plane_depth
    result = scaled_depth(board_photo, board_photo_setup, _StandInModel(true * 3.0))

    assert result.fit.scale == pytest.approx(1 / 3, rel=1e-3)
    assert result.fit.board_error < 1e-3
    on_board = board_pixels(result.centres, true.shape)
    assert result.depth_cm[on_board[:, 1], on_board[:, 0]] == pytest.approx(
        true[on_board[:, 1], on_board[:, 0]], rel=1e-3
    )


def test_rejects_depth_that_does_not_fit_the_board(
    board_photo: np.ndarray, board_photo_setup: DepthSetup, board_plane_depth: np.ndarray
):
    true = board_plane_depth
    # Vertical bands half as far again as their neighbours: wherever the board
    # sits in the photo, no single scale fits it.
    banded = true * np.where(np.arange(true.shape[1]) % 20 < 10, 1.0, 1.5).astype(np.float32)
    with pytest.raises(ValueError, match="Board depth error"):
        scaled_depth(board_photo, board_photo_setup, _StandInModel(banded))


def test_a_photo_without_the_board_fails_before_the_model_runs(board_photo_setup: DepthSetup):
    model = _StandInModel(np.ones((1200, 1600), np.float32))
    with pytest.raises(ValueError, match="Board not found"):
        scaled_depth(np.full((1200, 1600, 3), 255, np.uint8), board_photo_setup, model)
    assert model.calls == 0


def test_a_photo_at_another_resolution_is_refused(board_photo_setup: DepthSetup):
    with pytest.raises(ValueError, match="calibration is for 1600x1200"):
        board_photo_setup.calibration.undistort(np.zeros((600, 800, 3), np.uint8))


# -- Config -----------------------------------------------------------------------


def test_the_shipped_depth_config_resolves_everything_it_points_at(camera_calibration: CameraCalibration):
    setup = DepthSetup.load(DEPTH_CONFIG)
    assert setup.process_res == 1904
    assert setup.calibration.image_size == camera_calibration.image_size
    assert setup.board == camera_calibration.board
