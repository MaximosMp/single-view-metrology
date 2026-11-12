"""Tests for intrinsic calibration."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from svmetrology.board import CircleGrid
from svmetrology.calibration import CameraCalibration, RejectionSettings, View, calibrate, view_rms

BOARD = CircleGrid(4, 11, 1.0)
TRUTH = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
NO_DISTORTION = np.zeros(5)
LENIENT = RejectionSettings(max_view_error_px=1.0, max_rms_px=1.0, min_views=5)


def _project(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(BOARD.object_points(), rvec, tvec, TRUTH, NO_DISTORTION)
    return projected.reshape(-1, 2).astype(np.float32)


def _views(count: int) -> list[View]:
    """Render the board from ``count`` poses through the known camera."""
    return [
        View(
            f"view_{i}.jpg", _project(np.array([0.05 * i, 0.08 * i, 0.02 * i]), np.array([1.0 * i, 0.5 * i, 40.0 + i]))
        )
        for i in range(count)
    ]


# -- Per-view error -----------------------------------------------------------


def test_an_exact_projection_has_no_error():
    rvec, tvec = np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 30.0])
    error = view_rms(BOARD.object_points(), _project(rvec, tvec), rvec, tvec, TRUTH, NO_DISTORTION)
    assert error == pytest.approx(0.0, abs=1e-4)


def test_error_is_the_rms_distance_in_pixels():
    # Every centre moved by (2, 2) is 2*sqrt(2) px from where it should be, so
    # the RMS is exactly that, whatever the dot count.
    rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 30.0])
    error = view_rms(BOARD.object_points(), _project(rvec, tvec) + 2.0, rvec, tvec, TRUTH, NO_DISTORTION)
    assert error == pytest.approx(2 * math.sqrt(2), rel=1e-4)


# -- Fitting and rejection ----------------------------------------------------


def test_recovers_known_intrinsics():
    result = calibrate(_views(12), BOARD, (640, 480), LENIENT)
    assert result.rms_px < 0.1
    assert len(result.views_used) == 12
    assert result.camera_matrix[0, 0] == pytest.approx(900.0, rel=0.02)
    assert result.camera_matrix[1, 1] == pytest.approx(900.0, rel=0.02)
    assert result.distortion.shape == (5,)


def test_leaves_out_a_corrupted_view():
    views = _views(10)
    rng = np.random.default_rng(0)
    views[3] = View(views[3].name, views[3].centres + rng.normal(0, 15, views[3].centres.shape).astype(np.float32))
    result = calibrate(views, BOARD, (640, 480), LENIENT)
    assert "view_3.jpg" not in result.views_used
    assert len(result.views_used) >= LENIENT.min_views


def test_never_drops_below_min_views():
    rng = np.random.default_rng(1)
    noisy = [View(v.name, v.centres + rng.normal(0, 20, v.centres.shape).astype(np.float32)) for v in _views(8)]
    strict = RejectionSettings(max_view_error_px=0.001, max_rms_px=0.001, min_views=6)
    assert len(calibrate(noisy, BOARD, (640, 480), strict).views_used) == 6


# -- Saved file ---------------------------------------------------------------


def _saved_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array([[900.5, 0.0, 320.25], [0.0, 901.5, 240.75], [0.0, 0.0, 1.0]]),
        distortion=np.array([0.1, -0.2, 0.001, -0.002, 0.3]),
        rms_px=0.42,
        image_size=(640, 480),
        views_used=["a.jpg", "b.jpg"],
        board=CircleGrid(4, 11, 0.83),
    )


def test_saved_file_records_everything_downstream_needs(tmp_path: Path):
    path = tmp_path / "nested" / "camera.yaml"
    _saved_calibration().save(path)

    saved = yaml.safe_load(path.read_text())
    assert saved["image_size"] == {"width": 640, "height": 480}
    assert saved["intrinsics_px"] == pytest.approx({"fx": 900.5, "fy": 901.5, "cx": 320.25, "cy": 240.75})
    # OpenCV's order is (k1, k2, p1, p2, k3), easy to transpose by accident.
    assert list(saved["distortion"]) == ["k1", "k2", "p1", "p2", "k3"]
    assert saved["distortion"]["k3"] == pytest.approx(0.3)
    assert saved["fit"] == {"rms_px": pytest.approx(0.42), "views_used": ["a.jpg", "b.jpg"]}
    assert saved["board"] == {"cols": 4, "rows": 11, "pitch_cm": pytest.approx(0.83)}


def test_loading_returns_what_was_saved(tmp_path: Path):
    original = _saved_calibration()
    path = tmp_path / "camera.yaml"
    original.save(path)

    loaded = CameraCalibration.load(path)

    assert np.allclose(loaded.camera_matrix, original.camera_matrix)
    assert np.allclose(loaded.distortion, original.distortion)
    assert loaded.distortion.shape == (5,)
    assert loaded.rms_px == pytest.approx(original.rms_px)
    assert loaded.image_size == original.image_size
    assert loaded.views_used == original.views_used
    assert loaded.board == original.board


def test_loading_names_a_missing_key(tmp_path: Path):
    path = tmp_path / "camera.yaml"
    _saved_calibration().save(path)
    saved = yaml.safe_load(path.read_text())
    del saved["intrinsics_px"]["cy"]
    path.write_text(yaml.safe_dump(saved))

    with pytest.raises(ValueError, match="cy"):
        CameraCalibration.load(path)


def test_the_shipped_calibration_matches_the_shipped_board(camera_calibration: CameraCalibration, board: CircleGrid):
    assert camera_calibration.board == board
