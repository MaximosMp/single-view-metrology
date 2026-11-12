"""Shared fixtures for the test suite.

``fixtures/ruler.jpg`` is one photo of the board and a ruler, with the ruler's
mask beside it. Both are downscaled to a 1600 px long edge and written with
``cv2.imwrite``, which emits no metadata: phone originals carry EXIF GPS and
must never enter the repository.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from svmetrology.board import CircleGrid
from svmetrology.calibration import CalibrationConfig, CameraCalibration
from svmetrology.config import load_yaml_as, resolve_path
from svmetrology.depth import BoardLimits, DepthSetup
from svmetrology.detection import find_grid
from svmetrology.imaging import load_mask
from svmetrology.pose import BoardPose, solve_board_pose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
CALIBRATION_CONFIG = PROJECT_ROOT / "configs" / "calibration.yaml"


@pytest.fixture(scope="session")
def board_photo() -> np.ndarray:
    """Return the committed photo of the board and a ruler, downscaled to 1600x1200.

    Returns:
        H x W x 3 ``np.ndarray`` with ``dtype=uint8``, BGR, sensor frame.
    """
    bgr = cv2.imread(str(FIXTURE_DIR / "ruler.jpg"), cv2.IMREAD_COLOR)
    assert bgr is not None, "fixture image is missing or unreadable"
    return bgr


@pytest.fixture(scope="session")
def board_gray(board_photo: np.ndarray) -> np.ndarray:
    """Return the committed board photo as grayscale.

    Returns:
        H x W ``np.ndarray`` with ``dtype=uint8``.
    """
    return cv2.cvtColor(board_photo, cv2.COLOR_BGR2GRAY)


@pytest.fixture(scope="session")
def ruler_mask() -> np.ndarray:
    """Return the committed mask of the ruler as a boolean array.

    Loaded through the pipeline's own reader, so a change to how masks are
    decoded shows up here rather than being masked by a hand-rolled copy.

    Returns:
        H x W ``np.ndarray`` with ``dtype=bool``, True where the object is.
    """
    return load_mask(FIXTURE_DIR / "ruler_mask.png")


@pytest.fixture(scope="session")
def ruler_pair() -> tuple[Path, Path]:
    """Return the committed photo and mask of the ruler.

    Returns:
        ``(image_path, mask_path)``, the same size as each other.
    """
    return FIXTURE_DIR / "ruler.jpg", FIXTURE_DIR / "ruler_mask.png"


@pytest.fixture(scope="session")
def calibration_config() -> CalibrationConfig:
    """Return the project's real calibration settings.

    Loading the shipped config rather than a hand-built one keeps the tests
    honest: a change that breaks detection in practice breaks them too.

    Returns:
        The validated calibration config.
    """
    return load_yaml_as(CALIBRATION_CONFIG, CalibrationConfig)


@pytest.fixture(scope="session")
def board(calibration_config: CalibrationConfig) -> CircleGrid:
    """Return the board the shipped config points at.

    Returns:
        The printed 4x11 board.
    """
    return load_yaml_as(resolve_path(calibration_config.board, CALIBRATION_CONFIG), CircleGrid)


@pytest.fixture(scope="session")
def camera_calibration(calibration_config: CalibrationConfig) -> CameraCalibration:
    """Return the committed calibration the shipped config writes.

    Returns:
        The phone's calibration at native resolution.
    """
    return CameraCalibration.load(resolve_path(calibration_config.output, CALIBRATION_CONFIG))


@pytest.fixture(scope="session")
def board_photo_setup(
    board_photo: np.ndarray,
    camera_calibration: CameraCalibration,
    board: CircleGrid,
    calibration_config: CalibrationConfig,
) -> DepthSetup:
    """Return a depth setup for the downscaled board photo.

    The intrinsics are scaled to the photo's size and the distortion kept, as
    the resolution contract prescribes.

    Returns:
        The setup, with the shipped depth config's limits.
    """
    height, width = board_photo.shape[:2]
    full_width, full_height = camera_calibration.image_size
    scaled = camera_calibration.camera_matrix * np.array([[width / full_width], [height / full_height], [1.0]])
    return DepthSetup(
        calibration=replace(camera_calibration, camera_matrix=scaled, image_size=(width, height)),
        board=board,
        detection=calibration_config.detection,
        limits=BoardLimits(max_pose_error_px=2.0, max_board_error=0.05),
        process_res=756,
        output=Path("unused"),
    )


@pytest.fixture(scope="session")
def board_pose(board_photo: np.ndarray, board_photo_setup: DepthSetup) -> BoardPose:
    """Return the board's pose in the undistorted board photo.

    Returns:
        The pose.
    """
    image = board_photo_setup.calibration.undistort(board_photo)
    centres = find_grid(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), board_photo_setup.board, board_photo_setup.detection)
    assert centres is not None
    return solve_board_pose(centres, board_photo_setup.board, board_photo_setup.calibration.camera_matrix)


@pytest.fixture(scope="session")
def board_plane_depth(board_photo: np.ndarray, board_pose: BoardPose) -> np.ndarray:
    """Return the exact depth of the board's plane at every pixel of the undistorted board photo.

    Returns:
        H x W ``np.ndarray`` with ``dtype=float32``, cm.
    """
    v, u = np.indices(board_photo.shape[:2])
    return board_pose.plane_depth(np.column_stack([u.ravel(), v.ravel()])).reshape(v.shape).astype(np.float32)
