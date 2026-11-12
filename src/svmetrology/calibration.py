"""Intrinsic calibration: camera matrix and lens distortion from views of the board.

One blurred shot or one curled board drags the whole solution, so the fit is
repeated with the worst view left out until every remaining view agrees with
it, or until too few would remain to trust. The result is written as YAML and
is the pipeline's record of the camera; later stages read it back with
`CameraCalibration.load`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from svmetrology.board import CircleGrid
from svmetrology.config import load_yaml_as
from svmetrology.detection import DetectionSettings
from svmetrology.reporting import write_yaml

log = logging.getLogger(__name__)


@dataclass
class RejectionSettings:
    """When the fit stops leaving out views.

    Attributes:
        max_view_error_px: Every view's RMS reprojection error must be at most this.
        max_rms_px: The overall RMS reprojection error must be at most this.
        min_views: Never leave out a view once this few remain.
    """

    max_view_error_px: float
    max_rms_px: float
    min_views: int


@dataclass
class CalibrationConfig:
    """Settings for a calibration run.

    Attributes:
        images: Folder of board photos.
        output: Where the calibration YAML is written.
        board: Board description YAML.
        max_long_edge: Calibrate at this long edge in px, or ``None`` for the
            native sensor resolution.
        detection: How the grid is found in each photo.
        rejection: When the fit stops leaving views out.
    """

    images: str
    output: str
    board: str
    max_long_edge: int | None
    detection: DetectionSettings
    rejection: RejectionSettings


@dataclass
class View:
    """The board as detected in one image.

    Attributes:
        name: Image filename, for reporting which views were kept.
        centres: N x 2 ``np.ndarray`` with ``dtype=float32``, dot centres in px,
            in `CircleGrid.object_points` order.
    """

    name: str
    centres: np.ndarray


@dataclass
class CameraCalibration:
    """A solved camera.

    Attributes:
        camera_matrix: 3 x 3 ``np.ndarray`` with ``dtype=float64``, in px.
        distortion: ``np.ndarray`` of 5 coefficients in OpenCV order
            ``(k1, k2, p1, p2, k3)``.
        rms_px: Overall RMS reprojection error of the fit.
        image_size: ``(width, height)`` the intrinsics belong to, in px.
        views_used: Names of the views the final fit kept.
        board: The board the views show.
    """

    camera_matrix: np.ndarray
    distortion: np.ndarray
    rms_px: float
    image_size: tuple[int, int]
    views_used: list[str]
    board: CircleGrid

    def save(self, path: Path) -> None:
        """Write the calibration as YAML.

        Args:
            path: Destination file; parent directories are created.

        Raises:
            OSError: the path is not writable.
        """
        k = self.camera_matrix
        width, height = self.image_size
        record = _CalibrationFile(
            image_size=_ImageSize(width, height),
            intrinsics_px=_Intrinsics(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])),
            distortion=_Distortion(*map(float, self.distortion)),
            fit=_Fit(float(self.rms_px), self.views_used),
            board=self.board,
        )
        write_yaml(path, asdict(record))

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion, keeping ``camera_matrix`` valid for the result.

        Args:
            image: H x W or H x W x C ``np.ndarray``, decoded in the sensor
                frame at exactly ``image_size``.

        Returns:
            The undistorted image, same shape and dtype.

        Raises:
            ValueError: the image is not at the calibrated resolution.
        """
        height, width = image.shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(
                f"Image is {width}x{height}, the calibration is for {self.image_size[0]}x{self.image_size[1]}"
            )
        return cv2.undistort(image, self.camera_matrix, self.distortion)

    @classmethod
    def load(cls, path: Path) -> CameraCalibration:
        """Read a calibration written by `save`.

        The intrinsics belong to ``image_size``; scale them before use at any
        other resolution.

        Args:
            path: Calibration YAML.

        Returns:
            The calibration, with ``camera_matrix`` and ``distortion`` in the
            same form `calibrate` produces.

        Raises:
            FileNotFoundError: the file does not exist.
            ValueError: a key is missing, unknown, or holds the wrong type.
        """
        record = load_yaml_as(path, _CalibrationFile)
        k, d = record.intrinsics_px, record.distortion
        return cls(
            camera_matrix=np.array([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]]),
            distortion=np.array([d.k1, d.k2, d.p1, d.p2, d.k3]),
            rms_px=record.fit.rms_px,
            image_size=(record.image_size.width, record.image_size.height),
            views_used=record.fit.views_used,
            board=record.board,
        )


# The calibration file's layout. `save` and `load` both go through it, so the
# key names are written down once.


@dataclass
class _ImageSize:
    """Resolution the intrinsics belong to, in px."""

    width: int
    height: int


@dataclass
class _Intrinsics:
    """Focal lengths and principal point, in px."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class _Distortion:
    """Distortion coefficients in OpenCV order."""

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float


@dataclass
class _Fit:
    """How well the fit agreed, and which views it kept."""

    rms_px: float
    views_used: list[str]


@dataclass
class _CalibrationFile:
    """The whole calibration file."""

    # Intrinsics are only valid at the resolution they were solved at, so the
    # size leads rather than trailing as metadata.
    image_size: _ImageSize
    intrinsics_px: _Intrinsics
    distortion: _Distortion
    fit: _Fit
    board: CircleGrid


def view_rms(
    object_points: np.ndarray,
    centres: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    """Return one view's RMS reprojection error.

    The same statistic `cv2.calibrateCamera` reports for the whole fit, so a
    per-view threshold and the overall one are in the same units.

    Args:
        object_points: N x 3 ``np.ndarray``, the board's dot centres.
        centres: N x 2 ``np.ndarray``, where those dots were detected, in px.
        rvec: Rotation vector of the board in this view.
        tvec: Translation vector of the board in this view.
        camera_matrix: 3 x 3 ``np.ndarray``.
        distortion: Distortion coefficients in OpenCV order.

    Returns:
        RMS distance between detected and reprojected centres, in px.
    """
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    residuals = centres.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def calibrate(
    views: list[View],
    board: CircleGrid,
    image_size: tuple[int, int],
    rejection: RejectionSettings,
) -> CameraCalibration:
    """Solve the intrinsics, leaving out the worst view until the rest agree.

    Args:
        views: Detections, at least 3.
        board: The board the views show.
        image_size: ``(width, height)`` of every view, in px.
        rejection: When to stop leaving views out.

    Returns:
        The calibration from the last fit.
    """
    board_points = board.object_points()
    kept = list(views)
    while True:
        rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            [board_points] * len(kept), [v.centres for v in kept], image_size, None, None
        )
        errors = [
            view_rms(board_points, v.centres, r, t, camera_matrix, distortion)
            for v, r, t in zip(kept, rvecs, tvecs, strict=True)
        ]
        worst = int(np.argmax(errors))
        agreed = errors[worst] <= rejection.max_view_error_px and rms <= rejection.max_rms_px
        if agreed or len(kept) <= rejection.min_views:
            break
        log.info("Leaving out %s (%.3f px), %d views remain", kept[worst].name, errors[worst], len(kept) - 1)
        del kept[worst]

    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion=distortion.ravel(),
        rms_px=float(rms),
        image_size=image_size,
        views_used=[v.name for v in kept],
        board=board,
    )
