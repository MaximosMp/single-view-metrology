"""Metric depth from one photo, corrected to the board's scale.

A monocular model gets the shape of a scene right far more reliably than its
absolute distance: at 10-50 cm, UniDepth puts the board 1.2 to 3.8 times too
far away, while its shape along the board is within a percent or two. So the
model's depth is rescaled by one factor, the one that best puts the board where
its pose says it is. A photo whose board is missing, badly posed or poorly
matched fails here rather than yielding an unscaled number.

Everything works in the undistorted image at the calibrated resolution, and
depths are in cm, the board's unit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from svmetrology._vendor.unidepth import UniDepthV2
from svmetrology.board import CircleGrid
from svmetrology.calibration import CalibrationConfig, CameraCalibration
from svmetrology.config import load_yaml_as, resolve_path
from svmetrology.detection import DetectionSettings, find_grid
from svmetrology.imaging import resize_longest_edge
from svmetrology.pose import BoardPose, solve_board_pose

log = logging.getLogger(__name__)

# The checkpoint the vendored network was verified against; a new one is a
# code change, not a setting.
MODEL_REPO = "lpiccinelli/unidepth-v2-vitl14"
MODEL_REVISION = "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9"
CM_PER_M = 100.0


@dataclass
class BoardLimits:
    """When a photo's board is not trusted to set the scale.

    Attributes:
        max_pose_error_px: Largest RMS reprojection error of the board pose.
        max_board_error: Largest mean relative depth error across the board
            after the scale correction, as a fraction.
    """

    max_pose_error_px: float
    max_board_error: float


@dataclass
class DepthConfig:
    """Settings for a depth run.

    Attributes:
        calibration_config: The calibration config; its board, detection
            settings and calibration file are reused rather than repeated.
        output: Folder for results.
        process_res: Long side in px a photo is reduced to before the network.
        limits: When the board is rejected.
    """

    calibration_config: str
    output: str
    process_res: int
    limits: BoardLimits


@dataclass
class DepthSetup:
    """Everything a depth run reads before its first photo.

    Attributes:
        calibration: The camera.
        board: The board in the photos.
        detection: How the board is found.
        limits: When the board is rejected.
        process_res: Long side in px a photo is reduced to before the network.
        output: Folder for results.
    """

    calibration: CameraCalibration
    board: CircleGrid
    detection: DetectionSettings
    limits: BoardLimits
    process_res: int
    output: Path

    @classmethod
    def load(cls, config_path: Path) -> DepthSetup:
        """Load a depth config and the calibration config it points at.

        Args:
            config_path: Depth config YAML.

        Returns:
            The resolved setup.

        Raises:
            FileNotFoundError: a config, board or calibration file is missing.
            ValueError: one of them has a missing, unknown or mistyped key.
        """
        cfg = load_yaml_as(config_path, DepthConfig)
        calibration_config_path = resolve_path(cfg.calibration_config, config_path)
        calibration_config = load_yaml_as(calibration_config_path, CalibrationConfig)
        return cls(
            calibration=CameraCalibration.load(resolve_path(calibration_config.output, calibration_config_path)),
            board=load_yaml_as(resolve_path(calibration_config.board, calibration_config_path), CircleGrid),
            detection=calibration_config.detection,
            limits=cfg.limits,
            process_res=cfg.process_res,
            output=resolve_path(cfg.output, config_path),
        )


@dataclass
class ScaleFit:
    """The board's correction of the model's depth.

    Attributes:
        scale: Factor applied to the model's depth.
        board_error: Mean relative depth error across the board after the
            correction, as a fraction.
    """

    scale: float
    board_error: float


@dataclass
class ScaledDepth:
    """A photo's metric depth and the evidence behind its scale.

    Attributes:
        image: H x W x 3 ``np.ndarray`` with ``dtype=uint8``, the undistorted
            photo, BGR.
        depth_cm: H x W ``np.ndarray`` with ``dtype=float32``, corrected depth
            along the optical axis, in cm.
        camera_matrix: 3 x 3 ``np.ndarray``, valid for ``image``.
        centres: N x 2 ``np.ndarray``, the board's dot centres in ``image``.
        pose: The board's pose.
        fit: The scale correction.
    """

    image: np.ndarray
    depth_cm: np.ndarray
    camera_matrix: np.ndarray
    centres: np.ndarray
    pose: BoardPose
    fit: ScaleFit


class MetricDepth:
    """UniDepth V2, loaded once and run per photo."""

    def __init__(self, process_res: int) -> None:
        """Download (once) and load the model on the fastest available device.

        Args:
            process_res: Long side in px a photo is reduced to before the network.
        """
        self.process_res = process_res
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.network = UniDepthV2()
        self.network.load_checkpoint(
            load_file(hf_hub_download(MODEL_REPO, "model.safetensors", revision=MODEL_REVISION))
        )
        self.network.to(self.device).eval()
        log.debug("UniDepth loaded on %s", self.device)

    def predict(self, image: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Predict metric depth, before any board correction.

        Args:
            image: H x W x 3 ``np.ndarray`` with ``dtype=uint8``, undistorted, BGR.
            camera_matrix: 3 x 3 ``np.ndarray`` valid for ``image``.

        Returns:
            H x W ``np.ndarray`` with ``dtype=float32``, depth along the
            optical axis in cm.
        """
        height, width = image.shape[:2]
        # The network resamples its input without antialiasing, so a
        # full-resolution photo is area-averaged down before it gets there.
        small = resize_longest_edge(image, self.process_res)
        scale_x, scale_y = small.shape[1] / width, small.shape[0] / height

        # The intrinsics follow the resize; the principal point moves under
        # the pixel-centre convention, where pixel 0 spans -0.5 to 0.5.
        intrinsics = camera_matrix.astype(np.float64)
        intrinsics[0] *= scale_x
        intrinsics[1] *= scale_y
        intrinsics[0, 2] = (camera_matrix[0, 2] + 0.5) * scale_x - 0.5
        intrinsics[1, 2] = (camera_matrix[1, 2] + 0.5) * scale_y - 0.5

        rgb = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        depth_m = self.network(rgb, torch.from_numpy(intrinsics)).cpu().numpy()
        return cv2.resize(depth_m * CM_PER_M, (width, height), interpolation=cv2.INTER_LINEAR)


def board_pixels(centres: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return every pixel inside the convex hull of the board's dots.

    Args:
        centres: N x 2 ``np.ndarray``, dot centres in px.
        shape: ``(height, width)`` of the image.

    Returns:
        M x 2 ``np.ndarray`` with ``dtype=int64``, ``(u, v)`` pixel coordinates.
    """
    region = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(centres.astype(np.float32)).astype(np.int32), 1)
    v, u = np.nonzero(region)
    return np.column_stack([u, v])


def fit_scale(depth_cm: np.ndarray, pose: BoardPose, pixels: np.ndarray) -> ScaleFit:
    """Fit the factor that best puts the board where its pose says it is.

    The median of per-pixel ratios, so a glare spot or a curled corner does not
    drag the scale.

    Args:
        depth_cm: H x W ``np.ndarray``, the model's depth in cm.
        pose: The board's pose in the same image.
        pixels: M x 2 ``np.ndarray`` of ``(u, v)`` on the board.

    Returns:
        The scale and the board's remaining relative error.
    """
    true = pose.plane_depth(pixels.astype(np.float64))
    predicted = depth_cm[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    scale = float(np.median(true / predicted))
    return ScaleFit(scale=scale, board_error=float(np.mean(np.abs(scale * predicted - true) / true)))


def scaled_depth(photo: np.ndarray, setup: DepthSetup, model: MetricDepth) -> ScaledDepth:
    """Undistort a photo, find its board, and return board-scaled metric depth.

    Args:
        photo: H x W x 3 ``np.ndarray`` with ``dtype=uint8``, BGR, decoded in
            the sensor frame at the calibrated resolution.
        setup: Camera, board, detection settings and limits.
        model: The depth model.

    Returns:
        The corrected depth with the pose and fit behind it.

    Raises:
        ValueError: the photo is at another resolution, the board is not
            found, or the board's pose or fit exceeds the limits.
    """
    image = setup.calibration.undistort(photo)
    centres = find_grid(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), setup.board, setup.detection)
    if centres is None:
        raise ValueError("Board not found; without it the depth has no trustworthy scale")

    limits = setup.limits
    camera_matrix = setup.calibration.camera_matrix
    pose = solve_board_pose(centres, setup.board, camera_matrix)
    if pose.reprojection_px > limits.max_pose_error_px:
        raise ValueError(f"Board pose error {pose.reprojection_px:.2f} px exceeds {limits.max_pose_error_px} px")

    raw = model.predict(image, camera_matrix)
    fit = fit_scale(raw, pose, board_pixels(centres, raw.shape))
    log.info(
        "Scale %.3f, board error %.2f%%, pose error %.2f px", fit.scale, fit.board_error * 100, pose.reprojection_px
    )
    if fit.board_error > limits.max_board_error:
        raise ValueError(f"Board depth error {fit.board_error:.1%} exceeds {limits.max_board_error:.1%}")

    return ScaledDepth(
        image=image, depth_cm=raw * fit.scale, camera_matrix=camera_matrix, centres=centres, pose=pose, fit=fit
    )
