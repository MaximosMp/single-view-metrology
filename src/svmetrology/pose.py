"""The board's pose in a photo, and the depth of its plane at any pixel.

Everything here works in the undistorted image: the grid is found after
`cv2.undistort`, so the pose needs only the camera matrix, and the depth model
sees the same pixel grid. Lengths are in cm, the board's unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from svmetrology.board import CircleGrid
from svmetrology.calibration import view_rms

NO_DISTORTION = np.zeros(5)


@dataclass
class BoardPose:
    """Where the board sits relative to the camera.

    Attributes:
        rotation: 3 x 3 ``np.ndarray`` with ``dtype=float64``, board frame to
            camera frame.
        translation_cm: ``np.ndarray`` of shape (3,), the board origin in the
            camera frame, in cm.
        camera_matrix: 3 x 3 ``np.ndarray`` of the undistorted image the pose
            was solved in, in px.
        reprojection_px: RMS distance between the detected dot centres and the
            dots reprojected through the pose.
    """

    rotation: np.ndarray
    translation_cm: np.ndarray
    camera_matrix: np.ndarray
    reprojection_px: float

    def plane_depth(self, pixels: np.ndarray) -> np.ndarray:
        """Return the depth of the board's plane behind each pixel.

        Depth is the z coordinate along the optical axis, the quantity depth
        models predict, not the distance along the ray.

        Args:
            pixels: N x 2 ``np.ndarray`` of ``(u, v)`` in px, in the
                undistorted image. Rays parallel to the board, or meeting its
                plane behind the camera, give an infinite or negative depth.

        Returns:
            ``np.ndarray`` of shape (N,), depths in cm.
        """
        normal = self.rotation[:, 2]
        rays = np.linalg.solve(self.camera_matrix, np.column_stack([pixels, np.ones(len(pixels))]).T)
        return (normal @ self.translation_cm) / (normal @ rays)


def solve_board_pose(centres: np.ndarray, board: CircleGrid, camera_matrix: np.ndarray) -> BoardPose:
    """Solve the board's pose from its detected dots.

    Args:
        centres: N x 2 ``np.ndarray`` with ``dtype=float32``, dot centres in px
            in the undistorted image, in `CircleGrid.object_points` order.
        board: The board in the photo.
        camera_matrix: 3 x 3 ``np.ndarray`` of the undistorted image, at its
            resolution.

    Returns:
        The pose, with its reprojection error.

    Raises:
        ValueError: OpenCV found no pose for the dots.
    """
    object_points = board.object_points()
    # IPPE is the solver built for a planar target; the generic iterative one
    # can settle on the mirrored pose of a flat board.
    found, rvec, tvec = cv2.solvePnP(object_points, centres, camera_matrix, NO_DISTORTION, flags=cv2.SOLVEPNP_IPPE)
    if not found:
        raise ValueError("No board pose fits the detected dots")
    rotation, _ = cv2.Rodrigues(rvec)
    return BoardPose(
        rotation=rotation,
        translation_cm=tvec.ravel(),
        camera_matrix=camera_matrix,
        reprojection_px=view_rms(object_points, centres, rvec, tvec, camera_matrix, NO_DISTORTION),
    )
