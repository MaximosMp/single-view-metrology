"""Finding the circle grid in a photograph.

OpenCV finds the grid among the blobs its detector hands over, and that
detector has to be told how big a blob may be. A dot's size in pixels depends
on how far away the board is, which is not known beforehand. So the search
walks a short ladder of size guesses, each a fraction of the image's long edge,
and stops at the first one where the whole grid is found.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from svmetrology.board import CircleGrid


@dataclass
class DetectionSettings:
    """The dot sizes `find_grid` tries.

    Attributes:
        dot_sizes: Dot diameters to try, as fractions of the image's long edge,
            in the order tried.
        size_tolerance: Each try accepts dots from this factor below its
            diameter to this factor above. It must be at least the square root
            of the step between neighbouring sizes, or the tries leave gaps.
    """

    dot_sizes: list[float]
    size_tolerance: float


def find_grid(gray: np.ndarray, board: CircleGrid, settings: DetectionSettings) -> np.ndarray | None:
    """Locate every dot of the board in one image.

    Args:
        gray: H x W ``np.ndarray`` with ``dtype=uint8``.
        board: The board being looked for.
        settings: The dot sizes to try.

    Returns:
        ``(cols * rows) x 2`` ``np.ndarray`` with ``dtype=float32``: sub-pixel
        ``(x, y)`` dot centres in `CircleGrid.object_points` order, or ``None``
        when the grid is not found.
    """
    long_edge = max(gray.shape)
    for fraction in settings.dot_sizes:
        diameter = fraction * long_edge
        # OpenCV's own blob settings, dark blobs included; only the area is ours.
        params = cv2.SimpleBlobDetector.Params()
        params.minArea = np.pi * (diameter / settings.size_tolerance / 2) ** 2
        params.maxArea = np.pi * (diameter * settings.size_tolerance / 2) ** 2
        # The clustering search copes with a board seen at an angle; the plain
        # one loses the grid on most tilted shots.
        found, centres = cv2.findCirclesGrid(
            gray,
            board.shape,
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
            blobDetector=cv2.SimpleBlobDetector.create(params),
        )
        if found:
            # OpenCV majors disagree on N x 1 x 2 versus N x 2; settle it here.
            return centres.reshape(-1, 2)
    return None
