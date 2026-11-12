"""Real-world size of a masked object, from board-scaled depth.

The longest distance across a flat surface, in 3D, runs between two corners of
its outline's convex hull even when the surface is tilted: perspective maps a
plane's convex hull onto the image's convex hull. So the pixel hull from
`extent.mask_extent` is back-projected with the depth at each corner, and the
longest chord between the corners is the longest distance.

That chord is a diagonal for anything box-shaped, which is not what a ruler
held against the object reads. So the same corners also give a length and a
width: the sides of the smallest rectangle around them in the plane they lie in.

Depth right at an outline is unreliable: the model blurs the object into
whatever lies behind it, and for a raised object that is the table, which
would stretch the length. Each corner's depth is therefore extrapolated from
the object's interior, a margin inside the outline, rather than read off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from svmetrology.calibration import CameraCalibration
from svmetrology.depth import ScaledDepth
from svmetrology.extent import MaskExtent, mask_extent
from svmetrology.reporting import write_yaml


@dataclass
class MeasureConfig:
    """Settings for a measurement run.

    Attributes:
        depth_config: The depth config; the camera, board and model come from it.
        output: Where the YAML report is written.
        edge_margin_px: How far inside the outline, in px of the photo, depth
            starts to be trusted for extrapolating a hull corner's depth.
    """

    depth_config: str
    output: str
    edge_margin_px: int


@dataclass(frozen=True)
class Measurement:
    """An object's size and the geometry behind it.

    Attributes:
        longest_cm: Longest distance across the object, in cm.
        endpoints: ``((x0, y0), (x1, y1))`` hull corners realising it, in the
            undistorted image.
        length_cm: Long side of the smallest rectangle around the object, in cm.
        width_cm: Short side of that rectangle, in cm.
        rectangle: 4 x 2 ``np.ndarray``, that rectangle's corners as ``(x, y)``
            in the undistorted image.
        extent: Pixel extent of the undistorted mask, whose hull everything is
            measured over.
    """

    longest_cm: float
    endpoints: tuple[tuple[int, int], tuple[int, int]]
    length_cm: float
    width_cm: float
    rectangle: np.ndarray
    extent: MaskExtent


def back_project(pixels: np.ndarray, depth_cm: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Lift pixels to camera-frame points.

    Args:
        pixels: N x 2 ``np.ndarray`` of ``(u, v)`` in px, undistorted image.
        depth_cm: ``np.ndarray`` of shape (N,), depth along the optical axis.
        camera_matrix: 3 x 3 ``np.ndarray`` for the image.

    Returns:
        N x 3 ``np.ndarray`` of points in cm.
    """
    rays = np.linalg.solve(camera_matrix, np.column_stack([pixels, np.ones(len(pixels))]).T)
    return (rays * depth_cm).T


def corner_depths(corners: np.ndarray, mask: np.ndarray, depth_cm: np.ndarray, margin_px: int) -> np.ndarray:
    """Extrapolate each corner's depth from the object's interior nearby.

    Inverse depth over a plane is affine in pixel coordinates, so fitting
    ``1/z = a*u + b*v + c`` to the interior pixels near a corner and evaluating
    it there is exact for any flat surface, tilted or not. Averaging those
    pixels instead would report the interior's depth, which on a tilted surface
    is not the corner's.

    The window starts at twice the margin and doubles until it holds enough
    interior to fit, so a thin part of the object borrows the surface of the
    nearest part thick enough to have one.

    Args:
        corners: N x 2 ``np.ndarray`` of ``(x, y)`` on the mask's outline.
        mask: H x W ``np.ndarray`` with ``dtype=bool``.
        depth_cm: H x W ``np.ndarray``, depth in cm.
        margin_px: Distance inside the outline below which depth is not used.

    Returns:
        ``np.ndarray`` of shape (N,), depths in cm.

    Raises:
        ValueError: no part of the mask is wider than twice the margin.
    """
    size = 2 * margin_px + 1
    interior = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
    if interior.sum() < 3:
        raise ValueError(f"The mask is nowhere wider than twice the edge margin ({margin_px} px)")

    depths = []
    for x, y in corners:
        radius = max(2 * margin_px, 2)
        while True:
            top, left = max(y - radius, 0), max(x - radius, 0)
            v, u = np.nonzero(interior[top : y + radius + 1, left : x + radius + 1])
            if len(u) >= 3:
                break
            radius *= 2
        u, v = u + left, v + top
        design = np.column_stack([u, v, np.ones(len(u))]).astype(np.float64)
        coefficients = np.linalg.lstsq(design, 1.0 / depth_cm[v, u].astype(np.float64), rcond=None)[0]
        depths.append(1.0 / (coefficients @ (x, y, 1.0)))
    return np.array(depths)


def bounding_rectangle(points: np.ndarray) -> np.ndarray:
    """Return the smallest rectangle around points near a plane.

    The points are flattened onto their best-fit plane, spanned by the two
    leading right singular vectors of the centred points, and the rectangle is
    the minimum-area one around them there. For a box-shaped object that is its
    length and width; for any other flat outline, the box it fits in.

    A triangle has no such box: a rectangle standing on any of its three sides
    has the same area, twice the triangle's, so which one comes back is
    arbitrary. The longest distance is the measurement to use there.

    Args:
        points: N x 3 ``np.ndarray`` of points in cm, N at least 2.

    Returns:
        4 x 3 ``np.ndarray``, the rectangle's corners in cm, in order around it.
    """
    centre = points.mean(axis=0)
    in_plane = np.linalg.svd(points - centre, full_matrices=False)[2][:2]
    flat = ((points - centre) @ in_plane.T).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(flat)).astype(np.float64) @ in_plane + centre


def measure(mask: np.ndarray, depth: ScaledDepth, calibration: CameraCalibration, edge_margin_px: int) -> Measurement:
    """Measure a masked object: its longest distance, and its length and width.

    Args:
        mask: H x W ``np.ndarray`` with ``dtype=bool``, drawn on the photo as
            decoded, before undistortion.
        depth: The photo's board-scaled depth.
        calibration: The camera, to undistort the mask exactly as the photo was.
        edge_margin_px: Distance inside the outline below which depth is not used.

    Returns:
        The measurements and their geometry.

    Raises:
        ValueError: the mask is at another resolution, empty after
            undistortion, or too thin for the margin.
    """
    # Undistorting a 0/255 image bilinearly and cutting at mid-level keeps the
    # outline at sub-pixel accuracy, as `imaging.snap_mask_to_image` does.
    undistorted = calibration.undistort(mask.astype(np.uint8) * 255) > 127
    extent = mask_extent(undistorted)
    corners = extent.hull
    points = back_project(
        corners, corner_depths(corners, undistorted, depth.depth_cm, edge_margin_px), depth.camera_matrix
    )
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
    box = bounding_rectangle(points)
    width_cm, length_cm = sorted(np.linalg.norm(box[:2] - box[1:3], axis=1))  # two adjacent sides
    return Measurement(
        longest_cm=float(distances[i, j]),
        endpoints=((int(corners[i][0]), int(corners[i][1])), (int(corners[j][0]), int(corners[j][1]))),
        length_cm=float(length_cm),
        width_cm=float(width_cm),
        rectangle=(depth.camera_matrix @ (box / box[:, 2:]).T).T[:, :2],
        extent=extent,
    )


def save_measurements(
    path: Path,
    results: list[dict[str, Any]],
    unpaired: list[str],
    failed: list[dict[str, str]],
) -> None:
    """Write per-image measurements to a YAML report.

    Every length key carries its unit, and pixel coordinates are in the
    undistorted image, which the report states rather than leaving implied.

    Args:
        path: Destination YAML path; parent directories are created.
        results: One mapping per measured pair.
        unpaired: Filenames that had no counterpart.
        failed: Mappings of ``{"image": name, "reason": text}``.

    Raises:
        OSError: the path is not writable.
    """
    write_yaml(path, {"pixel_frame": "undistorted", "results": results, "unpaired": unpaired, "failed": failed})
