"""Pixel extent of a binary mask: the longest distance across it.

Pure pixel geometry. Nothing here opens the camera intrinsics or converts to
physical units — that is the metric stage's job, and keeping this module
ignorant of it means the extent can be verified on its own.

The longest distance across a set equals the longest distance across its convex
hull, so every function here works on hull vertices rather than on the mask's
foreground pixels. That is not only faster, it is what makes an exhaustive
pairwise check affordable as a correctness oracle.

Coordinates are pixel centres, in ``(x, y)`` with y increasing downward, which
is the convention the camera intrinsics act on. A one-pixel mask therefore has
an extent of 0.0 px, not 1.0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskExtent:
    """The longest chord across a mask, with the geometry it was derived from.

    Attributes:
        distance_px: Length of the longest chord, in pixels.
        endpoints: ``((x0, y0), (x1, y1))`` pixel centres realising that chord.
        hull: h x 2 ``np.ndarray`` with ``dtype=int64``, ``(x, y)`` vertices of
            the convex hull, counter-clockwise and strictly convex. Excluded
            from equality and hashing, because comparing arrays yields an array
            rather than a bool and would make ``==`` raise.
        area_px: Count of foreground pixels.
        component_count: Number of 8-connected foreground components.
    """

    distance_px: float
    endpoints: tuple[tuple[int, int], tuple[int, int]]
    hull: np.ndarray = field(compare=False)
    area_px: int
    component_count: int


# -- Helpers ------------------------------------------------------------------


def _as_binary(mask: np.ndarray) -> np.ndarray:
    """Reduce a mask to 0/1 ``uint8``.

    Args:
        mask: H x W ``np.ndarray``, any dtype. Nonzero is foreground.

    Returns:
        H x W ``np.ndarray`` with ``dtype=uint8`` holding only 0 and 1.
    """
    return (mask > 0).astype(np.uint8)


def _cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    """Return the z component of ``(a - origin) x (b - origin)``.

    Evaluated in Python integers so the sign is exact: pixel coordinates are
    integral, and an epsilon comparison here would make the caliper walk
    dependent on floating-point noise.

    Args:
        origin: Length-2 integer array, the shared vertex.
        a: Length-2 integer array.
        b: Length-2 integer array.

    Returns:
        Twice the signed area of the triangle, positive when ``origin -> a -> b``
        turns counter-clockwise.
    """
    return int(a[0] - origin[0]) * int(b[1] - origin[1]) - int(a[1] - origin[1]) * int(b[0] - origin[0])


def _sq_dist(a: np.ndarray, b: np.ndarray) -> int:
    """Return the exact squared distance between two integer points.

    Squared distances are compared throughout so the square root is taken once,
    at the end, and never participates in a tie-break.

    Args:
        a: Length-2 integer array.
        b: Length-2 integer array.

    Returns:
        The squared Euclidean distance, as a Python integer.
    """
    return (int(a[0]) - int(b[0])) ** 2 + (int(a[1]) - int(b[1])) ** 2


def _extreme_pair(points: np.ndarray) -> np.ndarray:
    """Return the two furthest-apart points of a collinear set.

    Args:
        points: n x 2 integer ``np.ndarray``, all on one line, n >= 1.

    Returns:
        2 x 2 ``np.ndarray`` with ``dtype=int64``, or 1 x 2 when every point
        coincides.
    """
    order = np.lexsort((points[:, 1], points[:, 0]))
    first, last = points[order[0]], points[order[-1]]
    if np.array_equal(first, last):
        return points[order[:1]]
    return np.vstack([first, last]).astype(np.int64)


def _strictly_convex(hull: np.ndarray) -> np.ndarray:
    """Normalise a hull into the form `hull_diameter` requires.

    Counter-clockwise, with duplicate and collinear vertices removed. OpenCV
    happens to supply this today, but the guarantee is undocumented and has
    changed across majors, so this module owns it rather than trusting it.

    Args:
        hull: h x 2 integer ``np.ndarray`` of ``(x, y)`` vertices in order.

    Returns:
        h' x 2 ``np.ndarray`` with ``dtype=int64``, counter-clockwise and
        strictly convex. Degenerate input yields its 1 or 2 extreme vertices.
    """
    hull = hull.astype(np.int64)
    # Drop consecutive duplicates first. The turn test below is zero for a
    # repeated vertex on both copies, so filtering on it alone would delete the
    # vertex outright rather than deduplicate it.
    distinct = np.any(hull != np.roll(hull, 1, axis=0), axis=1)
    hull = hull[distinct] if distinct.any() else hull[:1]
    if len(hull) < 3:
        return hull

    x, y = hull[:, 0], hull[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        hull = hull[::-1]

    incoming = hull - np.roll(hull, 1, axis=0)
    outgoing = np.roll(hull, -1, axis=0) - hull
    turns = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    if not (turns != 0).any():
        # Every vertex is collinear: the extent is the span of the line, so
        # return its endpoints. Collapsing to one vertex would report 0.0 px.
        return _extreme_pair(hull)
    return hull[turns != 0]


def _hull_of_binary(binary: np.ndarray) -> np.ndarray:
    """Return the convex hull of an already-binarised mask.

    Args:
        binary: H x W ``np.ndarray`` with ``dtype=uint8`` holding 0 and 1.

    Returns:
        h x 2 ``np.ndarray`` with ``dtype=int64``, ``(x, y)`` pixel centres.

    Raises:
        ValueError: the mask has no foreground pixels.
    """
    # Contours rather than every foreground pixel: the hull of a region is the
    # hull of its boundary, and this hands convexHull a few thousand points
    # instead of a few million. RETR_EXTERNAL still covers disconnected blobs.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("mask has no foreground pixels")

    points = np.vstack(contours).reshape(-1, 1, 2).astype(np.int32)
    return _strictly_convex(cv2.convexHull(points).reshape(-1, 2))


# -- Diameter -----------------------------------------------------------------


def hull_diameter(hull: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Return the longest distance between two hull vertices, by rotating calipers.

    Walks one caliper along each edge while advancing the opposite caliper to the
    furthest vertex, which visits only antipodal pairs and so costs O(h) rather
    than O(h^2). When an opposite edge is parallel the walk plateaus, and both of
    its endpoints are antipodal — hence the four candidate checks per edge.

    Requires the `MaskExtent.hull` contract: counter-clockwise and strictly convex. A
    hull with duplicate or collinear vertices can make the walk terminate early
    and under-report.

    Args:
        hull: h x 2 integer ``np.ndarray``, counter-clockwise and strictly
            convex, as stored in `MaskExtent.hull`.

    Returns:
        ``(distance_px, (i, j))`` where i and j index ``hull``.
    """
    h = len(hull)
    if h == 1:
        return 0.0, (0, 0)
    if h == 2:
        return math.sqrt(_sq_dist(hull[0], hull[1])), (0, 1)

    best = -1
    pair = (0, 0)
    j = 1
    for i in range(h):
        nxt = (i + 1) % h
        # Strict '>' is what terminates the advance on a plateau of equally
        # distant vertices; '>=' would spin around the hull forever.
        while _cross(hull[i], hull[nxt], hull[(j + 1) % h]) > _cross(hull[i], hull[nxt], hull[j]):
            j = (j + 1) % h
        for a in (i, nxt):
            for b in (j, (j + 1) % h):
                d = _sq_dist(hull[a], hull[b])
                if d > best:
                    best, pair = d, (a, b)

    return math.sqrt(best), pair


def mask_extent(mask: np.ndarray) -> MaskExtent:
    """Measure the longest distance across a mask.

    Args:
        mask: H x W ``np.ndarray``, any dtype. Nonzero is foreground.

    Returns:
        The populated `MaskExtent`.

    Raises:
        ValueError: the mask has no foreground pixels.
    """
    binary = _as_binary(mask)
    hull = _hull_of_binary(binary)
    distance, (i, j) = hull_diameter(hull)
    label_count, _ = cv2.connectedComponents(binary, connectivity=8)

    return MaskExtent(
        distance_px=distance,
        endpoints=((int(hull[i][0]), int(hull[i][1])), (int(hull[j][0]), int(hull[j][1]))),
        hull=hull,
        area_px=int(cv2.countNonZero(binary)),
        component_count=label_count - 1,
    )
