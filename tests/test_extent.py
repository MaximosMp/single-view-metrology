"""Tests for mask extent: convex hull, rotating calipers, and the report."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from svmetrology.extent import (
    _strictly_convex,
    hull_diameter,
    mask_extent,
)


def _mask_from_points(points: np.ndarray, size: int) -> np.ndarray:
    """Rasterise integer points into a mask of ``size`` x ``size``."""
    mask = np.zeros((size, size), np.uint8)
    mask[points[:, 1], points[:, 0]] = 1
    return mask


def _pairwise_diameter(points: np.ndarray) -> float:
    """Return the longest distance between any two points, by exhaustive search.

    The oracle the calipers are checked against. It has no precondition on point
    order, duplicates or collinear runs, which is what makes it a fair judge.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    deltas = p[:, None, :] - p[None, :, :]
    return float(np.hypot(deltas[..., 0], deltas[..., 1]).max())


# -- Analytic cases -----------------------------------------------------------


def test_diameter_of_a_rectangle_is_its_diagonal():
    mask = np.zeros((200, 300), np.uint8)
    mask[10:110, 20:220] = 1  # rows 10..109, cols 20..219
    # Pixel centres, so the span is one less than the pixel count on each axis.
    assert mask_extent(mask).distance_px == pytest.approx(math.hypot(199, 99))


def test_diameter_of_a_disc_is_twice_the_radius():
    radius, centre = 60, 80
    yy, xx = np.mgrid[0:161, 0:161]
    # Built by inequality rather than cv2.circle: this guarantees (centre +- r, centre)
    # are included and nothing exceeds r, so the answer is exactly 2r. cv2.circle's
    # rasteriser is an implementation detail and would make this a tolerance test.
    mask = ((xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2).astype(np.uint8)
    assert mask_extent(mask).distance_px == pytest.approx(2 * radius)


def test_diameter_of_a_single_pixel_is_zero():
    mask = np.zeros((10, 10), np.uint8)
    mask[5, 5] = 1
    # Pixel-centre convention: one pixel has no extent. Documented so it is not "fixed".
    assert mask_extent(mask).distance_px == 0.0


def test_diameter_of_two_pixels():
    mask = np.zeros((10, 10), np.uint8)
    mask[0, 0] = 1
    mask[3, 0] = 1
    assert mask_extent(mask).distance_px == pytest.approx(3.0)


def test_diameter_of_a_single_row_mask():
    mask = np.zeros((20, 100), np.uint8)
    mask[7, 5:96] = 1  # cols 5..95
    assert mask_extent(mask).distance_px == pytest.approx(90.0)


def test_diameter_of_a_diagonal_line():
    mask = np.zeros((60, 60), np.uint8)
    idx = np.arange(50)
    mask[idx, idx] = 1
    assert mask_extent(mask).distance_px == pytest.approx(math.hypot(49, 49))


def test_empty_mask_raises():
    with pytest.raises(ValueError, match="no foreground"):
        mask_extent(np.zeros((10, 10), np.uint8))


def test_diameter_spans_two_disconnected_blobs():
    mask = np.zeros((200, 200), np.uint8)
    mask[10:20, 10:20] = 1
    mask[150:160, 150:160] = 1
    extent = mask_extent(mask)
    # The union's span: far corner of one blob to far corner of the other.
    assert extent.distance_px == pytest.approx(math.hypot(149, 149))
    assert extent.component_count == 2


def test_extent_is_invariant_under_rotation():
    mask = np.zeros((120, 200), np.uint8)
    mask[20:90, 30:180] = 1
    straight = mask_extent(mask).distance_px
    turned = mask_extent(cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)).distance_px
    assert straight == pytest.approx(turned)


def test_extent_accepts_bool_uint8_and_0_255_masks():
    base = np.zeros((80, 80), np.uint8)
    base[10:40, 20:70] = 1
    answers = {
        mask_extent(base.astype(bool)).distance_px,
        mask_extent(base).distance_px,
        mask_extent(base * 255).distance_px,
    }
    assert len(answers) == 1


def test_endpoints_are_foreground_pixels():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:60, 20:80] = 1
    extent = mask_extent(mask)
    for x, y in extent.endpoints:
        assert mask[y, x] == 1


def test_area_counts_foreground_pixels():
    mask = np.zeros((50, 50), np.uint8)
    mask[5:15, 5:25] = 1
    assert mask_extent(mask).area_px == 10 * 20


# -- Calipers against the exhaustive oracle -----------------------------------


def test_calipers_matches_brute_force_on_random_clouds():
    rng = np.random.default_rng(0)
    for k in range(300):
        # Small spans force dense duplicates and collinear runs; the large span
        # exercises ordinary hulls. Both are where a caliper walk goes wrong.
        span = (3, 10, 500)[k % 3]
        count = int(rng.integers(1, 61))
        points = rng.integers(0, span + 1, size=(count, 2))
        hull = mask_extent(_mask_from_points(points, span + 1)).hull
        assert hull_diameter(hull)[0] == pytest.approx(_pairwise_diameter(points))


def test_calipers_matches_brute_force_on_regular_polygons():
    radius = 300
    for sides in range(3, 25):
        angles = np.arange(sides) * 2 * math.pi / sides
        points = np.stack(
            [np.round(radius + radius * np.cos(angles)), np.round(radius + radius * np.sin(angles))],
            axis=1,
        ).astype(np.int64)
        hull = mask_extent(_mask_from_points(points, 2 * radius + 1)).hull
        measured = hull_diameter(hull)[0]
        assert measured == pytest.approx(_pairwise_diameter(points))
        # Even-sided polygons have parallel opposite edges — the caliper plateau.
        analytic = 2 * radius if sides % 2 == 0 else 2 * radius * math.cos(math.pi / (2 * sides))
        assert measured == pytest.approx(analytic, abs=2.0)


# -- Hull normalisation -------------------------------------------------------


def test_hull_is_counter_clockwise():
    rng = np.random.default_rng(1)
    points = rng.integers(0, 200, size=(40, 2))
    hull = mask_extent(_mask_from_points(points, 200)).hull
    x, y = hull[:, 0], hull[:, 1]
    assert np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def test_hull_drops_collinear_vertices():
    mask = np.zeros((60, 60), np.uint8)
    for x, y in [(10, 10), (50, 10), (50, 50), (10, 50), (30, 10), (50, 30), (30, 50), (10, 30)]:
        mask[y, x] = 1
    # The four edge midpoints lie on the square's sides and must not survive.
    assert len(mask_extent(mask).hull) == 4


def test_hull_of_a_single_pixel_has_one_vertex():
    mask = np.zeros((10, 10), np.uint8)
    mask[4, 6] = 1
    hull = mask_extent(mask).hull
    assert hull.shape == (1, 2)
    assert tuple(hull[0]) == (6, 4)


# -- Report -------------------------------------------------------------------


def test_measures_the_object_in_a_real_mask(ruler_mask: np.ndarray):
    extent = mask_extent(ruler_mask)
    # Pinned from an actual run on the committed fixture, and deterministic, so
    # the tolerance is float noise rather than a band: a loose one would let a
    # pixel-centre or antipodal-pair regression through unnoticed.
    assert extent.distance_px == pytest.approx(804.2742, abs=1e-3)
    assert extent.component_count == 1
    for x, y in extent.endpoints:
        assert ruler_mask[y, x]


def test_real_mask_extent_matches_the_oracle(ruler_mask: np.ndarray):
    hull = mask_extent(ruler_mask).hull
    assert hull_diameter(hull)[0] == pytest.approx(_pairwise_diameter(hull))


# -- Hull normalisation, fed directly ------------------------------------------
#
# OpenCV currently returns hulls that already satisfy hull_diameter's contract,
# so normalisation is never exercised through mask_extent. These feed it the
# degenerate shapes a future OpenCV could return, which is the whole reason the
# normalisation exists.


def test_collinear_hull_keeps_its_endpoints():
    # Collapsing to one vertex here would report a line as having no extent.
    hull = _strictly_convex(np.array([[0, 0], [5, 5], [10, 10]]))
    assert len(hull) == 2
    assert hull_diameter(hull)[0] == pytest.approx(math.hypot(10, 10))


def test_collinear_horizontal_hull_keeps_its_endpoints():
    hull = _strictly_convex(np.array([[1, 2], [4, 2], [8, 2]]))
    assert hull_diameter(hull)[0] == pytest.approx(7.0)


def test_duplicate_vertices_are_deduplicated_not_deleted():
    # A repeated vertex has a zero turn on both copies, so filtering on the turn
    # alone would drop the vertex entirely and shrink the hull.
    hull = _strictly_convex(np.array([[0, 0], [10, 0], [10, 0], [10, 10], [0, 10]]))
    assert len(hull) == 4
    assert hull_diameter(hull)[0] == pytest.approx(math.hypot(10, 10))


def test_a_closed_ring_loses_only_its_repeated_vertex():
    hull = _strictly_convex(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]))
    assert len(hull) == 4


def test_a_hull_of_identical_points_has_no_extent():
    hull = _strictly_convex(np.array([[7, 7], [7, 7], [7, 7]]))
    assert hull_diameter(hull)[0] == 0.0


def test_clockwise_input_is_reoriented():
    counter = _strictly_convex(np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    clockwise = _strictly_convex(np.array([[0, 10], [10, 10], [10, 0], [0, 0]]))
    assert hull_diameter(counter)[0] == pytest.approx(hull_diameter(clockwise)[0])


def test_normalisation_matches_the_oracle_on_degenerate_hulls():
    rng = np.random.default_rng(7)
    for _ in range(200):
        # Points along a line, with duplicates, in random order: every shape the
        # normalisation is supposed to absorb.
        count = int(rng.integers(1, 12))
        steps = rng.integers(0, 4, size=count)
        points = np.stack([np.cumsum(steps) * 3, np.cumsum(steps) * 7], axis=1)
        hull = _strictly_convex(points)
        assert hull_diameter(hull)[0] == pytest.approx(_pairwise_diameter(points))


def test_extent_is_comparable_and_survives_equality():
    mask = np.zeros((40, 40), np.uint8)
    mask[5:20, 5:30] = 1
    # A frozen dataclass holding an ndarray raises on == unless the array is
    # excluded from comparison.
    assert mask_extent(mask) == mask_extent(mask.copy())
