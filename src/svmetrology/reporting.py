"""Shared output: logging setup, YAML writing, and check images.

The CLIs each need to configure logging identically, print a result banner,
persist a YAML document, and draw an overlay. Keeping one copy here is what
stops the three scripts drifting into three slightly different dialects.

Overlays are drawn with OpenCV rather than matplotlib: a check image exists to
confirm that pixels line up, so it must be written on the pixel grid it is
checking, not resampled into a figure canvas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

BANNER_WIDTH = 55
OVERLAY_LONG_EDGE = 1200

#: BGR, since OpenCV draws in BGR order.
HULL_COLOUR = (255, 229, 0)
RECTANGLE_COLOUR = (68, 23, 255)
MASK_COLOUR = (80, 200, 255)


def configure_logging(verbose: bool) -> None:
    """Set up logging for a CLI.

    Args:
        verbose: Emit debug-level records.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )


def log_banner(logger: logging.Logger, title: str) -> None:
    """Log a section banner above a run's result summary.

    Args:
        logger: Logger to write to, so the record keeps the caller's name.
        title: Banner text.
    """
    logger.info("")
    logger.info("=" * BANNER_WIDTH)
    logger.info(title)
    logger.info("=" * BANNER_WIDTH)


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    """Write a document as YAML, creating parent directories.

    Every YAML this pipeline emits goes through here, so the files stay in one
    dialect and a consumer reading two of them does not meet two.

    Args:
        path: Destination file.
        document: Mapping of YAML-native values.

    Raises:
        OSError: the path is not writable, or is a directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(document, handle, default_flow_style=False, sort_keys=False)
    log.info("Written to %s", path)


def check_output_path(path: Path, kind: str) -> None:
    """Fail early if a run's output destination is unusable.

    Called before the work starts, so a bad ``--output`` costs a message rather
    than a full run whose results are then thrown away by an exception.

    Args:
        path: File or directory the run intends to write.
        kind: Word for the destination, used in the error message.

    Raises:
        ValueError: the path exists as the wrong type, or its parent does not
            exist and cannot be created.
    """
    if path.is_dir():
        raise ValueError(f"{kind} '{path}' is a directory; give a file path.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"{kind} '{path}' is not writable: {exc}") from exc


def check_output_dir(path: Path, kind: str) -> None:
    """Fail early if a run's output directory is unusable.

    Args:
        path: Directory the run intends to write into.
        kind: Word for the destination, used in the error message.

    Raises:
        ValueError: the path exists as a file, or cannot be created.
    """
    if path.exists() and not path.is_dir():
        raise ValueError(f"{kind} '{path}' exists and is not a directory.")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"{kind} '{path}' cannot be created: {exc}") from exc


def _fit_for_overlay(bgr: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    """Downscale an image for an overlay and report the scale applied.

    Args:
        bgr: H x W x 3 ``np.ndarray`` with ``dtype=uint8``.
        long_edge: Maximum length of the result's longest edge, in pixels.

    Returns:
        ``(image, scale)`` where scale maps original coordinates onto the result.
    """
    height, width = bgr.shape[:2]
    scale = min(1.0, long_edge / max(height, width))
    if scale >= 1.0:
        return bgr.copy(), 1.0
    return cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA), scale


def save_mask_overlay(path: Path, bgr: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Save a check image of a mask tinted over its photo.

    Only foreground pixels are tinted. A uniform wash over the whole frame would
    hide the object the reader is trying to confirm the mask sits on.

    Args:
        path: Destination PNG.
        bgr: H x W x 3 ``np.ndarray`` with ``dtype=uint8``.
        mask: H x W ``np.ndarray`` with ``dtype=bool``, same size as ``bgr``.
        title: Caption drawn at the top left.

    Raises:
        OSError: the image could not be written.
    """
    canvas, _scale = _fit_for_overlay(bgr, OVERLAY_LONG_EDGE)
    small = cv2.resize(
        mask.astype(np.uint8), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    tint = np.zeros_like(canvas)
    tint[:] = MASK_COLOUR
    canvas[small] = cv2.addWeighted(canvas, 0.55, tint, 0.45, 0.0)[small]
    _caption(canvas, title)
    _write_png(path, canvas)


def save_extent_overlay(
    path: Path,
    bgr: np.ndarray,
    hull: np.ndarray,
    rectangle: np.ndarray,
    title: str,
) -> None:
    """Save a check image of a hull and the rectangle measured around it.

    Args:
        path: Destination PNG.
        bgr: H x W x 3 ``np.ndarray`` with ``dtype=uint8``.
        hull: h x 2 integer ``np.ndarray`` of ``(x, y)`` vertices, full resolution.
        rectangle: 4 x 2 ``np.ndarray`` of ``(x, y)`` corners, full resolution.
        title: Caption drawn at the top left.

    Raises:
        OSError: the image could not be written.
    """
    canvas, scale = _fit_for_overlay(bgr, OVERLAY_LONG_EDGE)
    scaled_hull = np.round(hull * scale).astype(np.int32)
    corners = np.round(rectangle * scale).astype(np.int32)

    cv2.polylines(canvas, [scaled_hull], isClosed=True, color=HULL_COLOUR, thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [corners], isClosed=True, color=RECTANGLE_COLOUR, thickness=3, lineType=cv2.LINE_AA)
    for corner in corners:
        cv2.circle(canvas, tuple(int(c) for c in corner), 7, RECTANGLE_COLOUR, thickness=-1, lineType=cv2.LINE_AA)
    _caption(canvas, title)
    _write_png(path, canvas)


def save_detection_overlay(
    path: Path,
    bgr: np.ndarray,
    centres: np.ndarray | None,
    grid_shape: tuple[int, int],
    title: str,
) -> None:
    """Save a check image of a detected grid over its photo.

    OpenCV's own drawing joins the dots in detection order, so a grid found
    with its rows scrambled shows up as a tangle rather than a clean zigzag.

    Args:
        path: Destination PNG.
        bgr: H x W x 3 ``np.ndarray`` with ``dtype=uint8``.
        centres: N x 2 ``np.ndarray`` of dot centres at full resolution, or
            ``None`` when detection failed and only the caption is drawn.
        grid_shape: ``(cols, rows)`` of the grid.
        title: Caption drawn at the top left.

    Raises:
        OSError: the image could not be written.
    """
    canvas, scale = _fit_for_overlay(bgr, OVERLAY_LONG_EDGE)
    if centres is not None:
        cv2.drawChessboardCorners(canvas, grid_shape, (centres * scale).astype(np.float32), True)
    _caption(canvas, title)
    _write_png(path, canvas)


def save_depth_overlay(path: Path, depth_cm: np.ndarray, centres: np.ndarray, title: str) -> None:
    """Save a check image of a depth map with the board outlined.

    Colours span the 2nd to 98th percentile, so a few wild pixels at an edge do
    not flatten the rest into one colour.

    Args:
        path: Destination PNG.
        depth_cm: H x W ``np.ndarray``, depth in cm.
        centres: N x 2 ``np.ndarray`` of the board's dot centres at full resolution.
        title: Caption drawn at the top left.

    Raises:
        OSError: the image could not be written.
    """
    near, far = np.percentile(depth_cm, (2, 98))
    levels = np.clip((depth_cm - near) / max(far - near, 1e-6) * 255, 0, 255).astype(np.uint8)
    canvas, scale = _fit_for_overlay(cv2.applyColorMap(levels, cv2.COLORMAP_TURBO), OVERLAY_LONG_EDGE)
    hull = cv2.convexHull((centres * scale).astype(np.float32)).astype(np.int32)
    cv2.polylines(canvas, [hull], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    _caption(canvas, f"{title}  ({near:.1f}-{far:.1f} cm)")
    _write_png(path, canvas)


def _caption(canvas: np.ndarray, text: str) -> None:
    """Draw a caption on a solid bar so it reads on any background.

    Args:
        canvas: H x W x 3 ``np.ndarray`` with ``dtype=uint8``, modified in place.
        text: Caption text.
    """
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    # A solid bar rather than a stroked outline: at this font size an outline
    # thick enough to read collides with its own glyphs.
    cv2.rectangle(canvas, (0, 0), (width + 20, height + baseline + 16), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, text, (10, height + 8), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _write_png(path: Path, canvas: np.ndarray) -> None:
    """Write an overlay image, failing loudly when the write does not happen.

    Args:
        path: Destination PNG.
        canvas: H x W x 3 ``np.ndarray`` with ``dtype=uint8``.

    Raises:
        OSError: ``cv2.imwrite`` reported failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 signals an unknown extension by raising and an unwritable path by
    # returning False; both mean the file is not there, so both become OSError.
    try:
        written = cv2.imwrite(str(path), canvas)
    except cv2.error as exc:
        raise OSError(f"Could not write overlay '{path}': {exc.err or exc}".strip()) from exc
    if not written:
        raise OSError(f"Could not write overlay '{path}'. Check the path and its extension.")
    log.info("Overlay written to %s", path)
