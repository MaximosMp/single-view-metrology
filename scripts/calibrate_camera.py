#!/usr/bin/env python3
"""Solve the camera's intrinsics from photos of the printed circle-grid board.

Reads configs/calibration.yaml, finds the grid in every photo of the input
folder, fits the camera matrix and distortion while leaving out views that
disagree, and writes the result as YAML for the rest of the pipeline.

Usage:
    uv run python scripts/calibrate_camera.py
    uv run python scripts/calibrate_camera.py images/calibration_images --overlay-dir results/detections -v
    uv run python scripts/calibrate_camera.py -o camera_parameters/other_camera.yaml --max-long-edge 1920
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import cv2
import typer

from svmetrology.board import CircleGrid
from svmetrology.calibration import CalibrationConfig, View, calibrate
from svmetrology.config import load_yaml_as, resolve_path
from svmetrology.detection import find_grid
from svmetrology.imaging import load_folder, resize_longest_edge
from svmetrology.reporting import (
    check_output_dir,
    check_output_path,
    configure_logging,
    log_banner,
    save_detection_overlay,
)

log = logging.getLogger("calibrate")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "calibration.yaml"

# cv2.calibrateCamera needs at least this many views to solve at all.
MIN_VIEWS_TO_SOLVE = 3


def main(
    images: Annotated[Path | None, typer.Argument(help="Board photos; overrides the config")] = None,
    config: Annotated[Path, typer.Option(help="Calibration config")] = DEFAULT_CONFIG,
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Calibration YAML; overrides the config")
    ] = None,
    pitch_cm: Annotated[float | None, typer.Option(help="Board row pitch in cm; overrides the board file")] = None,
    max_long_edge: Annotated[int | None, typer.Option(help="Calibrate at this long edge instead of native")] = None,
    overlay_dir: Annotated[Path | None, typer.Option(help="Write a detection check image per photo here")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose logging")] = False,
) -> None:
    """Calibrate the camera from photos of the circle-grid board."""
    configure_logging(verbose)
    config_path = config.expanduser().resolve()

    # Paths typed on the command line resolve against the working directory;
    # paths read from the config resolve against the config file.
    try:
        cfg = load_yaml_as(config_path, CalibrationConfig)
        board = load_yaml_as(resolve_path(cfg.board, config_path), CircleGrid)
        image_dir = images.resolve() if images else resolve_path(cfg.images, config_path)
        output_path = output.resolve() if output else resolve_path(cfg.output, config_path)
        check_output_path(output_path, "Calibration file")
        if overlay_dir is not None:
            check_output_dir(overlay_dir, "Overlay directory")
        photos = load_folder(image_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    if pitch_cm is not None:
        board = replace(board, pitch_cm=pitch_cm)
    long_edge = max_long_edge or cfg.max_long_edge
    if long_edge:
        photos = [(name, resize_longest_edge(bgr, long_edge)) for name, bgr in photos]

    # One calibration describes one camera at one resolution, so photos of
    # another size cannot join the fit. Keep the most common size.
    image_size = Counter((bgr.shape[1], bgr.shape[0]) for _, bgr in photos).most_common(1)[0][0]
    log.info("Finding the %dx%d grid in %d photo(s) at %dx%d", *board.shape, len(photos), *image_size)

    views: list[View] = []
    for name, bgr in photos:
        if (bgr.shape[1], bgr.shape[0]) != image_size:
            log.warning("  %-24s %dx%d, not %dx%d - skipped", name, bgr.shape[1], bgr.shape[0], *image_size)
            continue
        centres = find_grid(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), board, cfg.detection)
        log.info("  %-24s %s", name, "found" if centres is not None else "NOT FOUND")
        if centres is not None:
            views.append(View(name, centres))
        if overlay_dir is not None:
            status = "grid found" if centres is not None else "grid not found"
            save_detection_overlay(
                overlay_dir / f"{Path(name).stem}_grid.png", bgr, centres, board.shape, f"{name}: {status}"
            )

    if len(views) < MIN_VIEWS_TO_SOLVE:
        log.error(
            "The grid was found in %d photo(s); calibration needs at least %d. "
            "Rerun with --overlay-dir to see which photos fail and why.",
            len(views),
            MIN_VIEWS_TO_SOLVE,
        )
        sys.exit(1)

    result = calibrate(views, board, image_size, cfg.rejection)

    k = result.camera_matrix
    log_banner(log, "CALIBRATION")
    log.info("  RMS reprojection error : %.4f px", result.rms_px)
    log.info("  fx, fy                 : %.1f, %.1f", k[0, 0], k[1, 1])
    log.info("  cx, cy                 : %.1f, %.1f", k[0, 2], k[1, 2])
    log.info("  k1 k2 p1 p2 k3         : %s", " ".join(f"{c:+.4f}" for c in result.distortion))
    log.info("  Views kept             : %d of %d", len(result.views_used), len(views))
    if result.rms_px > 2.0:
        log.warning("An RMS above 2 px usually means a bent board or motion blur; reshoot on a rigid board.")

    try:
        result.save(output_path)
    except OSError as exc:
        log.error("%s", exc)
        sys.exit(1)
    print(f"\n  -> {output_path}")


if __name__ == "__main__":
    typer.run(main)
