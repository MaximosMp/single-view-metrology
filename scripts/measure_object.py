#!/usr/bin/env python3
"""Measure masked objects in cm, in photos that also show the circle-grid board.

The pipeline's entry point. Takes a folder holding photos and their masks,
pairs them by filename, and for each pair: undistorts the photo, finds the
board, estimates depth and rescales it to the board, then reports the masked
object's length and width (the sides of the smallest rectangle around it) and
the longest real-world distance across it. A pair whose board is missing or
does not fit fails rather than reporting an unscaled size.

Masks must match their photo exactly; scripts/prepare_mask.py exists to make
one that does. Photos must come from the calibrated camera at its native
resolution.

Usage:
    uv run python scripts/measure_object.py images/demo
    uv run python scripts/measure_object.py images/demo --overlay-dir results/overlays -v
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer

from svmetrology.config import load_yaml_as, resolve_path
from svmetrology.depth import DepthSetup, MetricDepth, scaled_depth
from svmetrology.imaging import find_image_mask_pairs, load_pair
from svmetrology.measure import MeasureConfig, measure, save_measurements
from svmetrology.reporting import (
    check_output_dir,
    check_output_path,
    configure_logging,
    log_banner,
    save_extent_overlay,
)

log = logging.getLogger("measure")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "measure.yaml"


def main(
    folder: Annotated[Path, typer.Argument(help="Folder holding images and their <name>_mask files")],
    config: Annotated[Path, typer.Option(help="Measurement config")] = DEFAULT_CONFIG,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="YAML report; overrides the config")] = None,
    overlay_dir: Annotated[Path | None, typer.Option(help="Write a check image per pair into this folder")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose logging")] = False,
) -> None:
    """Measure the longest real-world distance across each masked object."""
    configure_logging(verbose)
    config_path = config.expanduser().resolve()

    # Every input and destination is checked before the model loads: finding
    # an unwritable path after measuring a folder would throw the results away.
    try:
        cfg = load_yaml_as(config_path, MeasureConfig)
        setup = DepthSetup.load(resolve_path(cfg.depth_config, config_path))
        output_path = output.resolve() if output else resolve_path(cfg.output, config_path)
        check_output_path(output_path, "Report")
        if overlay_dir is not None:
            check_output_dir(overlay_dir, "Overlay directory")
        pairs, unpaired = find_image_mask_pairs(folder)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    for stray in unpaired:
        log.warning("%-28s has no counterpart - skipped", stray.name)

    model = MetricDepth(setup.process_res)
    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="measure_") as scratch:
        log.info("Measuring %d pair(s) from '%s'", len(pairs), folder)
        for image_path, mask_path in pairs:
            # One bad pair must not hide the rest: record it and keep going, then
            # fail at the end so a script or CI still sees a non-zero exit.
            try:
                bgr, mask = load_pair(image_path, mask_path, Path(scratch))
                depth = scaled_depth(bgr, setup, model)
                result = measure(mask, depth, setup.calibration, cfg.edge_margin_px)
                if overlay_dir is not None:
                    save_extent_overlay(
                        overlay_dir / f"{image_path.stem}_measure.png",
                        depth.image,
                        result.extent.hull,
                        result.rectangle,
                        f"{image_path.name}: {result.length_cm:.2f} x {result.width_cm:.2f} cm, "
                        f"longest {result.longest_cm:.2f} cm",
                    )
            except (FileNotFoundError, ValueError, OSError) as exc:
                log.error("%-28s %s", image_path.name, exc)
                failed.append({"image": image_path.name, "reason": str(exc)})
                continue

            extent = result.extent
            if extent.component_count > 1:
                log.warning(
                    "%-28s mask has %d disconnected components; the length spans all of them",
                    image_path.name,
                    extent.component_count,
                )
            log.info(
                "%-28s %.2f x %.2f cm, longest %.2f cm  (%.1f px across)",
                image_path.name,
                result.length_cm,
                result.width_cm,
                result.longest_cm,
                extent.distance_px,
            )
            results.append(
                {
                    "image": image_path.name,
                    "mask": mask_path.name,
                    "length_cm": round(result.length_cm, 3),
                    "width_cm": round(result.width_cm, 3),
                    "longest_distance_cm": round(result.longest_cm, 3),
                    "endpoints_px": [list(result.endpoints[0]), list(result.endpoints[1])],
                    "longest_distance_px": round(float(extent.distance_px), 4),
                    "mask_area_px": extent.area_px,
                    "component_count": extent.component_count,
                    "board": {
                        "scale": round(depth.fit.scale, 4),
                        "depth_error": round(depth.fit.board_error, 4),
                        "pose_error_px": round(depth.pose.reprojection_px, 3),
                    },
                }
            )

    log_banner(log, "MEASUREMENT")
    log.info("  Measured : %d", len(results))
    log.info("  Unpaired : %d", len(unpaired))
    log.info("  Failed   : %d", len(failed))

    # Written even when nothing could be measured, so a consumer reading the
    # report sees the same shape whatever the exit code.
    try:
        save_measurements(output_path, results, [p.name for p in unpaired], failed)
    except OSError as exc:
        log.error("%s", exc)
        sys.exit(1)
    print(f"\n  -> {output_path}")

    if failed or not results:
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
