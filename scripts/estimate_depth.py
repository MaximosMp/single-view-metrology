#!/usr/bin/env python3
"""Estimate board-scaled metric depth for photos containing the circle-grid board.

Reads configs/depth.yaml, undistorts each photo with the camera calibration,
finds the board, runs the UniDepth metric depth model and rescales its depth so
the board lands where its pose says it is. Writes the depth map in cm and a
check image per photo. A photo whose board is missing or does not fit fails
rather than producing an unscaled map.

The first run downloads the model weights (about 1.4 GB) from Hugging Face.

Usage:
    uv run python scripts/estimate_depth.py images/demo/IMG_0001.HEIC
    uv run python scripts/estimate_depth.py images/demo/*.HEIC -o results/depth -v
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from svmetrology.depth import DepthSetup, MetricDepth, scaled_depth
from svmetrology.imaging import decode_image
from svmetrology.reporting import check_output_dir, configure_logging, log_banner, save_depth_overlay

log = logging.getLogger("depth")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "depth.yaml"


def main(
    photos: Annotated[list[Path], typer.Argument(help="Photos showing the board")],
    config: Annotated[Path, typer.Option(help="Depth config")] = DEFAULT_CONFIG,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Results folder; overrides the config")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose logging")] = False,
) -> None:
    """Estimate board-scaled metric depth for each photo."""
    configure_logging(verbose)
    try:
        setup = DepthSetup.load(config.expanduser().resolve())
        output_dir = output.resolve() if output else setup.output
        check_output_dir(output_dir, "Results folder")
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    model = MetricDepth(setup.process_res)
    failed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="depth_") as scratch:
        for photo_path in photos:
            # One bad photo must not hide the rest: record it, keep going, and
            # exit non-zero at the end.
            try:
                _, photo = decode_image(photo_path, Path(scratch))
                result = scaled_depth(photo, setup, model)
                np.save(output_dir / f"{photo_path.stem}_depth.npy", result.depth_cm)
                save_depth_overlay(
                    output_dir / f"{photo_path.stem}_depth.png",
                    result.depth_cm,
                    result.centres,
                    f"{photo_path.name}: scale {result.fit.scale:.3f}, board error {result.fit.board_error:.1%}",
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                log.error("%-24s %s", photo_path.name, exc)
                failed.append(photo_path.name)

    log_banner(log, "DEPTH")
    log.info("  Estimated : %d", len(photos) - len(failed))
    log.info("  Failed    : %d", len(failed))
    print(f"\n  -> {output_dir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
