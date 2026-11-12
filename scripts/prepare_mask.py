#!/usr/bin/env python3
"""Snap a mask onto an image's pixel grid.

Masking tools generally work on the display frame — EXIF rotation applied — and
often export at a reduced resolution. The pipeline reads the sensor frame at
native size and refuses a mask that does not match, so this converts one to the
other, once, into a file you can inspect.

The image is never modified. Only a new mask is written.

Usage:
    uv run python scripts/prepare_mask.py images/demo/IMG_2686.HEIC rough_mask.png
    uv run python scripts/prepare_mask.py photo.jpg rough.png -o photo_mask.png --rotate 90
"""

from __future__ import annotations

import enum
import logging
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import cv2
import typer

from svmetrology.imaging import (
    MASK_SUFFIX,
    decode_image,
    display_to_sensor_rotation,
    load_mask,
    save_mask,
    snap_mask_to_image,
)
from svmetrology.reporting import check_output_path, configure_logging, log_banner, save_mask_overlay

log = logging.getLogger("prepare-mask")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Outside the folder the pipeline scans, so a check image never shows up as an
# unpaired file on every later run.
DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "results" / "mask_checks"


class Rotation(enum.StrEnum):
    """Rotation to apply to the mask before rescaling."""

    AUTO = "auto"
    NONE = "0"
    CW90 = "90"
    HALF = "180"
    CCW90 = "270"


#: Maps the CLI vocabulary onto OpenCV's, and back for logging. The two differ:
#: cv2.ROTATE_90_CLOCKWISE is 0, which as a bare number would read as "none".
ROTATION_CODES: dict[Rotation, int | None] = {
    Rotation.NONE: None,
    Rotation.CW90: cv2.ROTATE_90_CLOCKWISE,
    Rotation.HALF: cv2.ROTATE_180,
    Rotation.CCW90: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
DEGREES = {code: rotation.value for rotation, code in ROTATION_CODES.items()}


def main(
    image: Annotated[Path, typer.Argument(help="Photo the mask belongs to")],
    mask: Annotated[Path, typer.Argument(help="Mask to snap onto the photo's grid")],
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output mask path")] = None,
    rotate: Annotated[Rotation, typer.Option(help="Rotation to apply, in degrees clockwise")] = Rotation.AUTO,
    nearest: Annotated[bool, typer.Option("--nearest", help="Nearest-neighbour instead of linear")] = False,
    overlay: Annotated[Path | None, typer.Option(help="Where to write the check image")] = None,
    no_overlay: Annotated[bool, typer.Option("--no-overlay", help="Skip the check image")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose logging")] = False,
) -> None:
    """Rotate and rescale a mask to match an image exactly."""
    configure_logging(verbose)

    out_path = output or image.with_name(f"{image.stem}{MASK_SUFFIX}.png")
    if out_path.resolve() == mask.resolve():
        # The default name is exactly what the pairing convention asks for, so
        # this collides precisely when the user followed the docs. Overwriting
        # would consume the only copy of a mask whose rotation is unverified.
        log.error(
            "Output '%s' is the input mask. Pass -o with a different name so the original survives "
            "in case the rotation turns out wrong.",
            out_path,
        )
        sys.exit(1)

    try:
        check_output_path(out_path, "Output")
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="prepare_mask_") as scratch:
        try:
            decode_path, bgr = decode_image(image, Path(scratch))
            code = (
                ROTATION_CODES[rotate] if rotate is not Rotation.AUTO else display_to_sensor_rotation(decode_path, bgr)
            )
            source = load_mask(mask)
        except (FileNotFoundError, ValueError) as exc:
            log.error("%s", exc)
            sys.exit(1)

        log.info("Rotation applied: %s degrees clockwise (%s)", DEGREES[code], rotate.value)
        log.info("Mask %dx%d px  ->  image %dx%d px", source.shape[1], source.shape[0], bgr.shape[1], bgr.shape[0])

        try:
            snapped = snap_mask_to_image(source, bgr.shape[:2], code, nearest=nearest)
            save_mask(out_path, snapped)
            if not no_overlay:
                overlay_path = overlay or DEFAULT_OVERLAY_DIR / f"{out_path.stem}_check.png"
                save_mask_overlay(overlay_path, bgr, snapped, f"{image.name} + {out_path.name}")
        except (ValueError, OSError) as exc:
            log.error("%s", exc)
            sys.exit(1)

    log_banner(log, "MASK PREPARED")
    log.info("  Size      : %d x %d px", snapped.shape[1], snapped.shape[0])
    log.info("  Foreground: %d px", int(snapped.sum()))
    log.info("  Check the overlay before trusting any measurement from this mask.")

    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    typer.run(main)
