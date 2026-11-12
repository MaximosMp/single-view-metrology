"""Image and mask I/O.

Every photo enters the pipeline through `decode_image`, which keeps pixels in
the camera's sensor frame: EXIF orientation is ignored, so the intrinsics and
every later stage share one coordinate convention, and rotating for display is
left to whatever displays. HEIC captures are converted to JPEG with macOS
``sips`` on the way in.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Formats OpenCV decodes directly.
DECODABLE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})
#: iPhone captures, converted to JPEG before decoding.
HEIC_SUFFIXES = frozenset({".heic", ".heif"})

#: A file whose stem ends in this is the mask for the stem without it.
MASK_SUFFIX = "_mask"


def _heic_to_jpeg(source: Path, out_dir: Path) -> Path:
    """Convert a HEIC capture to JPEG with macOS ``sips``.

    Args:
        source: HEIC file.
        out_dir: Folder for the JPEG; caller-owned.

    Returns:
        Path of the JPEG.

    Raises:
        FileNotFoundError: not running on macOS, or ``sips`` failed.
    """
    if platform.system() != "Darwin":
        raise FileNotFoundError(f"'{source.name}' is HEIC, which converts only on macOS; export it as JPEG first")
    target = out_dir / f"{source.stem}.jpg"
    result = subprocess.run(
        ["sips", "-s", "format", "jpeg", str(source), "--out", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise FileNotFoundError(f"sips could not convert '{source.name}': {result.stderr.strip()}")
    return target


def _read_bgr(path: Path, *, sensor_frame: bool = True) -> np.ndarray | None:
    """Decode an image file to BGR.

    Args:
        path: A file OpenCV can decode.
        sensor_frame: Ignore EXIF orientation. Turned off only to see the frame
            a viewer would show.

    Returns:
        H x W x 3 ``np.ndarray`` with ``dtype=uint8``, or ``None`` when the
        file does not decode.
    """
    flags = cv2.IMREAD_COLOR | (cv2.IMREAD_IGNORE_ORIENTATION if sensor_frame else 0)
    return cv2.imread(str(path), flags)


def load_folder(folder: Path) -> list[tuple[str, np.ndarray]]:
    """Decode every photo directly inside a folder, in filename order.

    One unreadable capture should not cost the rest of a calibration set, so a
    file that fails to decode is logged and skipped.

    Args:
        folder: Folder to read; subfolders are ignored.

    Returns:
        ``(filename, bgr)`` pairs, bgr being H x W x 3 ``np.ndarray`` with
        ``dtype=uint8`` in the sensor frame.

    Raises:
        NotADirectoryError: ``folder`` is not a directory.
        FileNotFoundError: nothing in it decodes.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a directory")

    photos: list[tuple[str, np.ndarray]] = []
    with tempfile.TemporaryDirectory(prefix="svm_decode_") as scratch:
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in DECODABLE_SUFFIXES | HEIC_SUFFIXES:
                continue
            try:
                photos.append((path.name, decode_image(path, Path(scratch))[1]))
            except FileNotFoundError as exc:
                log.warning("Skipping %s: %s", path.name, exc)

    if not photos:
        raise FileNotFoundError(f"No decodable images in '{folder}'")
    log.info("Loaded %d image(s) from '%s'", len(photos), folder)
    return photos


def resize_longest_edge(image: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Shrink an image so its longer side is at most ``max_long_edge``; never enlarge.

    Area interpolation averages each output pixel's whole footprint, which keeps
    dot centroids where they were; point sampling would jitter them.

    Args:
        image: H x W or H x W x C ``np.ndarray`` with ``dtype=uint8``.
        max_long_edge: Longest allowed side, in px.

    Returns:
        The shrunk image, or ``image`` itself when it already fits.

    Raises:
        ValueError: ``max_long_edge`` is not positive.
    """
    if max_long_edge <= 0:
        raise ValueError(f"max_long_edge must be positive, got {max_long_edge}")
    height, width = image.shape[:2]
    if max(height, width) <= max_long_edge:
        return image
    scale = max_long_edge / max(height, width)
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


# -- Masks --------------------------------------------------------------------


def load_mask(path: Path, threshold: int = 127) -> np.ndarray:
    """Load a mask file as a boolean array.

    A transparent cutout is read from its alpha channel, which is what the
    common "remove the background" export actually encodes; reading such a file
    as greyscale would threshold the leftover colour instead and silently
    return the wrong region.

    Otherwise the grey level is thresholded rather than treated as "any nonzero
    value is foreground": an antialiased export has soft edges, and nonzero
    would bleed the object outward by a pixel or two without anything saying so.

    Args:
        path: Mask image file. Greyscale, colour, or RGBA cutout.
        threshold: Exclusive level above which a pixel is foreground.

    Returns:
        H x W ``np.ndarray`` with ``dtype=bool``, True where foreground.

    Raises:
        FileNotFoundError: the file cannot be decoded.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if raw is None:
        raise FileNotFoundError(f"Could not decode mask '{path}'")

    if raw.ndim == 3 and raw.shape[2] == 4:
        log.info("Mask '%s' has an alpha channel; using it as the foreground.", path.name)
        plane = raw[:, :, 3]
    elif raw.ndim == 3:
        plane = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        plane = raw

    mask = plane > threshold
    if not mask.any() and plane.any():
        # A 0/1 label map, or a dark-coloured object: every value is nonzero
        # somewhere but none clears the threshold. Saying so beats the bare
        # "no foreground pixels" the caller would otherwise report.
        log.warning(
            "Mask '%s' has values up to %d, none above the threshold of %d, so it reads as empty. "
            "Expected 0/255; rescale a 0/1 label map before measuring.",
            path.name,
            int(plane.max()),
            threshold,
        )
    return mask


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Write a boolean mask as a 0/255 image.

    The inverse of `load_mask`, kept beside it so the on-disk encoding is
    decided once rather than re-spelled at each call site.

    Args:
        path: Destination file; parent directories are created. Use PNG — a
            lossy format's ringing would move the binarisation on reload.
        mask: H x W ``np.ndarray``, bool or numeric. Nonzero is foreground.

    Raises:
        OSError: the image could not be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 signals an unknown extension by raising and an unwritable path by
    # returning False; both mean the file is not there, so both become OSError.
    try:
        written = cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)
    except cv2.error as exc:
        raise OSError(f"Could not write mask '{path}': {exc.err or exc}".strip()) from exc
    if not written:
        raise OSError(f"Could not write mask '{path}'. Check the path and its extension.")


def find_image_mask_pairs(folder: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Pair each image in ``folder`` with its mask by filename.

    ``photo.heic`` pairs with ``photo_mask.png``; the mask's extension is
    independent of the image's. A file whose stem already ends in the mask
    suffix is never treated as an image, so nothing hunts for ``x_mask_mask``.

    Args:
        folder: Folder to read; subfolders are ignored.

    Returns:
        ``(pairs, unpaired)``, where pairs are ``(image_path, mask_path)`` and
        unpaired holds every file that found no counterpart, sorted by name.

    Raises:
        NotADirectoryError: ``folder`` is not a directory.
        ValueError: one stem has more than one image, or more than one mask.
    """
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a directory")

    known = DECODABLE_SUFFIXES | HEIC_SUFFIXES
    masks: dict[str, list[Path]] = {}
    images: dict[str, list[Path]] = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in known:
            continue
        if p.stem.endswith(MASK_SUFFIX):
            masks.setdefault(p.stem[: -len(MASK_SUFFIX)], []).append(p)
        else:
            images.setdefault(p.stem, []).append(p)

    pairs: list[tuple[Path, Path]] = []
    unpaired: list[Path] = []
    for stem, candidates in images.items():
        # Ambiguity is symmetric: two photos of one stem is as unresolvable as
        # two masks. Letting sort order decide would measure a file the mask was
        # never drawn on, and report the other as having no counterpart at all.
        _reject_ambiguity(stem, candidates, "images")
        found = masks.pop(stem, [])
        _reject_ambiguity(stem, found, "masks")
        if found:
            pairs.append((candidates[0], found[0]))
        else:
            unpaired.append(candidates[0])

    for orphaned in masks.values():
        unpaired.extend(orphaned)

    return pairs, sorted(unpaired)


def _reject_ambiguity(stem: str, candidates: list[Path], kind: str) -> None:
    """Raise when one stem has more than one candidate file of a kind.

    Args:
        stem: The shared filename stem.
        candidates: Files found for it.
        kind: Plural word for what they are, used in the message.

    Raises:
        ValueError: more than one candidate was found.
    """
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise ValueError(
            f"'{stem}' has {len(candidates)} candidate {kind} ({names}). Keep exactly one — "
            "choosing arbitrarily would measure the wrong file without saying so."
        )


def decode_image(image_path: Path, scratch_dir: Path) -> tuple[Path, np.ndarray]:
    """Decode a photo in the sensor frame, converting from HEIC when needed.

    The one place format conversion happens, so every consumer of an image sees
    the same pixels. The decodable path is returned as well, because a caller
    that needs to re-read the file (to inspect its EXIF-applied frame, say)
    must read the converted JPEG rather than the original container.

    Args:
        image_path: Photo to decode.
        scratch_dir: Directory for any intermediate JPEG; caller-owned.

    Returns:
        ``(decodable_path, bgr)`` where bgr is H x W x 3 ``np.ndarray`` with
        ``dtype=uint8`` in the sensor frame.

    Raises:
        FileNotFoundError: the file cannot be converted or decoded.
    """
    is_heic = image_path.suffix.lower() in HEIC_SUFFIXES
    decodable = _heic_to_jpeg(image_path, scratch_dir) if is_heic else image_path
    bgr = _read_bgr(decodable)
    if bgr is None:
        raise FileNotFoundError(f"Could not decode image '{image_path}'")
    return decodable, bgr


def load_pair(image_path: Path, mask_path: Path, scratch_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and its mask, requiring identical pixel dimensions.

    The size check is made against the decoded image rather than file metadata,
    because for HEIC input the converted JPEG is the only array the pipeline
    ever sees. No rotation or resampling happens here: a mask that does not fit
    is an error, not something to silently reinterpret.

    Args:
        image_path: Photo to measure. HEIC is converted first.
        mask_path: Mask for that photo.
        scratch_dir: Directory for any intermediate JPEG; caller-owned.

    Returns:
        ``(bgr, mask)`` where bgr is H x W x 3 ``uint8`` in the sensor frame and
        mask is H x W ``bool``.

    Raises:
        FileNotFoundError: either file cannot be decoded or converted.
        ValueError: the mask's dimensions differ from the image's.
    """
    _, bgr = decode_image(image_path, scratch_dir)
    mask = load_mask(mask_path)

    if bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Mask '{mask_path.name}' is {mask.shape[1]}x{mask.shape[0]} px but image "
            f"'{image_path.name}' is {bgr.shape[1]}x{bgr.shape[0]} px. The pipeline never rotates or "
            "resizes a mask; run scripts/prepare_mask.py to snap it onto the image grid."
        )
    return bgr, mask


def display_to_sensor_rotation(path: Path, sensor: np.ndarray) -> int | None:
    """Find the rotation taking an image's display frame to its sensor frame.

    Decodes the file honouring EXIF orientation and reports which rotation maps
    that onto the sensor frame the caller already holds. This measures the
    transform a viewer actually applied — which is what a masking tool drew on —
    rather than inferring it from an EXIF tag.

    Args:
        path: Image file, already decodable (convert HEIC first).
        sensor: H x W x 3 ``np.ndarray`` with ``dtype=uint8``, the same file
            decoded in the sensor frame.

    Returns:
        A ``cv2.ROTATE_*`` constant, or ``None`` when the two frames already
        agree and no rotation is needed.

    Raises:
        FileNotFoundError: the file cannot be decoded.
        ValueError: the two frames differ by something other than a rotation,
            such as a mirror. Returning None here would be indistinguishable
            from "no rotation needed" and would produce a correctly sized mask
            in the wrong place.
    """
    display = _read_bgr(path, sensor_frame=False)
    if display is None:
        raise FileNotFoundError(f"Could not decode image '{path}'")

    for code in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        candidate = display if code is None else cv2.rotate(display, code)
        if np.array_equal(candidate, sensor):
            return code

    raise ValueError(
        f"No rotation maps the display frame of '{path.name}' onto its sensor frame — they differ by "
        "something else, such as a mirror. Pass an explicit --rotate rather than trusting detection."
    )


def snap_mask_to_image(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    rotate_code: int | None = None,
    nearest: bool = False,
) -> np.ndarray:
    """Rotate and rescale a mask onto an image's pixel grid.

    Rotation is applied before resizing so the operation has one fixed rounding
    path. The default resampling is linear followed by a mid-level cut, which
    approximates the true half-level contour; nearest-neighbour would quantise
    the boundary into blocks the size of the scale factor.

    Args:
        mask: H x W ``np.ndarray``, bool or ``uint8``.
        image_shape: ``(height, width)`` of the target image, in pixels.
        rotate_code: A ``cv2.ROTATE_*`` constant, or None for no rotation.
        nearest: Use nearest-neighbour resampling instead of linear.

    Returns:
        ``image_shape`` ``np.ndarray`` with ``dtype=bool``.

    Raises:
        ValueError: the rotated mask's aspect ratio does not match the target's.
    """
    grey = (mask > 0).astype(np.uint8) * 255
    oriented = grey if rotate_code is None else cv2.rotate(grey, rotate_code)

    target_h, target_w = image_shape
    src_h, src_w = oriented.shape[:2]
    target_ratio = target_w / target_h
    if abs(src_w / src_h - target_ratio) > 0.01 * target_ratio:
        raise ValueError(
            f"Mask is {src_w}x{src_h} px after rotation, aspect {src_w / src_h:.4f}, but the image is "
            f"{target_w}x{target_h} px, aspect {target_ratio:.4f}. Check the rotation before rescaling."
        )

    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(oriented, (target_w, target_h), interpolation=interpolation) > 127
