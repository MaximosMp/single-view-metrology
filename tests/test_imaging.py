"""Tests for image loading and resizing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from svmetrology.imaging import (
    display_to_sensor_rotation,
    find_image_mask_pairs,
    load_folder,
    load_mask,
    load_pair,
    resize_longest_edge,
    save_mask,
    snap_mask_to_image,
)


def test_resize_downscales_to_the_limit():
    image = np.zeros((400, 1000, 3), dtype=np.uint8)
    out = resize_longest_edge(image, 500)
    assert out.shape[:2] == (200, 500)


def test_resize_never_upscales():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = resize_longest_edge(image, 5000)
    assert out is image, "an image within the limit must be returned untouched"


def test_resize_is_idempotent():
    image = np.zeros((400, 1000, 3), dtype=np.uint8)
    once = resize_longest_edge(image, 500)
    assert resize_longest_edge(once, 500).shape == once.shape


def test_resize_preserves_aspect_ratio():
    image = np.zeros((768, 1024, 3), dtype=np.uint8)
    out = resize_longest_edge(image, 512)
    assert out.shape[1] / out.shape[0] == pytest.approx(1024 / 768, abs=0.01)


def test_resize_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="must be positive"):
        resize_longest_edge(np.zeros((10, 10), dtype=np.uint8), 0)


def test_load_folder_rejects_a_missing_folder(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        load_folder(tmp_path / "nope")


def test_load_folder_rejects_a_folder_with_no_images(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("not an image")
    with pytest.raises(FileNotFoundError):
        load_folder(tmp_path)


def test_load_folder_returns_names_and_bgr_in_order(tmp_path: Path):
    for name in ("b.png", "a.png"):
        cv2.imwrite(str(tmp_path / name), np.full((20, 30, 3), 128, dtype=np.uint8))
    loaded = load_folder(tmp_path)
    assert [name for name, _ in loaded] == ["a.png", "b.png"]
    assert loaded[0][1].shape == (20, 30, 3)
    assert loaded[0][1].dtype == np.uint8


def test_load_folder_skips_unsupported_extensions(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "keep.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    (tmp_path / "skip.txt").write_text("x")
    assert [name for name, _ in load_folder(tmp_path)] == ["keep.png"]


def test_load_folder_skips_a_file_that_does_not_decode(tmp_path: Path):
    # One corrupt capture must not cost the rest of a calibration set.
    cv2.imwrite(str(tmp_path / "good.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    (tmp_path / "corrupt.jpg").write_bytes(b"not a jpeg")
    assert [name for name, _ in load_folder(tmp_path)] == ["good.png"]


# -- Masks --------------------------------------------------------------------


def _write_image(path: Path, height: int, width: int):
    """Write a black test image of the given size."""
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def _write_mask(path: Path, height: int, width: int):
    """Write an all-foreground test mask of the given size."""
    cv2.imwrite(str(path), np.full((height, width), 255, dtype=np.uint8))


def test_load_mask_binarises_at_the_threshold(tmp_path: Path):
    grey = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    path = tmp_path / "m.png"
    cv2.imwrite(str(path), grey)
    assert load_mask(path, threshold=127).tolist() == [[False, False], [True, True]]


def test_pairs_an_image_with_its_mask(tmp_path: Path):
    _write_image(tmp_path / "shot.jpg", 20, 30)
    _write_mask(tmp_path / "shot_mask.png", 20, 30)
    pairs, unpaired = find_image_mask_pairs(tmp_path)
    assert [(i.name, m.name) for i, m in pairs] == [("shot.jpg", "shot_mask.png")]
    assert unpaired == []


def test_reports_an_image_without_a_mask(tmp_path: Path):
    _write_image(tmp_path / "lonely.jpg", 10, 10)
    pairs, unpaired = find_image_mask_pairs(tmp_path)
    assert pairs == []
    assert [p.name for p in unpaired] == ["lonely.jpg"]


def test_reports_a_mask_without_an_image(tmp_path: Path):
    _write_mask(tmp_path / "orphan_mask.png", 10, 10)
    pairs, unpaired = find_image_mask_pairs(tmp_path)
    assert pairs == []
    assert [p.name for p in unpaired] == ["orphan_mask.png"]


def test_rejects_two_candidate_masks_for_one_image(tmp_path: Path):
    _write_image(tmp_path / "shot.jpg", 10, 10)
    _write_mask(tmp_path / "shot_mask.png", 10, 10)
    _write_mask(tmp_path / "shot_mask.jpg", 10, 10)
    with pytest.raises(ValueError, match="candidate masks"):
        find_image_mask_pairs(tmp_path)


def test_a_mask_file_is_never_paired_as_an_image(tmp_path: Path):
    _write_mask(tmp_path / "shot_mask.png", 10, 10)
    _write_image(tmp_path / "shot.jpg", 10, 10)
    pairs, _ = find_image_mask_pairs(tmp_path)
    # Nothing should go looking for shot_mask_mask.png.
    assert len(pairs) == 1


def test_load_pair_returns_image_and_mask(tmp_path: Path):
    _write_image(tmp_path / "shot.png", 20, 30)
    _write_mask(tmp_path / "shot_mask.png", 20, 30)
    bgr, mask = load_pair(tmp_path / "shot.png", tmp_path / "shot_mask.png", tmp_path)
    assert bgr.shape == (20, 30, 3)
    assert mask.shape == (20, 30)
    assert mask.dtype == np.bool_


def test_load_pair_rejects_a_mask_of_the_wrong_size(tmp_path: Path):
    _write_image(tmp_path / "shot.png", 20, 30)
    _write_mask(tmp_path / "shot_mask.png", 40, 60)
    with pytest.raises(ValueError, match="never rotates or resizes"):
        load_pair(tmp_path / "shot.png", tmp_path / "shot_mask.png", tmp_path)


def test_snap_rotates_and_rescales_onto_the_image_grid():
    mask = np.zeros((40, 30), dtype=bool)  # portrait, half scale
    mask[10:20, 5:15] = True
    snapped = snap_mask_to_image(mask, (60, 80), rotate_code=cv2.ROTATE_90_CLOCKWISE)
    assert snapped.shape == (60, 80)
    assert snapped.any()


def test_snap_without_rotation_only_rescales():
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:25, 10:30] = True
    snapped = snap_mask_to_image(mask, (60, 80))
    assert snapped.shape == (60, 80)
    # A block covering half the source should still cover about half the target.
    assert snapped.sum() == pytest.approx(mask.sum() * 4, rel=0.1)


def test_snap_rejects_a_mismatched_aspect_ratio():
    mask = np.zeros((40, 30), dtype=bool)
    with pytest.raises(ValueError, match="aspect"):
        snap_mask_to_image(mask, (60, 80))


def test_rejects_two_images_sharing_a_stem(tmp_path: Path):
    # Symmetric with the two-masks rule: letting sort order pick would measure
    # a file the mask may never have been drawn on.
    _write_image(tmp_path / "shot.png", 10, 10)
    _write_image(tmp_path / "shot.jpg", 10, 10)
    _write_mask(tmp_path / "shot_mask.png", 10, 10)
    with pytest.raises(ValueError, match="candidate images"):
        find_image_mask_pairs(tmp_path)


def test_load_mask_reads_a_cutout_from_its_alpha_channel(tmp_path: Path):
    # A "remove the background" export keeps colour under transparent pixels;
    # read as greyscale the whole frame would come back as foreground.
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[5:15, 10:20, 3] = 255
    path = tmp_path / "cutout.png"
    cv2.imwrite(str(path), rgba)
    mask = load_mask(path)
    assert mask.sum() == 10 * 10


def test_load_mask_warns_when_nothing_clears_the_threshold(tmp_path: Path, caplog):
    # A 0/1 label map: plainly non-empty, but empty after thresholding.
    path = tmp_path / "labels.png"
    cv2.imwrite(str(path), np.ones((10, 10), dtype=np.uint8))
    with caplog.at_level("WARNING"):
        assert not load_mask(path).any()
    assert "reads as empty" in caplog.text


def test_save_mask_roundtrips_through_load_mask(tmp_path: Path):
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:20, 10:30] = True
    path = tmp_path / "nested" / "m.png"
    save_mask(path, mask)
    assert np.array_equal(load_mask(path), mask)


def test_save_mask_rejects_an_unwritable_extension(tmp_path: Path):
    with pytest.raises(OSError, match="Could not write mask"):
        save_mask(tmp_path / "m.txt", np.ones((4, 4), dtype=bool))


def test_rotation_detection_reports_none_when_frames_agree(tmp_path: Path):
    path = tmp_path / "plain.jpg"
    _write_image(path, 20, 30)
    sensor = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    assert display_to_sensor_rotation(path, sensor) is None


def test_rotation_detection_refuses_a_non_rotation(tmp_path: Path):
    # A mirrored frame preserves the aspect ratio, so nothing downstream would
    # catch it; returning None here would read as "no rotation needed".
    path = tmp_path / "plain.jpg"
    cv2.imwrite(str(path), np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8))
    sensor = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    with pytest.raises(ValueError, match="No rotation maps"):
        display_to_sensor_rotation(path, cv2.flip(sensor, 1))


def test_load_pair_accepts_the_committed_fixture(ruler_pair: tuple[Path, Path], tmp_path: Path):
    # The real pair through the real loader, so the strict size check is
    # exercised on data rather than only on synthetic squares.
    image_path, mask_path = ruler_pair
    bgr, mask = load_pair(image_path, mask_path, tmp_path)
    assert bgr.shape[:2] == mask.shape[:2] == (1200, 1600)
    assert mask.any()
