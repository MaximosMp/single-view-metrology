"""Tests for typed config loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from svmetrology.board import CircleGrid
from svmetrology.calibration import CalibrationConfig, RejectionSettings
from svmetrology.config import load_yaml_as, resolve_path
from svmetrology.detection import DetectionSettings

BOARD_YAML = "cols: 4\nrows: 11\npitch_cm: 0.83\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_shipped_config_loads_into_typed_settings(calibration_config: CalibrationConfig):
    assert isinstance(calibration_config.detection, DetectionSettings)
    assert calibration_config.detection.dot_sizes, "at least one dot size to try"
    assert isinstance(calibration_config.rejection, RejectionSettings)
    assert calibration_config.max_long_edge is None


def test_loads_a_board(tmp_path: Path):
    assert load_yaml_as(_write(tmp_path, BOARD_YAML), CircleGrid) == CircleGrid(4, 11, 0.83)


def test_a_missing_key_is_named(tmp_path: Path):
    with pytest.raises(ValueError, match="pitch_cm"):
        load_yaml_as(_write(tmp_path, "cols: 4\nrows: 11\n"), CircleGrid)


def test_an_unknown_key_is_named(tmp_path: Path):
    # A misspelled key must fail, not sit beside a default nobody overrode.
    with pytest.raises(ValueError, match="pitch_mm"):
        load_yaml_as(_write(tmp_path, BOARD_YAML + "pitch_mm: 8.3\n"), CircleGrid)


def test_a_wrongly_typed_value_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="cols"):
        load_yaml_as(_write(tmp_path, BOARD_YAML.replace("cols: 4", "cols: four")), CircleGrid)


def test_a_missing_file_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml_as(tmp_path / "absent.yaml", CircleGrid)


def test_relative_paths_resolve_against_the_config_file(tmp_path: Path):
    config = tmp_path / "configs" / "calibration.yaml"
    assert resolve_path("../images", config) == (tmp_path / "images").resolve()


def test_absolute_paths_are_kept(tmp_path: Path):
    assert resolve_path(tmp_path / "x", tmp_path / "configs" / "c.yaml") == tmp_path / "x"


def test_home_relative_paths_expand(tmp_path: Path):
    assert resolve_path("~/x", tmp_path / "c.yaml") == Path.home() / "x"
