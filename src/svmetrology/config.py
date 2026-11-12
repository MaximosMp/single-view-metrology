"""Typed loading of the YAML settings.

Each config file is checked against a dataclass: a missing key, an unknown key
or a value of the wrong type stops the run and names the key, rather than a
built-in fallback quietly standing in for it. The YAML file is the only place
a value is set.

Relative paths inside a config resolve against that config file, so a run
behaves the same from any working directory.

The schemas live beside the code they configure; this module imports nothing
from the package, so any module can load its own settings without a cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException


def load_yaml_as[T](path: Path, schema: type[T]) -> T:
    """Load a YAML file into a dataclass, validating every key.

    Args:
        path: YAML file.
        schema: Dataclass the file must match exactly.

    Returns:
        An instance of ``schema``, nested dataclasses included.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: a key is missing, unknown, or holds the wrong type.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{path}' not found")
    try:
        merged = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(path))
        return cast(T, OmegaConf.to_object(merged))
    except OmegaConfBaseException as exc:
        # OmegaConf keeps the offending key in an attribute, not always in the
        # message's first line, and the key is what the reader needs.
        key = getattr(exc, "full_key", None)
        raise ValueError(f"{path}: {f'{key}: ' if key else ''}{str(exc).splitlines()[0]}") from exc


def resolve_path(value: str | Path, config_path: Path) -> Path:
    """Resolve a path read from a config against that config file's folder.

    Args:
        value: Absolute, home-relative or relative path.
        config_path: The config file the value came from.

    Returns:
        An absolute path.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()
