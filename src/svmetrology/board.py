"""The printed calibration board: an OpenCV asymmetric circle grid.

The board is the one object in a capture whose physical size is known. That
makes it the input to intrinsic calibration, and later the scale reference that
turns pixels into centimetres. Its geometry lives in a small YAML file under
``configs/boards/``, so a reprint at another size is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CircleGrid:
    """Geometry of an asymmetric circle grid.

    Every other row is shifted by one pitch, which is what makes the grid's
    orientation unambiguous from a single view.

    Attributes:
        cols: Dots per row, as OpenCV counts them.
        rows: Number of rows.
        pitch_cm: Row pitch in cm: half the centre distance between two
            neighbouring dots in the same row.
    """

    cols: int
    rows: int
    pitch_cm: float

    @property
    def shape(self) -> tuple[int, int]:
        """``(cols, rows)``, the pattern size OpenCV's grid finder expects."""
        return self.cols, self.rows

    def object_points(self) -> np.ndarray:
        """Return the dot centres in the board's own frame.

        Ordered row by row, the order OpenCV reports detected centres in, so
        index i here and index i of a detection are the same physical dot.

        Returns:
            ``(cols * rows) x 3`` ``np.ndarray`` with ``dtype=float32``: x and y
            in cm, z = 0 because the board is flat.
        """
        row, col = np.divmod(np.arange(self.cols * self.rows), self.cols)
        x = (2 * col + row % 2) * self.pitch_cm
        y = row * self.pitch_cm
        return np.column_stack([x, y, np.zeros_like(x)]).astype(np.float32)
