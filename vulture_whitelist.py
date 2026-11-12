"""Whitelist of names vulture cannot see being used.

Vulture parses Python source only, so a name written in Python and read
somewhere else looks unused. Referencing it here marks it live. Keep this file
minimal: every entry suppresses a warning, so an unnecessary one hides real dead
code. tests/test_vulture_whitelist.py fails on any entry that stops being needed.
"""

import cv2
from torch import nn

# cv2.SimpleBlobDetector.Params fields. Assigned in Python, read by OpenCV's
# C++ detector, so no Python source ever reads them back.
_params = cv2.SimpleBlobDetector.Params()
_params.minArea
_params.maxArea

# nn.Module.forward. PyTorch calls it from Module.__call__, so `network(batch)`
# runs every forward method of the depth network without any source naming it.
nn.Module.forward
