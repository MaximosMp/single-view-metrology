"""Every vulture whitelist entry must still suppress a real finding.

Vulture reports names that are defined and never used, but never a whitelist
entry that outlived the code it covered: an entry is a use, not a definition.
A stale entry therefore sits there silently, ready to hide a future name that
happens to share it. This test is the check vulture does not do.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHITELIST = PROJECT_ROOT / "vulture_whitelist.py"


def _whitelisted_names() -> set[str]:
    """Return the attribute names the whitelist marks as used."""
    tree = ast.parse(WHITELIST.read_text(encoding="utf-8"))
    return {
        node.value.attr for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Attribute)
    }


def _findings_without_whitelist() -> set[str]:
    """Return the names vulture flags in production code when the whitelist is left out."""
    result = subprocess.run(
        [sys.executable, "-m", "vulture", "src", "scripts", "--min-confidence", "60"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return set(re.findall(r"unused \w+ '(\w+)'", result.stdout))


def test_whitelist_is_not_empty_by_accident():
    # Guards the parser: if it stopped recognising entries, the main test would
    # pass vacuously.
    assert _whitelisted_names()


def test_every_whitelist_entry_suppresses_a_finding():
    stale = _whitelisted_names() - _findings_without_whitelist()
    assert not stale, f"Whitelist entries that no longer match any finding, remove them: {sorted(stale)}"
