import os
import sys


def add_repo_root() -> str:
    """Add repo root to sys.path so iqa_router imports work from scripts."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
