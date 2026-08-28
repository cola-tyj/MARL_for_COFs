"""Public entry points for the frozen Our ET-Flow v5 inference route.

Imports are lazy so that lightweight tooling can inspect this package without
loading RDKit or the external ET-Flow runtime.
"""

from __future__ import annotations

from typing import Any


__all__ = ["generate_v5"]


def __getattr__(name: str) -> Any:
    if name == "generate_v5":
        from .generate_etflow_symmetric_xyz_v5 import generate_v5

        return generate_v5
    raise AttributeError(name)
