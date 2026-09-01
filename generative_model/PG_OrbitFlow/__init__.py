"""PG-OrbitFlow: an isolated point-group conditioned molecular flow branch.

This package intentionally does not import or modify the historical
``our_ET_Flow`` inference pipeline.  It only consumes the frozen canonical v2
dataset contract.
"""

from .model import PGOrbitFlow

__all__ = ["PGOrbitFlow"]
__version__ = "0.1.0"
