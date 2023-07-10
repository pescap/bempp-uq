"""Global initialization for Bempp-UQ."""

from __future__ import absolute_import

__all__ = [
    "assembly",
    "foa",
    "functions",
    "operators",
    "preconditioning",
    "shapes",
    "utils",
    "config",
]

from . import (
    assembly,
    config,
    foa,
    functions,
    operators,
    preconditioning,
    shapes,
    utils,
)
