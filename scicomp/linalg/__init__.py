"""
====================================
Linear algebra (:mod:`scicomp.linalg`)
====================================


"""

from ._eigen import *


__all__ = [s for s in dir() if not s.startswith("_")]
