"""
====================================
Linear algebra (:mod:`scicomp.linalg`)
====================================


"""

from ._linalg import *
from ._eigen import *
from ._qr_factorization import *
from ._iterative import *


__all__ = [s for s in dir() if not s.startswith("_")]
