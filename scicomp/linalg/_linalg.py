"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
"""

__all__ = ["forward_substitution", "backward_substitution"]

import numpy as np
from numpy.linalg import LinAlgError


def forward_substitution(l, b):
    """

    Parameters
    ----------
    l
    b

    Returns
    -------

    """
    n = l.shape[0]

    x = np.zeros(n)  # solution component

    for j in range(0, n):  # loop over columns
        if l[j, j] == 0:  # stop if matrix is singular
            raise LinAlgError("Singular matrix")
        x[j] = b[j] / l[j, j]
        for i in range(j + 1, n):
            b[i] -= l[i, j] * x[j]  # update right-hand side

    return x


def backward_substitution(u, b):
    """

    Parameters
    ----------
    u
    b

    Returns
    -------

    """
    n = u.shape[0]

    x = np.zeros(n)  # solution component

    for j in range(n - 1, -1, -1):  # loop backwards over columns
        if u[j, j] == 0:  # stop if matrix is singular
            raise LinAlgError("Singular matrix")
        x[j] = b[j] / u[j, j]
        for i in range(0, j):
            b[i] -= u[i, j] * x[j]  # update right-hand side

    return x
