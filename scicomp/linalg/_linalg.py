"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
"""

__all__ = ["forward_substitution", "backward_substitution",
           "lu_factorization", "lu_factorization_with_partial_pivoting",
           "cholesky_factorization"]

import math

import numpy as np
from numpy.linalg import LinAlgError


# TODO: pretty print for intermediate iterations.

def forward_substitution(l, b):
    """Forward-Substitution for Lower Triangular System.

    # TODO: check matrix l is lower triangular
    # TODO: overwrite option

    Parameters
    ----------
    l : (N, N) array_like
        Lower triangular matrix

    b : (N,) array_like
        Right-hand side vector in l x = b

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    Solve the lower triangular system l x = b, where::

             [3  0  0  0]       [4]
        l =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    A simple application of the forward substitution algorithm is:

    >>> l = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=float)
    >>> b = np.array([4, 2, 4, 2], dtype=float)
    >>> forward_substitution(l, b)
    >>> b
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> l.dot(b)
    array([4., 2., 4., 2.])

    """
    n = l.shape[0]

    # x = np.zeros(n)  # solution component

    for j in range(0, n):  # loop over columns
        if l[j, j] == 0:  # stop if matrix is singular
            raise LinAlgError("Singular matrix")
        # x[j] = b[j] / l[j, j]
        b[j] /= l[j, j]
        for i in range(j + 1, n):
            # b[i] -= l[i, j] * x[j]  # update right-hand side
            b[i] -= l[i, j] * b[j]  # update right-hand side
    # return x


def backward_substitution(u, b):
    """Backward-Substitution for Lower Triangular System.

    # TODO: check matrix u is upper triangular
    # TODO: overwrite option

    Parameters
    ----------
    u : (N, N) array_like
        Upper triangular matrix

    b : (N,) array_like
        Right-hand side vector in l x = b

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    Solve the upper triangular system u x = b, where::

             [1  2  2]       [ 3]
        u =  [0 -4 -6]   b = [-6]
             [0  0 -1]       [ 1]

    A simple application of the backward substitution algorithm is:

    >>> u = np.array([[1, 2, 2], [0, -4, -6], [0, 0, -1]], dtype=float)
    >>> b = np.array([3, -6, 1], dtype=float)
    >>> backward_substitution(u, b)
    >>> b
    array([-1.,  3., -1.])
    >>> u.dot(b)
    array([ 3., -6.,  1.])

    """
    n = u.shape[0]

    x = np.zeros(n)  # solution component

    for j in range(n - 1, -1, -1):  # loop backwards over columns
        if u[j, j] == 0:  # stop if matrix is singular
            raise LinAlgError("Singular matrix")
        # x[j] = b[j] / u[j, j]
        b[j] /= u[j, j]  # compute solution component
        for i in range(0, j):
            # b[i] -= u[i, j] * x[j]  # update right-hand side
            b[i] -= u[i, j] * b[j]  # update right-hand side
    # return x


def solve(a, b, lu_factorized=False):
    """
    Solves the linear equation set ``a @ x == b`` for the unknown ``x``
    for square `a` matrix.

    Parameters
    ----------
    a : (M, N) array_like
        a is a known m * n matrix
    b : (N,) array_like
        Right-hand side n-vector in a x = b
    lu_factorized : bool, default: False
        Whether the factorization phase has been done. The LU formulation
        makes it clearer that the factorization phase need
        not be repeated when solving additional systems having different
        right-hand-side vectors but the same matrix A,
        since the L and U factors can be reused.

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    Solve the lower triangular system l x = b, where::

             [1  2  2]       [3]
        a =  [4  4  2]   b = [6]
             [4  6  4]       [10]

    A simple application of solve is:

    >>> a = np.array([[1., 2., 2.],
    ...               [4., 4., 2.],
    ...               [4., 6., 4.]], dtype=float)
    >>> b = np.array([3, 6, 10], dtype=float)
    >>> solve(a, b)
    >>> b
    array([-1.,  3., -1.])

    If given the LU factorization of a

    >>> a = np.array([[1., 2., 2.],
    ...               [4., 4., 2.],
    ...               [4., 6., 4.]], dtype=float)
    >>> b = np.array([3, 6, 10], dtype=float)
    >>> lu_factorization(a)
    >>> a
    array([[ 1.,  2.,  2.],
           [ 4., -4., -6.],
           [ 4., 0.5, -1.]])

    The factorization phase need not be repeated

    >>> solve(a, b, lu_factorized=True)
    >>> b
    array([-1.,  3., -1.])

    When solving additional systems having different right-hand-side
    vectors but the same matrix A, since the L and U factors can be reused.

    >>> b = np.array([2, 1, 9], dtype=float)
    >>> b
    array([-6.,  8.5, -4.5])

    """
    # TODO: memory optimization. l and u matrix in place instead of memory allocation
    n = a.shape[0]

    if not lu_factorized:
        lu_factorization(a)

    l = np.tril(a, k=-1) + np.eye(n)
    u = np.triu(a)
    forward_substitution(l, b)
    backward_substitution(u, b)


def lu_factorization(a):
    """LU Factorization by Gaussian Elimination without partial pivoting.

    # TODO: overwrite option

    Parameters
    ----------
    a : (N, N) array_like
        Array to decompose

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    A simple application of the cholesky factorization algorithm is:

    >>> a = np.array([[1., 2., 2.],
    ...               [4., 4., 2.],
    ...               [4., 6., 4.]], dtype=float)
    >>> lu_factorization(a)
    >>> a
    array([[ 1. ,  2. ,  2. ],
           [ 4. , -4. , -6. ],
           [ 4. ,  0.5, -1. ]])

    """
    n = a.shape[0]

    # TODO: in place option
    # # Then check overwrite permission
    # if not inplace:
    #     a1 = a.copy()

    for k in range(0, n - 1):
        if a[k, k] == 0:  # stop if pivot is zero
            raise LinAlgError("Singular matrix")
        for j in range(k + 1, n):
            for i in range(k + 1, n):
                m_ik = a[i, k] / a[k, k]  # compute multipliers for current column
                a[i, k] = m_ik
                a[i, j] = a[i, j] - m_ik * a[k, j]  # apply transformation to remaining submatrix


def lu_factorization_with_partial_pivoting(a):
    n = a.shape[0]

    # auxiliary integer vector is used to keep track of the new row order.
    # the net effect of all of the interchanges is
    # still just a permutation of the integers 1,...,n.
    interchange_vector = np.arange(n, dtype=int)

    for k in range(0, n - 1):
        p = np.argmax(abs(a[k:, k]))  # search for pivot in current column
        if p != k:  # interchange rows, if necessary
            # interchange_vector[k], interchange_vector[p] = interchange_vector[p], interchange_vector[k]
            a[[k, p], :] = a[[p, k], :]
        if a[k, k] == 0:  # skip current column if it's already zero
            continue
        for j in range(k + 1, n):
            for i in range(k + 1, n):
                # pi = np.where(interchange_vector == i)[0][0]
                # pk = np.where(interchange_vector == k)[0][0]
                m_ik = a[i, k] / a[k, k]  # compute multipliers for current column
                # a[i, k] = m_ik
                a[i, j] = a[i, j] - m_ik * a[k, j]  # apply transformation to remaining submatrix


def cholesky_factorization(a):
    """Cholesky factorization.

    # TODO: varify the matrix A is symmetric and positive definite
    # TODO: overwrite option

    Parameters
    ----------
    a : (N, N) array_like
        Symmetric and positive definite.

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    A simple application of the cholesky factorization algorithm is:

    >>> a = np.array([[3., -1., -1.],
    ...               [-1., 3., -1.],
    ...               [-1., -1., 3.]])
    >>> cholesky_factorization(a)
    >>> a
    array([[ 1.73205081, -1.        , -1.        ],
           [-0.57735027,  1.63299316, -1.        ],
           [-0.57735027, -0.81649658,  1.41421356]])

    """
    n = a.shape[0]
    for k in range(0, n):  # loop over columns
        a[k, k] = math.sqrt(a[k, k])
        for i in range(k + 1, n):
            a[i, k] = a[i, k] / a[k, k]  # scale current column
        # from each remaining column, subtract multiple of current column
        for j in range(k + 1, n):
            for i in range(j, n):
                a[i, j] -= a[i, k] * a[j, k]


def tridiagonal_lu_factorization_without_pivoting(a):
    """Tridiagonal LU Factorization Without Pivoting.

    # TODO: not an optimized algo for memory
    # TODO: only Heath 2.5.3 one non-zero lower and upper diagonals. special case

    Parameters
    ----------
    a : (N, N) array_like
        Symmetric and positive definite.

    Returns
    -------
    l : (N, N) array_like
        resulting lower triangular factors of a

    u : (N, N) array_like
        resulting upper triangular factors of a

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    A simple application of the tridiagonal lu factorization without pivoting is:

    >>> a = np.array([[5., 2., 0., 0., 0.],
    ...               [1., 4., 2., 0., 0.],
    ...               [0., 1., 3., 2., 0.],
    ...               [0., 0., 1., 2., 2.],
    ...               [0., 0., 0., 1., 1.]])
    >>> l, u = tridiagonal_lu_factorization_without_pivoting(a)
    >>> l
    array([[1.        , 0.        , 0.        , 0.        , 0.        ],
           [0.2       , 1.        , 0.        , 0.        , 0.        ],
           [0.        , 0.27777778, 1.        , 0.        , 0.        ],
           [0.        , 0.        , 0.40909091, 1.        , 0.        ],
           [0.        , 0.        , 0.        , 0.84615385, 1.        ]])
    >>> u
    array([[ 5.        ,  2.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  3.6       ,  2.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  2.44444444,  2.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.18181818,  2.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        , -0.69230769]])
    >>> l @ u
    array([[5., 2., 0., 0., 0.],
           [1., 4., 2., 0., 0.],
           [0., 1., 3., 2., 0.],
           [0., 0., 1., 2., 2.],
           [0., 0., 0., 1., 1.]])

    """
    n = a.shape[0]

    l = np.eye(n)
    u = np.triu(a, k=1)

    u[0, 0] = a[0, 0]
    for i in range(1, n):  # loop over columns
        l[i, i-1] = a[i, i-1] / u[i-1, i-1]  # compute multiplier
        u[i, i] = a[i, i] - l[i, i-1] * a[i-1, i]  # apply transformation

    return l, u


def blas_general_band_storage_mode(a, l_and_u):
    """Transform general band matrix to BLAS-general-band storage mode.

    Parameters
    ----------
    a : (M, N) array_like
        General band matrix

    (l, u) : (integer, integer)
        Number of non-zero lower and upper diagonals

    Returns
    -------
    agb : (l+u+1, N) array_like
        BLAS-general-band storage mode

    References
    ----------
    .. [1] `BLAS-General-Band Storage Mode
        <https://www.ibm.com/docs/en/essl/6.3?topic=representation-blas-general-band-storage-mode>`_

    Examples
    --------
    Following is an example where m>n, and general band matrix A is 9 by 8 with band widths of ml = 2 and mu = 3.
    Given the following matrix A:

    >>> a = np.array([[11., 12., 13., 14., 0., 0., 0., 0.],
    ...               [21., 22., 23., 24., 25., 0., 0., 0.],
    ...               [31., 32., 33., 34., 35., 36., 0., 0.],
    ...               [0., 42., 43., 44., 45., 46., 47., 0.],
    ...               [0., 0., 53., 54., 55., 56., 57., 58.],
    ...               [0., 0., 0., 64., 65., 66., 67., 68.],
    ...               [0., 0., 0., 0., 75., 76., 77., 78.],
    ...               [0., 0., 0., 0., 0., 86., 87., 88.],
    ...               [0., 0., 0., 0., 0., 0., 97., 98.]])

    You store it in array AGB, declared as AGB(6,8), as follows, where a_11 is stored in AGB(4,1):

    >>> agb = blas_general_band_storage_mode(a, (2, 3))
    >>> agb
    array([[ 0.,  0.,  0., 14., 25., 36., 47., 58.],
           [ 0.,  0., 13., 24., 35., 46., 57., 68.],
           [ 0., 12., 23., 34., 45., 56., 67., 78.],
           [11., 22., 33., 44., 55., 66., 77., 88.],
           [21., 32., 43., 54., 65., 76., 87., 98.],
           [31., 42., 53., 64., 75., 86., 97.,  0.]])

    Following is an example where m < n, and general band matrix A is 7 by 9 with band widths of ml = 2 and mu = 3.
    Given the following matrix A:

    >>> a = np.array([[11., 12., 13., 14., 0., 0., 0., 0., 0.],
    ...               [21., 22., 23., 24., 25., 0., 0., 0., 0.],
    ...               [31., 32., 33., 34., 35., 36., 0., 0., 0.],
    ...               [0., 42., 43., 44., 45., 46., 47., 0., 0.],
    ...               [0., 0., 53., 54., 55., 56., 57., 58., 0.],
    ...               [0., 0., 0., 64., 65., 66., 67., 68., 69.],
    ...               [0., 0., 0., 0., 75., 76., 77., 78., 79.]])

    You store it in array AGB, declared as AGB(6,9), as follows, where a11 is stored in AGB(4,1) and the leading diagonal does not fill up the whole row:

    >>> agb = blas_general_band_storage_mode(a, (2, 3))
    >>> agb
    array([[ 0.,  0.,  0., 14., 25., 36., 47., 58., 69.],
           [ 0.,  0., 13., 24., 35., 46., 57., 68., 79.],
           [ 0., 12., 23., 34., 45., 56., 67., 78.,  0.],
           [11., 22., 33., 44., 55., 66., 77.,  0.,  0.],
           [21., 32., 43., 54., 65., 76.,  0.,  0.,  0.],
           [31., 42., 53., 64., 75.,  0.,  0.,  0.,  0.]])

    """
    m = a.shape[0]
    n = a.shape[1]
    ml, mu = l_and_u

    agb = np.zeros((ml+mu+1, n))

    for j in range(0, n):
        k = mu + 1 - j - 1
        for i in range(max(0, j-mu), min(m, j+ml+1)):
            agb[k+i, j] = a[i, j]

    return agb


