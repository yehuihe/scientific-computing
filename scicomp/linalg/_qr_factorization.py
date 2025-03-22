from functools import partial
import math
import warnings

import numpy as np
from numpy.linalg import LinAlgError

from ._linalg import backward_substitution

__all__ = ["householder", "solve_linear_least_square",
           "polynomial_fitting",
           "classical_gram_schmidt_orthogonalization",
           "modified_gram_schmidt_orthogonalization"]

# TODO: normal equation

def householder(a, b=None, explicit_q=False, economy=False):
    """Householder QR Factorization.

    # TODO: Rank Deficiency and column pivoting

    Parameters
    ----------
    a : (N, N) array_like
        Array to decompose

    b : (N,) array_like, default: None
        solve a linear least squares problem A @ x ~= b. If b specified, not a in place algorithm.
        A new copy of Q is required.

    explicit_q : bool, default: False
        If Q is needed explicitly

    economy : bool, default: False
        Return reduced, or "economy size" QR factorization of A

    Returns
    -------
    Q : complex ndarray
        Q matrix in QR factorization. Only return when explicit_q is True

    R : complex ndarray
        R matrix in QR factorization.

    C_1 : ndarray
        The transformed righthand side c_1 = Q_1.T @ b. Only when b is specified

    References
    ----------
    .. [1] Alston S. Householder. Unitary triangularization of a nonsymmetric matrix.
    J. ACM, 5(4):339–342, October 1958. ISSN 0004-5411. doi: 10.1145/320941.320947.
    URL https://doi.org/10.1145/320941.320947.

    .. [2] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    A simple application of the householder QR factorization without explicit Q is:

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> r = householder(a)
    >>> r
    array([[-1.73205081e+00,  5.77350269e-01,  5.77350269e-01],
           [ 0.00000000e+00, -1.63299316e+00,  8.16496581e-01],
           [ 0.00000000e+00,  0.00000000e+00, -1.41421356e+00],
           [ 0.00000000e+00,  0.00000000e+00,  6.93889390e-18],
           [ 0.00000000e+00,  0.00000000e+00,  1.11022302e-16],
           [ 0.00000000e+00,  0.00000000e+00,  1.11022302e-16]])

    If Q is needed explicitly for some other reason, this computation
    will require additional computation and storage.

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> q, r = householder(a, explicit_q=True)
    >>> q
    array([[-5.77350269e-01, -2.04124145e-01, -3.53553391e-01,
            5.11339220e-01,  4.87831518e-01, -2.35077025e-02],
           [0.00000000e+00, -6.12372436e-01, -3.53553391e-01,
            -4.87831518e-01,  2.35077025e-02,  5.11339220e-01],
           [0.00000000e+00,  0.00000000e+00, -7.07106781e-01,
            -2.35077025e-02, -5.11339220e-01, -4.87831518e-01],
           [5.77350269e-01, -4.08248290e-01, -1.80092860e-17,
            6.66390246e-01, -1.78558728e-01,  1.55051026e-01],
           [5.77350269e-01,  2.04124145e-01, -3.53553391e-01,
            -1.55051026e-01,  6.66390246e-01, -1.78558728e-01],
           [0.00000000e+00,  6.12372436e-01, -3.53553391e-01,
            1.78558728e-01, -1.55051026e-01,  6.66390246e-01]])
    >>> r
    array([[-1.73205081e+00,  5.77350269e-01,  5.77350269e-01],
           [0.00000000e+00, -1.63299316e+00,  8.16496581e-01],
           [0.00000000e+00,  0.00000000e+00, -1.41421356e+00],
           [0.00000000e+00,  0.00000000e+00,  6.93889390e-18],
           [0.00000000e+00,  0.00000000e+00,  1.11022302e-16],
           [0.00000000e+00,  0.00000000e+00,  1.11022302e-16]])

    When using householder to solve a linear least squares problem A @ x ~= b,
    b is provided, compute the reduced QR factorization of the resulting
    m * (n + 1) augmented matrix:

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> b = np.array([1237., 1941., 2417.,  711., 1177.,  475.])

    With economy size:

    >>> q, r, c_1 = householder(a, b, economy=True)
    >>> q
    array([[-5.77350269e-01, -2.04124145e-01, -3.53553391e-01],
           [ 0.00000000e+00, -6.12372436e-01, -3.53553391e-01],
           [ 0.00000000e+00,  0.00000000e+00, -7.07106781e-01],
           [ 5.77350269e-01, -4.08248290e-01, -1.80092860e-17],
           [ 5.77350269e-01,  2.04124145e-01, -3.53553391e-01],
           [ 0.00000000e+00,  6.12372436e-01, -3.53553391e-01]])
    >>> r
    array([[-1.73205081,  0.57735027,  0.57735027],
           [ 0.        , -1.63299316,  0.81649658],
           [ 0.        ,  0.        , -1.41421356]])
    >>> c_1
    array([  375.85502524, -1200.24997396, -3416.73996669])

    Rank Deficiency case:
    >>> a = np.array([[-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)

    """
    if b is not None:
        if explicit_q is False:
            warnings.warn(
                "Setting b will always explicitly compute Q"
            )
        explicit_q = True
        b = np.reshape(b, (1, b.shape[0]))
        a = np.concatenate((a, b.T), axis=1)
    m, n = a.shape
    if explicit_q:
        q = np.eye(m)

    for k in range(min(n, m)):  # loop over columns
        # compute Householder vector for current col
        alpha = -math.copysign(1, a[k, k]) * np.linalg.norm(a[k:, k], 2)
        v = np.hstack((np.zeros(k), a[k:, k]))
        v[k] -= alpha

        if explicit_q:
            h = np.eye(v.shape[0]) - 2 * np.outer(v, v) / np.dot(v, v)
            q = q @ h

        beta = np.dot(v, v)
        if beta == 0:  # skip current column if it's already zero
            continue
        for j in range(k, n):  # apply transformation to remaining submatrix
            gamma = np.dot(v, a[:, j])
            a[:, j] -= (2 * gamma / beta) * v
        # print(a)

    if b is not None:
        return (q[:, :n-1], a[:n-1, :-1], a[:n-1, -1]) if economy else (q, a[:, :-1], a[:n-1, -1])  # Q1, R, c_1
    else:
        if explicit_q:
            return (q[:, :n], a[:n, :]) if economy else (q, a)
        else:
            return a[:n, :] if economy else a


def classical_gram_schmidt_orthogonalization(a, b=None):
    """
    Classical Gram-Schmidt Orthogonalization.

    # TODO: Inplace q instead of using a
    # TODO: Rank Deficiency and column pivoting

    Parameters
    ----------
    a : (N, N) array_like
        Array to decompose

    b : (N,) array_like, default: None
        Solve a linear least squares problem A @ x ~= b. If b specified, not a in place algorithm.
        A new copy of Q is required.

    Returns
    -------
    Q : complex ndarray
        Q matrix in QR factorization.

    R : complex ndarray
        R matrix in QR factorization.

    C_1 : ndarray
        The transformed righthand side c_1 = Q_1.T @ b. Only when b is specified.


    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    A simple application of the classical Gram-Schmidt QR factorization is:

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> q, r = classical_gram_schmidt_orthogonalization(a)
    >>> q
    array([[ 5.77350269e-01,  2.04124145e-01,  3.53553391e-01],
           [ 0.00000000e+00,  6.12372436e-01,  3.53553391e-01],
           [ 0.00000000e+00,  0.00000000e+00,  7.07106781e-01],
           [-5.77350269e-01,  4.08248290e-01, -1.57009246e-16],
           [-5.77350269e-01, -2.04124145e-01,  3.53553391e-01],
           [ 0.00000000e+00, -6.12372436e-01,  3.53553391e-01]])
    >>> r
    array([[ 1.73205081, -0.57735027, -0.57735027],
           [ 0.        ,  1.63299316, -0.81649658],
           [ 0.        ,  0.        ,  1.41421356]])

    When using modifed Gram-Schmidt to solve a linear least squares problem A @ x = b,
    b is provided, compute the reduced QR factorization of the resulting
    m * (n + 1) augmented matrix:

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> b = np.array([1237., 1941., 2417.,  711., 1177.,  475.])
    >>> q, r, c_1 = classical_gram_schmidt_orthogonalization(a, b)
    >>> c_1
    array([-375.85502524, 1200.24997396, 3416.73996669])

    """
    if b is not None:
        b = np.reshape(b, (1, b.shape[0]))
        a = np.concatenate((a, b.T), axis=1)
    m, n = a.shape
    # q = np.zeros((m, n))
    r = np.zeros((n, n))
    for k in range(n):  # loop over columns
        # q[:, k] = a[:, k]
        for j in range(k):  # subtract from current column its components in preceding columns
            r[j, k] = np.dot(a[:, j], a[:, k])
            # r[k, j] = np.dot(a[:, k], a[:, j])
            a[:, k] -= r[j, k] * a[:, j]
        r[k, k] = np.linalg.norm(a[:, k], 2)
        if r[k, k] == 0:
            raise LinAlgError("A is not of full column rank.")  # stop if linearly dependent
        a[:, k] /= r[k, k]  # normalize current column

    if b is not None:
        return a[:, :-1], r[:-1, :-1], r[:-1, -1]  # Q1, R, c_1
    return a, r


def modified_gram_schmidt_orthogonalization(a, b=None):
    """
    Modified Gram-Schmidt Orthogonalization.

    # TODO: Inplace q instead of using a
    # TODO: Rank Deficiency and column pivoting

    Parameters
    ----------
    a : (N, N) array_like
        Array to decompose

    b : (N,) array_like, default: None
        solve a linear least squares problem A @ x ~= b. If b specified, not a in place algorithm.
        A new copy of Q is required.

    Returns
    -------
    Q : complex ndarray
        Q matrix in QR factorization.

    R : complex ndarray
        R matrix in QR factorization.

    C_1 : ndarray
        The transformed righthand side c_1 = Q_1.T @ b. Only when b is specified.

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    A simple application of the modified Gram-Schmidt QR factorization is:

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> q, r = modified_gram_schmidt_orthogonalization(a)
    >>> q
    array([[ 5.77350269e-01,  2.04124145e-01,  3.53553391e-01],
           [ 0.00000000e+00,  6.12372436e-01,  3.53553391e-01],
           [ 0.00000000e+00,  0.00000000e+00,  7.07106781e-01],
           [-5.77350269e-01,  4.08248290e-01, -1.57009246e-16],
           [-5.77350269e-01, -2.04124145e-01,  3.53553391e-01],
           [ 0.00000000e+00, -6.12372436e-01,  3.53553391e-01]])
    >>> r
    array([[ 1.73205081, -0.57735027, -0.57735027],
           [ 0.        ,  1.63299316, -0.81649658],
           [ 0.        ,  0.        ,  1.41421356]])

    When using modifed Gram-Schmidt to solve a linear least squares problem A @ x ~= b,
    b is provided, compute the reduced QR factorization of the resulting
    m * (n + 1) augmented matrix:

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> b = np.array([1237., 1941., 2417.,  711., 1177.,  475.])
    >>> q, r, c_1 = modified_gram_schmidt_orthogonalization(a, b)
    >>> c_1
    array([-375.85502524, 1200.24997396, 3416.73996669])

    """
    if b is not None:
        b = np.reshape(b, (1, b.shape[0]))
        a = np.concatenate((a, b.T), axis=1)
    m, n = a.shape
    r = np.zeros((n, n))
    for k in range(n):  # loop over columns
        r[k, k] = np.linalg.norm(a[:, k], 2)
        if r[k, k] == 0:
            raise LinAlgError("A is not of full column rank.")  # stop if linearly dependent
        a[:, k] = a[:, k] / r[k, k]  # normalize current column
        for j in range(k+1, n):
            r[k, j] = np.dot(a[:, k], a[:, j])  # subtract from succeeding columns their components in current column
            a[:, j] -= r[k, j] * a[:, k]

    if b is not None:
        return a[:, :-1], r[:-1, :-1], r[:-1, -1]  # Q1, R, c_1
    return a, r


def solve_linear_least_square(a, b, method="householder"):
    """Solving linear least squares problems.

    # TODO: Check overdetermined m > n?
    # TODO: return  residual : complex ndarray. Minimum residual norm squared
    # TODO: refactor method option
    # TODO: suppress warnings in householder here

    Parameters
    ----------
    a : (M, N) array_like
        m * n matrix with m > n

    b : (M,) array_like
        m-vector

    method : {'householder', 'classical_gram_schmidt', 'modified_gram_schmidt', None}, default='householder'
        Orthogonalization Methods.  Should be one of

            - 'householder'            :ref:`(see here) <linalg.qr_factorization-householder>`
            - 'classical_gram_schmidt' :ref:`(see here) <linalg.qr_factorization-classical_gram_schmidt_orthogonalization>`
            - 'modified_gram_schmidt'  :ref:`(see here) <linalg.qr_factorization-modified_gram_schmidt_orthogonalization>`

        If not given, chosen to be householder.

    Returns
    -------
    x : (N,) array_like
        Solution to this linear least squares problems

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.

    Examples
    --------
    A simple application of solving linear least square problem is:

    >>> a = np.array([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 1],
    ...               [-1, 1, 0],
    ...               [-1, 0, 1],
    ...               [0, -1, 1]], dtype=float)
    >>> b = np.array([1237., 1941., 2417.,  711., 1177.,  475.])
    >>> x = solve_linear_least_square(a, b)
    >>> x
    array([1236., 1943., 2416.])

    """
    if method == "classical_gram_schmidt":
        method = classical_gram_schmidt_orthogonalization
    elif method == "modified_gram_schmidt":
        method = modified_gram_schmidt_orthogonalization
    else:
        method = partial(householder, economy=True)

    # m, n = a.shape
    # q, r = householder(a, explicit_q=True)
    q, r, c_1 = method(a, b)
    # c_1 = (q.T @ b)[:n]
    # c_2 = (q.T @ b)[n:]
    # backward_substitution(r[:n, :], c_1)
    backward_substitution(r, c_1)
    # return c_1, np.linalg.norm(c_2) ** 2
    return c_1


def polynomial_fitting(x, y, deg=2):
    """Solving linear least squares problems.

    # TODO: Check overdetermined m > n?
    # TODO: implement my own numpy.polynomial.polynomial.polyvander instead
    # TODO: return  residual : complex ndarray. Minimum residual norm squared
    # TODO: suppress warnings in householder here

    Parameters
    ----------
    x : array_like with shape (n,)
        Given data points (x_i, y_i), i = 1,...,n

    y : array_like with shape (n,)
        Given data points (x_i, y_i), i = 1,...,n

    deg : int, default: 2
        Degrees of the polynomial

    Returns
    -------
    coef : (deg,) array_like
        Coefficients of the polynomial

    residual : complex ndarray
        Minimum residual norm squared

    References
    ----------
    .. [1] Heath, Michael T., 2018. Scientific Computing, Revised Second Edition
    , Society for Industrial and Applied Mathematics.
    .. [2] Cormen, T.H., Leiserson, C.E., Rivest, R.L., Stein, C., 2009. Introduction
    to Algorithms, Third Edition. 3rd ed., The MIT Press.

    Examples
    --------
    A simple application of polynomial fitting problem is:

    >>> x = np.linspace(0.0, 10.0, 21)
    >>> y = np.array([2.9, 2.7, 4.8, 5.3, 7.1, 7.6, 7.7, 7.6, 9.4,
                      9.0, 9.6, 10.0, 10.2, 9.7, 8.3, 8.4, 9.0, 8.3,
                      6.6, 6.7, 4.1])
    >>> coef = polynomial_fitting(x, y, deg=2)
    >>> coef
    array([ 2.17571993,  2.67041339, -0.23844394])

    Another example from Cormen book:

    >>> x = np.array([-1., 1., 2., 3., 5.])
    >>> y = np.array([2., 1., 1., 0., 3.])
    >>> coef = polynomial_fitting(x, y, deg=2)
    >>> coef
    array([ 1.2       , -0.75714286,  0.21428571])

    Degree one, fit line, example 3.21 from Heath:

    >>> x = np.array([-2, -1, 3], dtype=float)
    >>> y = np.array([-1, 3, -2], dtype=float)
    >>> coef = polynomial_fitting(x, y, deg=1)
    >>> coef
    array([-0. , -0.5])

    """
    from numpy.polynomial import polynomial as P

    a = P.polyvander(x=x, deg=deg)
    # coef, residue = solve_linear_least_square(a, y)
    coef = solve_linear_least_square(a, y)
    # return coef, residue
    return coef

def real_givens_rotations_without_scaling(f, g):
    fg2 = f**2 + g**2
    r = math.sqrt(fg2)
    rr = 1 / r
    c = abs(f) * rr
    s = g * rr
    if f < 0:
        s = -s
        r = -r
    return s, c


def real_givens_rotations_with_scaling(f, g):
    # TODO: implement
    # scale = max(abs(f), abs(g))
    # if scale > z**2:
    #     pass
    pass
