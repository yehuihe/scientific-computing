"""Eigenvalue problems."""

import math

import numpy as np

# Local imports


__all__ = [
    "power_iteration",
    "normalized_power_iteration",
    "inverse_iteration",
    "rayleigh_quotient_iteration",
    "orthogonal_iteration",
    "qr_iteration",
    "qr_iteration_with_shifts",
]


def power_iteration(a, num_iterations, x0=None, verbose=False):
    # Arbitrary nonzero vector.
    x = np.random.rand(a.shape[1]) if x0 is None else x0
    prev_norm = np.linalg.norm(x, ord=np.inf)
    ratio = 0
    for k in range(num_iterations):  # Generate next vector.
        x = a @ x
        norm = np.linalg.norm(x, ord=np.inf)
        ratio = norm / prev_norm
        prev_norm = norm
        # TODO: pretty format
        if verbose:
            # print(0, x0, )
            print(k + 1, x, ratio)
    return x, ratio


def normalized_power_iteration(a, num_iterations, x0=None, verbose=False):
    # Arbitrary nonzero vector.
    x = np.random.rand(a.shape[1]) if x0 is None else x0
    norm = 0
    for k in range(num_iterations):
        x = a @ x
        norm = np.linalg.norm(x, ord=np.inf)
        x = x / norm
        # TODO: pretty format
        if verbose:
            # print(0, x0, )
            print(k + 1, x, norm)
    return x, norm


def inverse_iteration(a, num_iterations, x0=None, verbose=False):
    # Arbitrary nonzero vector.
    x = np.random.rand(a.shape[1]) if x0 is None else x0
    norm = 0
    for k in range(num_iterations):
        # TODO: change to my solver
        x = np.linalg.solve(a, x)
        norm = np.linalg.norm(x, ord=np.inf)
        x = x / norm
        # TODO: pretty format
        if verbose:
            print(k + 1, x, norm)
    return x, norm


def rayleigh_quotient_iteration(a, num_iterations, x0=None, verbose=False):
    # TODO: figure out iteration number,
    #  there is singluar matrix error if exceeding converge iter number
    # Arbitrary nonzero vector.
    x = np.random.rand(a.shape[1]) if x0 is None else x0
    sigma = 0
    for k in range(num_iterations):
        sigma = np.dot(x, np.dot(a, x)) / np.dot(x, x)
        # TODO: change to my solver
        x = np.linalg.solve(a - sigma * np.eye(a.shape[0]), x)
        norm = np.linalg.norm(x, ord=np.inf)
        x = x / norm
        # TODO: pretty format
        if verbose:
            print(k + 1, x, sigma)
    return x, sigma


def orthogonal_iteration(a, num_iterations, x0, verbose=False):
    # TODO: return actual eigenvalues instead of r
    x = x0
    r = 0
    for k in range(num_iterations):
        # TODO: change to my qr decomposition
        q, r = np.linalg.qr(x)  # normalize
        x = a @ q  # generate next matrix
        # # TODO: pretty format
        # if verbose:
        #     print(k + 1, x, sigma)
    return r


def qr_iteration(a, num_iterations, verbose=False):
    for k in range(num_iterations):
        # TODO: change to my qr decomposition
        q, r = np.linalg.qr(a)  # normalize
        a = r @ q  # generate next matrix
        # # TODO: pretty format
        if verbose:
            print(k + 1, a)

    return a


def _wilkinson(a):
    n = a.shape[0]

    delta = (a[n - 2, n - 2] - a[n - 1, n - 1]) / 2
    sign = math.copysign(1, delta) if delta != 0 else 1
    mu = a[n - 1, n - 1] - sign * a[n - 2, n - 1] ** 2 / (
        abs(delta) + math.sqrt(delta**2 + a[n - 2, n - 1] ** 2)
    )

    return mu


def qr_iteration_with_shifts(
    a, num_iterations, shift="rayleigh", verbose=False
):
    n = a.shape[0]
    for k in range(num_iterations):
        sigma = 0
        if shift == "rayleigh":  # choose shift sigma
            sigma = a[n - 1, n - 1]
        elif shift == "wilkinson":
            sigma = _wilkinson(a)
        # TODO: change to my qr decomposition
        q, r = np.linalg.qr(a - sigma * np.eye(n))  # normalize
        a = r @ q + sigma * np.eye(n)  # generate next matrix
        # # TODO: pretty format
        if verbose:
            print(k + 1, a, sigma)

    return a
