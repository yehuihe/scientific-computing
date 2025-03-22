"""Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
"""

_nmax = 100
_tol = 2e-12

__all__ = ["jacobi", "gauss_seidel", "sor"]

from functools import wraps
import time

import numpy as np
from memory_profiler import profile


# from numpy.linalg import LinAlgError


# TODO 1: warning after num_iter has been reached, not yet converge
# TODO 2: abstract stopping criterian
# TODO 3: pretty print/verbose
# TODO 4: Parallel algorithms


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn:" + fn.__name__ + " took " + str(t2 - t1) + " seconds")
        return result
    return measure_time


@timefn
def jacobi(a, b, x0, tol=_tol, num_iterations=_nmax):
    x_k = x0.copy()
    x_k_1 = np.zeros_like(x0)

    k = 0
    while k < num_iterations:
        for i in range(a.shape[0]):
            sigma = 0
            for j in range(a.shape[0]):
                if j != i:
                    sigma += a[i, j] * x_k[j]
            x_k_1[i] = (b[i] - sigma) / a[i, i]

        # error = np.linalg.norm(x_k_1 - x_k)
        relative_change = np.linalg.norm(x_k_1 - x_k) / np.linalg.norm(x_k)
        if relative_change <= tol:
            break

        x_k = x_k_1.copy()

        k += 1
        print('iteration: ' + str(k) + '\t' + np.array2string(x_k_1, precision=3))

    return x_k_1


@timefn
def gauss_seidel(a, b, x0, tol=_tol, num_iterations=_nmax):
    x_k = x0.copy()

    k = 0
    while k < num_iterations:
        # prev_norm = np.linalg.norm(x_k)
        for i in range(a.shape[0]):
            sigma = 0
            for j in range(a.shape[0]):
                if j != i:
                    sigma += a[i, j] * x_k[j]
            x_k[i] = (b[i] - sigma) / a[i, i]

        residual = np.linalg.norm(a @ x_k - b)
        if residual <= tol:
            break
        # relative_change = np.linalg.norm(x_k - x_k) / np.linalg.norm(x_k)
        # if relative_change <= tol:
        #     break

        k += 1
        print('iteration: ' + str(k) + '\t' + np.array2string(x_k, precision=3))

    return x_k


@timefn
def sor(a, b, x0, omega, tol=_tol, num_iterations=_nmax):
    x_k = x0.copy()

    k = 0
    while k < num_iterations:
        for i in range(a.shape[0]):
            sigma = 0
            for j in range(a.shape[0]):
                if j != i:
                    sigma += a[i, j] * x_k[j]
            x_k[i] = (1 - omega) * x_k[i] + omega * (b[i] - sigma) / a[i, i]

        residual = np.linalg.norm(a @ x_k - b)
        if residual <= tol:
            break
        # relative_change = np.linalg.norm(x_k - x_k) / np.linalg.norm(x_k)
        # if relative_change <= tol:
        #     break

        k += 1
        print('iteration: ' + str(k) + '\t' + np.array2string(x_k, precision=3))

    return x_k


if __name__ == "__main__":
    A = np.array([[4., -1., -1., 0.],
                  [-1., 4., 0., -1.],
                  [-1., 0., 4., -1.],
                  [0., -1., -1., 4.]], dtype=np.float64)
    b = np.array([0., 0., 1., 1.], dtype=np.float64)
    x0 = np.array([0., 0., 0., 0.], dtype=np.float64)
    print("---jacobi---")
    print(jacobi(A, b, x0, num_iter=9))
    print("---gauss_seidel---")
    print(gauss_seidel(A, b, x0, num_iter=6))
    print("---sor---")
    print(sor(A, b, x0, omega=1.072, num_iter=5))
