"""Systems of Nonlinear equations and root finding."""

import math

import numpy as np

# Local imports


_nmax = 100
_tol = 2e-12


__all__ = ["ndim_newtons_method"]


def ndim_newtons_method(
    f, f_jacobian, x0, tol=_tol, num_iterations=_nmax, verbose=False
):
    x = x0
    for k in range(num_iterations):
        # h = - f(x) / fprime(x)
        s = np.linalg.solve(f_jacobian(x), -f(x))  # Compute Newton step.
        # Stopping criterion:
        # The iteration can be terminated when
        # ||f(x_k)|| is as small as desired.
        if np.linalg.norm(f(x)) < tol:
            return x
        x = x + s
        if verbose:
            print(k + 1, x, f(x), f_jacobian(x), s)
    # TODO: raise exception "Method failed. after max iterations"
    print("Method failed.")


# def broydens_method(f, f_jacobian, x0, tol=_tol, num_iterations=_nmax, verbose=False):
