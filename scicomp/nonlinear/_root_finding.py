"""Nonlinear equations and root finding."""

import math

import numpy as np

# Local imports


_nmax = 100
_tol = 2e-12


__all__ = ["interval_bisection", "newtons_method", "secant_method"]


def interval_bisection(f, a, b, tol=_tol, num_iterations=_nmax, verbose=False):
    # Check if a and b bound a root.
    if math.copysign(1, f(a)) == math.copysign(1, f(b)):
        # TODO: make my own exception
        raise Exception(
            "The scalars a: %f and b: %f do not bound a root" % (a, b)
        )

    k = 1
    while k < num_iterations:
        # while (b - a) > tol:
        m = a + (b - a) / 2.0
        if (b - a) < tol:
            return m
        if verbose:
            print(k, a, f(a), b, f(b), f(m))
        if math.copysign(1, f(a)) == math.copysign(1, f(m)):
            a = m
        else:
            b = m
        k += 1
    # TODO: raise exception "Method failed. after max iterations"
    print("Method failed.")


def newtons_method(
    f, fprime, x0, tol=_tol, num_iterations=_nmax, verbose=False
):
    x = x0
    for k in range(num_iterations):
        h = -f(x) / fprime(x)
        # Stopping criterion:
        # The iteration can be terminated when
        # |h_k|/|x_k| or |f(x_k)|, or both, are as small as desired.
        if abs(h) / abs(x) < tol and abs(f(x)) < tol:
            return x
        x = x + h
        if verbose:
            print(k + 1, x, f(x), fprime(x), h)
    # TODO: raise exception "Method failed. after max iterations"
    print("Method failed.")


def secant_method(f, x0, x1, tol=_tol, num_iterations=_nmax, verbose=False):
    x_prev = x0
    x = x1
    for k in range(num_iterations):
        h = -(f(x) * (x - x_prev)) / (f(x) - f(x_prev))
        if abs(h) / abs(x) < tol and abs(f(x)) < tol:
            return x
        x_prev = x
        x = x + h
        if verbose:
            print(k + 1, x, f(x), h)
    # TODO: raise exception "Method failed. after max iterations"
    print("Method failed.")
