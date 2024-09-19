import pytest

from math import sin, cos
import numpy as np

from scicomp.nonlinear import _root_finding as zeros


# A few test functions used frequently:
# # A simple quadratic, (x-1)^2 - 1
def f1(x):
    return x ** 2 - 4 * sin(x)


def f1_prime(x):
    return 2*x - 4*cos(x)


class TestRootFinding:
    def test_interval_bisection(self):
        a = 1.
        b = 3.
        zero = zeros.interval_bisection(f1, a, b, tol=zeros._tol, num_iterations=zeros._nmax, verbose=False)
        assert 1.933753 == pytest.approx(zero, abs=1e-6)

    def test_newtons_method(self):
        x0 = 3.
        zero = zeros.newtons_method(f1, f1_prime, x0, tol=zeros._tol, num_iterations=zeros._nmax, verbose=False)
        assert 1.933753 == pytest.approx(zero, abs=1e-6)

    def test_secant_method(self):
        x0 = 1.
        x1 = 3.
        zero = zeros.secant_method(f1, x0, x1, tol=zeros._tol, num_iterations=zeros._nmax, verbose=False)
        assert 1.933753 == pytest.approx(zero, abs=1e-6)

