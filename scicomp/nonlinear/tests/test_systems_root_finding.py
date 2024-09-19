import pytest

from math import sin, cos
import numpy as np

from scicomp.nonlinear import _systems_root_finding as zeros


def f1(x):
    x1, x2 = x
    return np.array([x1 + 2*x2 - 2,
                     x1**2 + 4 * x2**2 - 4]
                    )


def f1_jacobian(x):
    x1, x2 = x
    return np.array([[1, 2],
                     [2*x1, 8*x2]]
                    )


class TestRootFinding:
    def test_newtons_method(self):
        x0 = np.array([1., 2.])
        zero = zeros.ndim_newtons_method(f1, f1_jacobian, x0, tol=zeros._tol, num_iterations=zeros._nmax, verbose=False)
        # TODO: assert numpy array
        # assert 1.933753 == pytest.approx(zero, abs=1e-6)

