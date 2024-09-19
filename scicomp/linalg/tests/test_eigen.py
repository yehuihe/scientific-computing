import pytest

import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_, assert_allclose)

from scicomp.linalg import (power_iteration, normalized_power_iteration, inverse_iteration,
                            rayleigh_quotient_iteration, orthogonal_iteration, qr_iteration,
                            qr_iteration_with_shifts)


np.set_printoptions(precision=4)


class TestEigen:
    def test_power_iteration(self):
        A = np.array([[3., 1.],
                      [1., 3.]])
        # test power iteration with given initial x0
        x0 = np.array([0., 1.])
        eigenvector, eigenvalue = power_iteration(A, num_iterations=9, x0=x0, verbose=False)
        exact_eigenvector = np.array([130816., 131328.])
        exact_eigenvalue = 3.992
        assert_array_almost_equal(eigenvector, exact_eigenvector)
        assert_allclose(eigenvalue, exact_eigenvalue, atol=1e-3)

        # TODO: set seeds for random inital x0
        # # power iteration arbitrary nonzero vector
        # eigenvector, eigenvalue = power_iteration(A, num_iterations=9, verbose=False)
        # exact_eigenvector = np.array([205332.2278, 205454.7083])
        # exact_eigenvalue = 3.998
        # assert_array_almost_equal(eigenvector, exact_eigenvector)
        # assert_allclose(eigenvalue, exact_eigenvalue, atol=1e-3)

    def test_normalized_power_iteration(self):
        A = np.array([[3., 1.],
                      [1., 3.]])
        # test normalized power iteration with given initial x0
        x0 = np.array([0., 1.])
        eigenvector, eigenvalue = normalized_power_iteration(A, num_iterations=9, x0=x0, verbose=False)
        exact_eigenvector = np.array([0.996101, 1.])
        exact_eigenvalue = 3.992
        assert_array_almost_equal(eigenvector, exact_eigenvector)
        assert_allclose(eigenvalue, exact_eigenvalue, atol=1e-3)
