import numpy as np
from _linalg import *
from _qr_factorization import householder
from _eigen import (
    power_iteration,
    normalized_power_iteration,
    inverse_iteration,
    rayleigh_quotient_iteration,
    orthogonal_iteration,
    qr_iteration,
    qr_iteration_with_shifts,
)

np.set_printoptions(precision=4)
# backward_substitution unit tests
# n = 5
#
# A = np.random.randn(n, n) * np.tri(n).T
# print(A)
#
# x = np.random.randn(n)
# print(x)
#
# b = A @ x
#
# # A = np.array([[1, 2, 2], [0, -4, -6], [0, 0, -1]])
# # b = np.array([3, -6, 1])
#
# x = np.linalg.solve(A, b)
# print(np.allclose(np.dot(A, x), b))
#
# print(backward_substitution(A, b))
#
# print(x)
#

# # householder unit tests
# A = np.array([[1., 0., 0.],
#               [0., 1., 0.],
#               [0., 0., 1.],
#               [-1., 1., 0.],
#               [-1., 0., 1.],
#               [0., -1., 1.]])
#
# b = np.array([[1237., 1941., 2417., 711., 1177., 475.]])
#
# print(householder(A))

# A = np.array(
#     [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 3.0, 9.0], [1.0, 4.0, 16.0]]
# )

# b = np.array([[1237., 1941., 2417., 711., 1177., 475.]])

# print(householder(A))


#
#
# eigen normalized_power_iteration unit tests
# A = np.array([[3.0, 1.0], [1.0, 3.0]])
#
# x0 = np.array([0.0, 1.0])
# print('\npower iteration: ')
# print(power_iteration(A, num_iterations=9, x0=x0, verbose=True))
# print('\npower iteration arbitrary nonzero vector: ')
# print(power_iteration(A, num_iterations=9, verbose=True))
#
# print("\nnormalized power iteration: ")
# print(normalized_power_iteration(A, num_iterations=9, x0=x0, verbose=True))
# print("\nnormalized power iteration arbitrary nonzero vector: ")
# print(normalized_power_iteration(A, num_iterations=9, verbose=True))
#
# print('\ninverse iteration: ')
# print(inverse_iteration(A, num_iterations=9, x0=x0, verbose=True))
# print('\ninverse iteration arbitrary nonzero vector: ')
# print(inverse_iteration(A, num_iterations=9, verbose=True))
#
# print('\nrayleigh quotient iteration: ')
# x0 = np.array([0.807, 0.397])
# print(rayleigh_quotient_iteration(A, num_iterations=3, x0=x0, verbose=True))
# print('\nrayleigh quotient iteration arbitrary nonzero vector: ')
# print(rayleigh_quotient_iteration(A, num_iterations=3, verbose=True))
#
# print('\northogonal iteration: ')
# n = 5
# np.random.seed(70)
# eigvecs = np.random.randn(n, n)
# eigvals = np.sort(np.random.randn(n))
#
# A = np.dot(np.linalg.solve(eigvecs, np.diag(eigvals)), eigvecs)
# print(eigvals)
#
# x0 = np.random.randn(n, n)
#
# print(orthogonal_iteration(A, num_iterations=15, x0=x0, verbose=False))
#
# print('\nqr iteration: ')
#
# A = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
#               [0.3945, 2.7328, -0.3097, 0.1129],
#               [0.4198, -0.3097, 2.5675, 0.6079],
#               [1.1159, 0.1129, 0.6079, 1.7231]])
#
# eigenvalues, _ = np.linalg.eig(A)
# print(eigenvalues)
# print(qr_iteration(A, num_iterations=3, verbose=True))
#
# # print('\nqr iteration with shifts: ')
# #
# # A = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
# #               [0.3945, 2.7328, -0.3097, 0.1129],
# #               [0.4198, -0.3097, 2.5675, 0.6079],
# #               [1.1159, 0.1129, 0.6079, 1.7231]])
# #
# # print(qr_iteration_with_shifts(A, num_iterations=3, verbose=True))
#
# print('\nqr iteration test #2: ')
# A = np.array([[0, 2],
#               [2, 3]])
#
# print(qr_iteration(A, num_iterations=20, verbose=True))

# print('\nqr iteration with shifts: ')
#
# A = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
#               [0.3945, 2.7328, -0.3097, 0.1129],
#               [0.4198, -0.3097, 2.5675, 0.6079],
#               [1.1159, 0.1129, 0.6079, 1.7231]])
#
# print(qr_iteration_with_shifts(A, num_iterations=3, shift='rayleigh', verbose=True))

# lu_factorization unit tests

A = np.array([[1, 2, 2],
              [4, 4, 2],
              [4, 6, 4]], dtype=float)

b = np.array([3, 6, 10], dtype=float)

lu_factorization(A)
print(A)

# lu_factorization_with_partial_pivoting unit tests

A = np.array([[1, 2, 2],
              [4, 4, 2],
              [4, 6, 4]], dtype=float)

b = np.array([3, 6, 10], dtype=float)

lu_factorization_with_partial_pivoting(A)
print(A)

