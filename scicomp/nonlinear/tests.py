from math import sin, cos
import numpy as np

from scicomp.nonlinear import _root_finding as zeros
from scicomp.nonlinear import _systems_root_finding as nzeros

np.set_printoptions(precision=6)


# A few test functions used frequently:
# # A simple quadratic, (x-1)^2 - 1
def f1(x):
    return x**2 - 4 * sin(x)


def f1_prime(x):
    return 2 * x - 4 * cos(x)


def f2(x):
    return x**2 - 1


def f2_prime(x):
    return 2 * x


def f3(x):
    return x**2 - 2 * x + 1


def f3_prime(x):
    return 2 * x - 2


def fn1(x):
    x1, x2 = x
    return np.array([x1 + 2 * x2 - 2, x1**2 + 4 * x2**2 - 4])


def fn1_jacobian(x):
    x1, x2 = x
    return np.array([[1, 2], [2 * x1, 8 * x2]])


# # Example 5.7 Interval Bisection.
# a = 1
# b = 3
# # root = zeros.interval_bisection(f1, a, b, tol=zeros._tol, num_iterations=zeros._nmax, verbose=True)
# root = zeros.interval_bisection(f1, a, b, tol=zeros._tol, num_iterations=23, verbose=True)
# print(root)
#
# # Example 5.10 Newton's Method.
# # x0 = 3
# # root = zeros.newtons_method(f1, f1_prime, x0, tol=zeros._tol, num_iterations=5, verbose=True)
# # print(root)
#
# # Example 5.11 Newton's Method for Multiple Root.
# # x0 = 2
# # root = zeros.newtons_method(f2, f2_prime, x0, tol=zeros._tol, num_iterations=5, verbose=True)
# # root = zeros.newtons_method(f3, f3_prime, x0, tol=zeros._tol, num_iterations=5, verbose=True)
#
# # Example 5.12 Secant Method.
# print('\nSecant Method.: ')
# x0 = 1.
# x1 = 3.
# root = zeros.secant_method(f1, x0, x1, tol=zeros._tol, num_iterations=7, verbose=True)
# print(root)


# Example 5.15 Newton's Method.
print('\nExample 5.15 Newton"s Method.: ')
x0 = np.array([1.0, 2.0])
root = nzeros.ndim_newtons_method(
    fn1, fn1_jacobian, x0, tol=nzeros._tol, num_iterations=2, verbose=True
)
print(root)
