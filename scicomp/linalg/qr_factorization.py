import math

import numpy as np


def householder(a):
    m, n = a.shape
    for k in range(min(n, m)):  # loop over columns
        # compute Householder vector for current col
        alpha = -math.copysign(1, a[k, k]) * np.linalg.norm(a[k:, k], 2)
        v = np.hstack((np.zeros(k), a[k:, k]))
        v[k] -= alpha

        beta = np.dot(v, v)
        if beta == 0:  # skip current column if it's already zero
            continue
        for j in range(k, n):  # apply transformation to remaining submatrix
            gamma = np.dot(v, a[:, j])
            a[:, j] -= (2 * gamma / beta) * v
        # print(a)

    return a
