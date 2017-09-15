# -*- encoding: utf-8 -*-

import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
import numpy as np

import pbvi

#                s0   s1
cT = np.array([[            # s0
                [0.9, 0.1],     # as
                [0.1, 0.9]],    # ag
               [            # s1
                [0.1, 0.9],     # as
                [0.9, 0.1]]])   # ag

cR = np.array([[0.0, 1.0],
               [0.0, 1.0]])

cO = np.array([[[0.6, 0.4],     # s0 o0 o1
                [0.4, 0.6]],
               [[0.6, 0.4],     # s0 o0 o1
                [0.4, 0.6]]])   # s1

gamma = 1.0


apbvi = pbvi.PBVI(cR, cT, cO, gamma)

V = np.zeros((1, 2), np.float64)

B = np.array([[0.4, 0.6],
              [0.6, 0.4]])

for _ in xrange(8):
    gamma   = apbvi.gamma(V)
    epsilon = apbvi.epsilon(B, gamma)
    V       = apbvi.V(epsilon, B)


print V
