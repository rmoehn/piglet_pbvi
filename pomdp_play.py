# -*- encoding: utf-8 -*-

import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
import numpy as np

import pbvi
import naive_pbvi

#                s0   s1
cT = np.array([[            # s0
                [0.9, 0.1],     # as
                [0.1, 0.9]],    # ag
               [            # s1
                [0.1, 0.9],     # as
                [0.9, 0.1]]])   # ag

#               as   ag
cR = np.array([[0.0, 0.0],      # s0
               [1.0, 1.0]])     # s1

#                o0   o1
cO = np.array([[            # a0
                [0.6, 0.4],     # s' = s0
                [0.4, 0.6]],    #      s1
               [            # a1
                [0.6, 0.4],     #      s0
                [0.4, 0.6]]])   #      s1

discount_gamma = 1.0


apbvi = pbvi.PBVI(cR, cT, cO, discount_gamma)

V = np.zeros((1, 2), np.float64)

B = np.array([[0.4, 0.6],
              [0.6, 0.4],
              [0.2, 0.8],
              [0.8, 0.2]])

for _ in xrange(2):
    gamma   = apbvi.gamma(V)
    epsilon = apbvi.epsilon(B, gamma)
    V       = apbvi.V(epsilon, B)
    print V


anpbvi = naive_pbvi.NaivePBVI(cR, cT, cO, discount_gamma)

#nV = np.zeros((1,2), np.float64)
nV = np.array([[0., 1.]])

for _ in xrange(2):
    ngamma = anpbvi.gamma(nV)
    nepsilon = anpbvi.epsilon(B, ngamma)
    nV = anpbvi.V(nepsilon, B)
    print nV
