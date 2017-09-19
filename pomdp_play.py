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

b1 = np.linspace(0.1, 0.9, 8)
B = np.stack([1 - b1, b1], axis=-1)

for _ in xrange(9):
    gamma   = apbvi.gamma(V)
    epsilon = apbvi.epsilon(B, gamma)
    V       = apbvi.V(epsilon, B)
    print V

fig, ax = plt.subplots()
for v in V:
    ax.plot([0, 1], v)
plt.show()


anpbvi = naive_pbvi.NaivePBVI(cR, cT, cO, discount_gamma)

nV = np.zeros((1, 2), np.float64)

for _ in xrange(0):
    ngamma = anpbvi.gamma(nV)
    nepsilon = anpbvi.epsilon(B, ngamma)
    nV = anpbvi.V(nepsilon, B)

print "\nV\n", nV

for i in xrange(0):
    print "\n== Round %d ==" % i
    print "GAMMA"
    ngamma = anpbvi.gamma(nV)
    print ngamma
    gamma = apbvi.gamma(V)
    print gamma

    print "EPSILON"
    nepsilon = anpbvi.epsilon(B, ngamma)
    print nepsilon
    epsilon = apbvi.epsilon(B, gamma)
    print epsilon

    print "V"
    nV = anpbvi.V(nepsilon, B)
    print nV
    V = apbvi.V(epsilon, B)
    print V
