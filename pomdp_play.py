# -*- encoding: utf-8 -*-

import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
import numpy as np

#                s0   s1
cT = np.array([[            # s0
                [0.9, 0.1],     # as
                [0.1, 0.9]],    # ag
               [            # s1
                [0.1, 0.9],     # as
                [0.9, 0.1]]])   # ag

cR = np.array([0.0, 1.0])

cO = np.array([[0.6, 0.4],      # s0 o0 o1
               [0.4, 0.6]])     # s1

gamma = 1.0


def params_for_plan(T, R, p):
    alpha_p = R + np.dot(T[:, p, :], R)  # [dot(T[0,p,:],R), …, dot(T[n,p,:],R)]
    return alpha_p[1] - alpha_p[0], alpha_p[0]


m0, n0 = params_for_plan(cT, cR, 0)
m1, n1 = params_for_plan(cT, cR, 1)

print m0, n0, m1, n1
plt.plot([0.0, 1.0], [n0, m0+n0], 'b', [0.0, 1.0], [n1, m1+n1], 'g')
plt.show()


def params_for_plan_d(T, R, O, p0, d):
    if d < 1:
        raise AssertionError("Should not be reached.")
    if d == 1:
        return params_for_plan(T, R, p0)


def gamma_b_a(gamma_ao, gamma_aast, A, B):
    return gamma_aast


def new_V(agamma_b_a, A, B):
    return {max(A, lambda a, b: np.dot(agamma_b_a[b][a], b)) for b in B}


E = np.array([[[1,3], [2,4]],
              [[5,7], [6,8]],
              [[9,11], [10,12]]])

B = np.array([[1,2],[3,4],[5,6]])


# alpha · b for all a, o, b
crossprods = np.matmul(B, np.swapaxes(gamma, -1, -2))
best_alpha_inds = np.argmax(crossprods, -1)

(n_as, n_os) = gamma.shape[0:1]

best_per_o = gamma[np.arange(n_as)[:,None,None],
                   np.arange(n_os)[None,:,None],
                   best_alpha_inds]

# Gamma_a_b for each a and b
E = np.sum(best_per_o, 1)

values = np.squeeze( np.matmul(E, B[:,:,None]) )

best_as = np.argmax(values, axis=1)

V = E[np.arange(E.shape[0]), best_as]
print V

# https://stackoverflow.com/questions/40357335/numpy-how-to-get-a-max-from-an-argmax-result
