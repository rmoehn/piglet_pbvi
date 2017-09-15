import collections

import numpy as np


def _gamma_ast(R):
    return R.T.copy()
    # Copy in order to reorder in memory.


def _tau(T):
    """
    Reorder the transition probability array for performance.

    In::

        |S| x |A| x |S|
        s     a     s'

    Out::

        |A| x 1 x 1 x |S| x |S|  (1 for newaxis)
        a             s'    s
    """
    return np.moveaxis(T, [0,1,2], [2,0,1])[:,None,None,:].copy()


class PBVI(object):
    def __init__(self, R, T, omega, discount_gamma):
        self.gamma_ast      = _gamma_ast(R)
        self.omega          = omega[:,:,None].copy()
        self.discount_gamma = discount_gamma
        self.tau            = _tau(T)
        self._outs          = collections.defaultdict(dict)


    def gamma(self, V_):
        prod1 = np.multiply(self.omega[:,:,None], V_,
                            out=self._outs['gamma'].get('prod1'))
        prod2 = np.multiply(self.tau, prod1,
                            out=self._outs['gamma'].get('prod2'))
        sum_s_ = np.sum(prod2, -1, out=self._outs['gamma'].get('sum_s_'))
        return np.multiply(self.discount_gamma, sum_s_,
                           out=self._outs['gamma'].get('result'))
