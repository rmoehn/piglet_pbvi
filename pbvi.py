# -*- encoding: utf-8 -*-

import collections

import numpy as np


def _gamma_ast(R):
    return R.T[:,None,:].copy()
    # Copy in order to reorder in memory.


def _tau(T):
    """
    Reorder the transition probability array for performance.

    In::

        |S| x |A| x |S|
         s     a     s'

    Out::

        |A| x  1  x  1  x |S| x |S|  (1 for newaxis)
         a                 s'    s
    """
    return np.moveaxis(T, [0,1,2], [2,0,1])[:,None,None,:].copy()


def _omega(omega):
    """
    Reorder the observation probability array for performance.

    In::

        |A| x |S| x |O|
         a     s'    o

    Out::

        |A| x |O| x  1  x |S|
         a     o           s'
    """
    return np.swapaxes(omega, -2, -1)[:,:,None].copy()


# TODO: Use different names for external and internal omega. (RM 2017-09-22)
# Note: Calculating this with little use of NumPy for better readability and
# it's initialization code, so performance doesn't matter.
# TODO: Make n_a/n_as etc. consistent. (RM 2017-09-22)
def _psi(T, omega):
    """
    Psi(st, at+1, ot+1) = P(ot+1|at, at+1)
    """
    (n_a, n_s, n_o) = omega.shape
    res = np.empty((n_s, n_a, n_o))
    for (st, at1, ot1), _ in np.ndenumerate(res):
        res[st, at1, ot1] = sum([T[st, at1, st1] * omega[at1, st1, ot1]
                                 for st1 in xrange(n_s)])

    return res




# pylint: disable=too-many-instance-attributes
# I've thoughta about these too-many's and I think it's still okay.
class PBVI(object):
    # pylint: disable=too-many-arguments
    def __init__(self, R, T, omega, discount_gamma, seed=None):
        self.gamma_ast      = _gamma_ast(R)
        self.omega          = _omega(omega)
        self.discount_gamma = discount_gamma
        self.tau            = _tau(T)
        self._outs          = collections.defaultdict(dict)
        self.random         = np.random.RandomState(seed)
        self.previous_n_alphas = 0


    def gamma(self, V_):
        if self.previous_n_alphas != len(V_):
            self._outs.clear()
            self.previous_n_alphas = len(V_)

        l = self._outs['gamma']  # l for locals

        l['prod1']   = np.multiply(self.omega, V_, out=l.get('prod1'))
        l['prod2']   = np.multiply(self.tau, l['prod1'][...,None],
                                   out=l.get('prod2'))
        l['sum_s_']  = np.sum(l['prod2'], 3, out=l.get('sum_s_'))
        l['result']  = np.multiply(self.discount_gamma, l['sum_s_'],
                                   out=l.get('result'))

        return l['result']


    def epsilon(self, B, gamma):
        l = self._outs['epsilon']

        # alpha · b for all a, o, b
        l['crossprods']         = np.matmul(B, np.swapaxes(gamma, -1, -2),
                                            out=l.get('crossprods'))
        l['best_alpha_inds']    = np.argmax(l['crossprods'], -1,
                                            out=l.get('best_alpha_inds'))

        (n_as, n_os)    = gamma.shape[0:2]

        # Credits: https://stackoverflow.com/questions/40357335/
        #          numpy-how-to-get-a-max-from-an-argmax-result
        best_per_o      = gamma[np.arange(n_as)[:,None,None],
                                np.arange(n_os)[None,:,None],
                                l['best_alpha_inds']]

        # Gamma_a_b for each a and b
        l['E'] = np.sum(best_per_o, 1, out=l.get('E'))  # not yet E
        l['E'] += self.gamma_ast   # Now it's E.
        # If this is slow, try E.swapaxes(…) += gamma_ast (gamma_ast unmodified)

        return l['E']

    # TODO: See if it makes sense to return the E in a different shape, so we
    # don't have to do the swapaxes later. (RM 2017-09-18)
    def V(self, E, B):
        l = self._outs['V']
        E = E.swapaxes(0,1)

        # Note: Does anyone know how to short this? Essentially it's a
        # broadcast matrix-vector multiplication.
        l['product']    = np.multiply(E, B[:,None,:], out=l.get('product'))
        l['values']     = np.sum(l['product'], -1, out=l.get('values'))

        l['best_as']    = np.argmax(np.squeeze(l['values']), axis=1,
                                    out=l.get('best_as'))

        return np.unique(E[np.arange(E.shape[0]), l['best_as']],
                         axis=0)
        # The np.unique is the pruning step.
        # Requires NumPy 1.13.1!


    def expanded_B(self, B):
        n_as = self._tau.shape[0]
        n_ss = B.shape[1]

        s_samples = np.array([self.random.choice(n_ss, size=n_as, replace=True,
                                                 p=b) 
                              for b in B])

