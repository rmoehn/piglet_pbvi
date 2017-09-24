# -*- encoding: utf-8 -*-

# TODO: Finish those naming conventions. (RM 2017-09-23)
"""

Naming conventions

* Names for matrices that represent important parts of the problem start with a
  capital letter.
* Names of inputs are single letters or words not from the Greek alphabet.
* Names of internal or intermediate result matrices are names of Greek letters,
  starting with a capital letter, shortened to their first four or five letters.

Examples:

* T, Op, R
"""

import collections

import numpy as np


def _Gamma_ast(R):
    return R.T[:,None,:].copy()
    # Copy in order to reorder in memory.


def _T(T):
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


def _Omega(Omega):
    """
    Reorder the observation probability array for performance.

    In::

        |A| x |S| x |O|
         a     s'    o

    Out::

        |A| x |O| x  1  x |S|
         a     o           s'
    """
    return np.swapaxes(Omega, -2, -1)[:,:,None].copy()


# Note: Calculating this with little use of NumPy for better readability and
# it's initialization code, so performance doesn't matter.
def _Psi(T, Omega):
    """
    Psi(st, at+1, ot+1) = P(ot+1 | st, at+1)

    Note that for easier calculation further down, the shape is as follows::

        |O| x |S| x |A|
        ot+1   st   at+1
    """
    (n_a, n_s, n_o) = Omega.shape
    res = np.empty((n_o, n_s, n_a))
    for (ot1, st, at1), _ in np.ndenumerate(res):
        res[ot1, st, at1] = sum([T[st, at1, st1] * Omega[at1, st1, ot1]
                                 for st1 in xrange(n_s)])

    return res


Input   = collections.namedtuple("Input", ["T", "Omega", "R", "gamma"])
Size    = collections.namedtuple("Size", ['s', 'a', 'o'])


# pylint: disable=too-many-instance-attributes
# I've thought about these too-many's and I think it's still okay.
# Note:
#  - If the name of an attribute starts with an underscore, it usually
#    corresponds to a name in the problem or algorithm definition, but is
#    changed in some way to suit the computation.
#  - I a name ends with an underscore, it means name'.
#  - If a name looks like cSomething, it's a function that calculates Something.
#    I prepend the c only if there would otherwise be a naming conflict between
#    the global function and the local result.
class PBVI(object):
    # pylint: disable=too-many-arguments
    def __init__(self, T, Omega, R, gamma, seed=None):
        # Unmodified inputs
        self.i              = Input(T, Omega, R, gamma)
        self._Gamma_ast     = _Gamma_ast(R)
        self._Omega         = _Omega(Omega)
        self._T             = _T(T)
        self._Psi           = _Psi(T, Omega)
        self._outs          = collections.defaultdict(dict)
        self.random         = np.random.RandomState(seed)
        self.previous_n_alphas = 0
        n_s, n_a, n_o       = Omega.shape
        self.n              = Size(s=n_s, a=n_a, o=n_o)


    def Gamma(self, V_):
        if self.previous_n_alphas != len(V_):
            self._outs.clear()
            self.previous_n_alphas = len(V_)

        l = self._outs['Gamma']  # l for locals

        l['prod1']   = np.multiply(self._Omega, V_, out=l.get('prod1'))
        l['prod2']   = np.multiply(self._T, l['prod1'][...,None],
                                   out=l.get('prod2'))
        l['sum_s_']  = np.sum(l['prod2'], 3, out=l.get('sum_s_'))
        l['result']  = np.multiply(self.i.gamma, l['sum_s_'],
                                   out=l.get('result'))

        return l['result']


    def Epsi(self, B, Gamma):
        l = self._outs['Epsi']

        # alpha · b for all a, o, b
        l['crossprods']         = np.matmul(B, np.swapaxes(Gamma, -1, -2),
                                            out=l.get('crossprods'))
        l['best_alpha_inds']    = np.argmax(l['crossprods'], -1,
                                            out=l.get('best_alpha_inds'))

        # Credits: https://stackoverflow.com/questions/40357335/
        #          numpy-how-to-get-a-max-from-an-argmax-result
        best_per_o      = Gamma[np.arange(self.n.a)[:,None,None],
                                np.arange(self.n.o)[None,:,None],
                                l['best_alpha_inds']]

        # Gamma_a_b for each a and b
        l['result'] = np.sum(best_per_o, 1, out=l.get('result'))  # Not yet Epsi.
        l['result'] += self._Gamma_ast  # Now it's Epsi.
        # If this is slow, try E.swapaxes(…) += Gamma_ast (Gamma_ast unmodified)

        return l['result']


    # TODO: See if it makes sense to return the E in a different shape, so we
    # don't have to do the swapaxes later. (RM 2017-09-18)
    def V(self, Epsi, B):
        l = self._outs['V']
        Epsi = Epsi.swapaxes(0,1)

        # Note: Does anyone know how to short this? Essentially it's a
        # broadcast matrix-vector multiplication.
        l['product']    = np.multiply(Epsi, B[:,None,:], out=l.get('product'))
        l['values']     = np.sum(l['product'], -1, out=l.get('values'))

        l['best_as']    = np.argmax(np.squeeze(l['values']), axis=1,
                                    out=l.get('best_as'))

        return np.unique(Epsi[np.arange(Epsi.shape[0]), l['best_as']],
                         axis=0)
        # The np.unique is the pruning step.
        # Requires NumPy 1.13.1!


    def expanded_B(self, B):
        s_samples = np.array([self.random.choice(self.n.s, size=self.n.a,
                                                 replace=True, p=b) 
                              for b in B])

