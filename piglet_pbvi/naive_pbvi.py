"""Implementation of PBVI that emphasizes readibility over speed.

The interface of :class:`NaivePBVI` is be the same as that of
:class:`pbvi.PBVI`.
"""

import collections

import numpy as np


Size = collections.namedtuple("Size", ['s', 'a', 'o'])


class NaivePBVI(object):
    def __init__(self, T, Omega, R, gamma):
        self.Gamma_ast  = R.T
        self.T          = T
        self.Omega      = Omega
        self.gamma      = gamma
        n_a, n_s, n_o   = self.Omega.shape
        self.n          = Size(s=n_s, a=n_a, o=n_o)


    def Gamma(self, V_):
        res = np.empty((self.n.a, self.n.o, len(V_), self.n.s))

        for (a, o, i, s), _ in np.ndenumerate(res):
            res[a,o,i,s] \
                = self.gamma \
                * sum(self.T[s,a,s_] * self.Omega[a,o,s_] * V_[i,s_]
                      for s_ in xrange(self.n.s))

        return res


    def Epsi(self, B, Gamma):
        res = np.empty((B.shape[0], self.n.a, self.n.s))

        for b in xrange(B.shape[0]):
            for a in xrange(self.n.a):
                per_o = np.empty((self.n.o,self.n.s))
                for o in xrange(self.n.o):
                    dot         = np.dot(Gamma[a,o], B[b])
                    amax        = np.argmax(dot)
                    per_o[o]    = Gamma[a,o,amax]

                res[b,a] = self.Gamma_ast[a] + np.sum(per_o, 0)

        return res


    def V(self, Epsi, B):
        res = np.empty((B.shape[0], self.n.s))
        for b in xrange(B.shape[0]):
            res[b] = Epsi[b, np.argmax(np.dot(Epsi[b], B[b]))]

        return np.unique(res, axis=0)


    def _bayes_update(self, b, a, o):
        b_ = np.empty_like(b)
        for s_ in xrange(len(b_)):
            b_[s_] = self.gamma * self.Omega[a,s_,o] * np.dot(self.T[:,a,s_], b)

        return b_ / np.sum(b_)


    def expanded_B(self, B):
        s_b_a   = np.array([[np.random.choice(self.n.s, p=b)
                             for _ in xrange(self.n.a)]
                            for b in B])

        s__b_a  = np.array([[np.random.choice(self.n.s,
                                              p=self.T[s_b_a[i_b,a], a])
                             for a in xrange(self.n.a)]
                            for i_b in xrange(len(B))])

        o_b_a   = np.array([[np.random.choice(self.n.o,
                                              p=self.Omega[a, s__b_a[i_b,a]])
                             for a in xrange(self.n.a)]
                            for i_b in xrange(len(B))])

        b__b_a  = np.array([[self._bayes_update(b, a, o_b_a[i_b, a])
                             for a in xrange(self.n.a)]
                            for i_b, b in enumerate(B)])

        b_s = list(B)
        for i_b, _ in enumerate(B):
            min_dist = [min(np.linalg.norm(b__b_a[i_b, a] - b, ord=1)
                            for b in b_s)
                        for a in xrange(self.n.a)]
            # Problematic naming here, because suddenly we have both bs and b's
            # in one list, b_s.

            max_a = np.argmax(min_dist)

            if min_dist[max_a] != 0:
                b_s.append(b__b_a[i_b, max_a])

        return np.array(b_s)
