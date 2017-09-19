import numpy as np


class NaivePBVI(object):
    def __init__(self, R, T, omega, discount_gamma):
        self.gamma_ast      = R.T
        self.T              = T
        self.omega          = omega
        self.discount_gamma = discount_gamma
        self.n_a, self.n_s, self.n_o    = self.omega.shape


    def gamma(self, V_):
        res = np.empty((self.n_a, self.n_o, len(V_), self.n_s))

        for (a, o, i, s), _ in np.ndenumerate(res):
            res[a,o,i,s] \
                = self.discount_gamma \
                * sum(self.T[s,a,s_] * self.omega[a,o,s_] * V_[i,s_]
                      for s_ in xrange(self.n_s))

        return res


    def epsilon(self, B, gamma):
        res = np.empty((B.shape[0], self.n_a, self.n_s))

        for b in xrange(B.shape[0]):
            for a in xrange(self.n_a):
                per_o = np.empty((self.n_o,self.n_s))
                for o in xrange(self.n_o):
                    dot         = np.dot(gamma[a,o], B[b])
                    amax        = np.argmax(dot)
                    per_o[o]    = gamma[a,o,amax]

                res[b,a] = self.gamma_ast[a] + np.sum(per_o, 0)

        return res


    def V(self, E, B):
        res = np.empty((B.shape[0], self.n_s))
        for b in xrange(B.shape[0]):
            res[b] = E[b, np.argmax(np.dot(E[b], B[b]))]

        return np.unique(res, axis=0)
