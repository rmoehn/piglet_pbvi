import numpy as np


class NaivePBVI(object):
    def __init__(self, T, Omega, R, gamma):
        self.Gamma_ast      = R.T
        self.T              = T
        self.Omega          = Omega
        self.gamma          = gamma
        self.n_a, self.n_s, self.n_o    = self.Omega.shape


    def Gamma(self, V_):
        res = np.empty((self.n_a, self.n_o, len(V_), self.n_s))

        for (a, o, i, s), _ in np.ndenumerate(res):
            res[a,o,i,s] \
                = self.gamma \
                * sum(self.T[s,a,s_] * self.Omega[a,o,s_] * V_[i,s_]
                      for s_ in xrange(self.n_s))

        return res


    def Epsi(self, B, Gamma):
        res = np.empty((B.shape[0], self.n_a, self.n_s))

        for b in xrange(B.shape[0]):
            for a in xrange(self.n_a):
                per_o = np.empty((self.n_o,self.n_s))
                for o in xrange(self.n_o):
                    dot         = np.dot(Gamma[a,o], B[b])
                    amax        = np.argmax(dot)
                    per_o[o]    = Gamma[a,o,amax]

                res[b,a] = self.Gamma_ast[a] + np.sum(per_o, 0)

        return res


    def V(self, Epsi, B):
        res = np.empty((B.shape[0], self.n_s))
        for b in xrange(B.shape[0]):
            res[b] = Epsi[b, np.argmax(np.dot(Epsi[b], B[b]))]

        return np.unique(res, axis=0)
