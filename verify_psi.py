import numpy as np

# Note: The names and argument order in this file is not consistent with that in
# the main files. If you use the code here, adjust it first.


def psi_samples(st, at1, T, omega, n_root=100):
    """

    You get n_root**2 samples. So n_root is the square root of the number of
    samples.
    """
    (n_s, n_o) = omega.shape[1:]
    st1_samples = np.random.choice(n_s, size=n_root, replace=True, p=T[st, at1])
    ot1_samples = [np.random.choice(n_o, size=n_root, replace=True,
                                    p=omega[at1, st1])
                   for st1 in st1_samples]
    return np.array(ot1_samples).flatten()


def sample_psi(T, omega, n_root=100):
    (n_a, n_s, n_o) = omega.shape
    res = np.empty((n_s, n_a, n_o))

    for (st, at1, ot1), _ in np.ndenumerate(res):
        ot1_samples = psi_samples(st, at1, T, omega, n_root=n_root)

        res[st, at1, ot1] = np.mean(ot1_samples == ot1)

    return res
