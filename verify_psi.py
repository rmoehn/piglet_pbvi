import numpy as np


def psi_samples(st, at1, T, omega, n_root=100):
    """

    You get n_root**2 samples. So n_root is the square root of the number of
    samples.
    """
    (n_s, n_o) = omega.shape[1:]
    st1_samples = np.choice(n_s, size=n_root, replace=True, p=T[st, at1])
    ot1_samples = [np.choice(n_o, size=n_root, replace=True, p=omega[at1, st1])
                   for st1 in st1_samples]
    return np.flatten(ot1_samples)
