import numpy as np
from scipy.integrate import solve_ivp
import torch, pdb, tqdm


def lyapunov(step, T, u0, d0=1e-2, inittest=None, Ttr=0, delta_t=1, d0_upper=None, d0_lower=None, show_progress=False, **kwargs):
    if d0_upper is None:
        d0_upper = d0*1e+3
    if d0_lower is None:
        d0_lower = d0*1e-3

    def inittest_default(D):
        return lambda state1, d0: state1 + d0 / np.sqrt(D)

    def lambda_dist(states):
        u1 = states[0]
        u2 = states[1]
        return np.sqrt(np.sum((u1 - u2)**2))

    def lambda_rescale(states, a):
        u1 = states[0]
        u2 = states[1]
        u2[:] = u1 + (u2 - u1) / a

    current_time = 0
    dimension = len(u0)

    if inittest is None:
        inittest = inittest_default(dimension)

    states = np.stack([u0, inittest(u0, d0)])

    # Transient
    while current_time < Ttr:
        states = step(states, delta_t)
        current_time += delta_t
        d = lambda_dist(states)
        if not (d0_lower <= d <= d0_upper):
            lambda_rescale(states, d / d0)

    # Set up algorithm
    t0 = current_time
    d = lambda_dist(states)
    # print('d at first', d)
    if d == 0:
        raise ValueError("Initial distance between states is zero!!!")

    if d != d0:
        lambda_rescale(states, d / d0)

    lambda_ = 0.0

    while current_time < t0 + T:
        d = lambda_dist(states)
        while d0_lower <= d <= d0_upper:
            states = step(states, delta_t)
            current_time += delta_t
            d = lambda_dist(states)
            if current_time >= t0 + T:
                break

        a = d / d0
        lambda_ += np.log(a)

        ## if d == 0, break
        d = lambda_dist(states)
        if d == 0:
            break

        lambda_rescale(states, a)

    # Final rescale, in case no other happened
    d = lambda_dist(states)
    if d == 0:
        return 0
    a = d / d0
    lambda_ += np.log(a)

    return lambda_ / (current_time - t0)
