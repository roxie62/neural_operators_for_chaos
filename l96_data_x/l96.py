import numpy as np
from scipy.integrate import solve_ivp
import torch

def lorenz96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F
    return dxdt

def generate_l96_data(F, N=60, T=200, dt=0.0005, initial_conditions = np.array([0]), t_res = 200):
    seed = int(F[-1])
    np.random.seed(seed)
    F = F[0]
    t_span = (0, T)
    if initial_conditions.sum() == 0:
        initial_conditions = np.random.normal(0, 1, N)
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(
        lorenz96,
        t_span,
        initial_conditions,
        args=(F,),
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-9,  # Relative tolerance
        atol=1e-12,  # Absolute tolerance
    )
    return sol.y.T[::t_res, :]
