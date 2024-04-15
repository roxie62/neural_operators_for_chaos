import numpy as np
from scipy.integrate import solve_ivp
import torch, pdb, os
from tqdm import tqdm
from torch.utils.data import Dataset
path = os.getcwd()
os.chdir(path)
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from dataloader.dataloader_l96 import TrainingData, TestingData


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

    t0 = current_time
    d = lambda_dist(states)
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

def get_all_data(dataset, num_workers=30, shuffle=False):
    dataset_size = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size,
                             num_workers=num_workers, shuffle=shuffle)
    all_data = {}
    for i_batch, sample_batched in tqdm(enumerate(data_loader)):
        all_data = sample_batched
    return all_data

noisy_scale = 0.3
TestingData_Initial = TestingData(200, noisy_scale = noisy_scale, convert_to_pil = False)
TestingData_eval = TestingData(200, noisy_scale = 0, convert_to_pil = False)
all_data_initial = get_all_data(TestingData_Initial)
all_data_eval = get_all_data(TestingData_eval)
params_initial, data_initial = all_data_initial[0], all_data_initial[1]
params_eval, data_eval = all_data_eval[0], all_data_eval[1]
del all_data_initial
del all_data_eval
eval_size = 200
params_initial, data_initial = params_initial[:eval_size], data_initial[:eval_size]
params_eval, data_eval = params_eval[:eval_size], data_eval[:eval_size]


def lorenz96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F
    return dxdt

def step(states, dt):
    T = dt
    dtt = 0.0005
    t_eval = np.arange(0, T, dtt)
    states[0] = solve_ivp(lorenz96, (0, dt), states[0], args=(F,), t_eval=t_eval, method='DOP853', rtol=1e-9, atol=1e-12).y[:,-1]
    states[1] = solve_ivp(lorenz96, (0, dt), states[1], args=(F,), t_eval=t_eval, method='DOP853', rtol=1e-9, atol=1e-12).y[:,-1]
    return states

F = 18
N = 60
t_start = 500
LE_result_list = []
# LE_result_list = torch.load(f'output_folder/LE_results/{t_start}_LE_result_{noisy_scale}.pth')
for i_data in range(len(LE_result_list), eval_size):
    print(i_data)
    F = params_eval[i_data].cpu().data.numpy().item()
    u0 = data_initial[i_data, t_start].cpu().data.numpy()
    LE_result = lyapunov(step, 1000, u0, delta_t=0.1, Ttr=10, show_progress=True)
    print(F, LE_result)
    LE_result_list.append([F, LE_result])
    torch.save(LE_result_list, f'output_folder/LE_results/{t_start}_LE_result_{noisy_scale}.pth')
