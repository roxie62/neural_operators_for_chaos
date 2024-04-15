import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb, os, time
path = os.getcwd()
os.chdir(path)
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from eval_scripts.LE import lyapunov
from tqdm import tqdm

def get_all_data(dataset, num_workers=30, shuffle=False):
    dataset_size = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size,
                             num_workers=num_workers, shuffle=shuffle)
    all_data = {}
    for i_batch, sample_batched in tqdm(enumerate(data_loader)):
        all_data = sample_batched
    return all_data

def cal_LE(operator, args, noisy_scale = 0, x_len = 200, calculate_l2 = True):
    if args.l96:
        from dataloader.dataloader_l96 import TrainingData, TestingData
        TestingData_Initial = TestingData(200, noisy_scale = args.noisy_scale, convert_to_pil = False)
        TestingData_eval = TestingData(200, noisy_scale = noisy_scale, convert_to_pil = False)

    all_data_initial = get_all_data(TestingData_Initial)
    all_data_eval = get_all_data(TestingData_eval)
    params_initial, data_initial = all_data_initial[0], all_data_initial[1]
    params_eval, data_eval = all_data_eval[0], all_data_eval[1]
    del all_data_initial
    del all_data_eval
    eval_size = 200
    params_initial, data_initial = params_initial[:eval_size], data_initial[:eval_size]
    params_eval, data_eval = params_eval[:eval_size], data_eval[:eval_size]
    embed_distance_list = []
    t_0 = 500
    x_0 = data_initial[:, t_0][:, None, :].to(args.gpu)
    params_list = params_eval
    assert params_list.shape[0] == x_0.shape[0]

    def step_wrapper(model, params_i, args):
        def step(states, psued_dt):
            if not isinstance(states, torch.Tensor):
                states = torch.from_numpy(states).to(args.gpu).float()
            states = states.reshape(2, 1, -1)
            model.eval()
            states = model(states, params_i.repeat(states.shape[0], 1)).cpu().data.numpy().squeeze()
            return states
        return step

    def step_wrapper_spectrum(model, params_i, args):
        def step(states):
            if not isinstance(states, torch.Tensor):
                states = torch.from_numpy(states).to(args.gpu).float()
            states = states.reshape(states.shape[0], 1, -1)
            model.eval()
            states = model(states, params_i.repeat(states.shape[0], 1)).cpu().data.numpy().squeeze()
            return states
        return step

    LE_result_list = []
    for i in tqdm(range(0, 200)):
        param_i = params_list[:, 0][i].reshape(-1, 1)
        x0_i = x_0[i].data.cpu().numpy()
        operator.eval()
        step_fn = step_wrapper(operator, param_i, args)
        step_fn_spectrum = step_wrapper_spectrum(operator, param_i, args)
        os.makedirs(f'output_folder/LE_results', exist_ok = True)
        if args.l96:
            Ttr = 100
            total_step = 1000
            use_dapper = False
            LE_result = lyapunov(step_fn, total_step, x0_i, d0 = 1e-2, delta_t=0.1, Ttr=Ttr, show_progress=True)
            LE_result_list.append([param_i.cpu().data.numpy(), LE_result])
            torch.save(LE_result_list, f'output_folder/LE_results/{args.prefix}.pth')
