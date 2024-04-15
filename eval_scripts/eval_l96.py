import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data
import pdb, os, time
from scipy.ndimage import gaussian_filter
path = os.getcwd()
os.chdir(path)
import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from tqdm import tqdm
from scripts.cal_stats_l96 import cal_stats_l1_score
from scripts.train_utils import LpLoss_

def spectrum(u):
    u = torch.fft.rfft(u)
    u = u.abs()**2
    u = u.mean(dim=0)
    return u

def get_all_data(dataset, num_workers=30, shuffle=False):
    dataset_size = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size,
                             num_workers=num_workers, shuffle=shuffle)
    all_data = {}
    for i_batch, sample_batched in tqdm(enumerate(data_loader)):
        all_data = sample_batched
    return all_data

def eval_l96(operator, args, noisy_scale = 0, x_len = 100, calculate_l2 = True, output_path = ''):

    from dataloader.dataloader_l96 import TrainingData, TestingData
    l2_stats_list, param_list, out_list, x_list = [], [], [], []
    slow_stats_list = []
    l2_lploss_list, dist_spectrum = [], []

    eval_size = 200
    crop_T = 2000
    if noisy_scale == 0:
        TestingData_Initial = TestingData(crop_T, noisy_scale = args.noisy_scale, convert_to_pil = False)
        TestingData_eval = TestingData(crop_T, noisy_scale = noisy_scale, convert_to_pil = False)
    else:
        TestingData_Initial = TrainingData(crop_T, noisy_scale = args.noisy_scale, convert_to_pil = False, validation = True, train_operator = True, train_size = eval_size)
        TestingData_eval = TrainingData(crop_T, noisy_scale = noisy_scale, convert_to_pil = False, validation = True, train_operator = True, train_size = eval_size)

    all_data_initial = get_all_data(TestingData_Initial)
    all_data_eval = get_all_data(TestingData_eval)
    params_initial, data_initial = all_data_initial[0], all_data_initial[1]
    params_eval, data_eval = all_data_eval[0], all_data_eval[1]
    del all_data_initial
    del all_data_eval
    params_initial, data_initial = params_initial[:eval_size], data_initial[:eval_size].squeeze()
    params_eval, data_eval = params_eval[:eval_size], data_eval[:eval_size].squeeze()


    if x_len == 100:
        x_0_start_list = [100]
    else:
        x_0_start_list = [300]

    for x_0_start in tqdm(x_0_start_list):
        x_len = x_len
        x_end = x_0_start + x_len
        with torch.no_grad():
            operator.eval()
            params_list = params_eval
            x_0 = data_initial[:, x_0_start][:, None, :].to(args.gpu)
            data_list = [x_0]
            if x_len == 100:
                for ix in range(x_len-1):
                    if ix % 2 == 0:
                        x_1 = data_initial[:, ix][:, None, :].to(args.gpu)
                        data_list.append(x_1)
                        x_0 = x_1
                    else:
                        x_1 = operator(x_0, params_eval[:, 0][:, None].to(args.gpu))
                        data_list.append(x_1)
                        x_0 = x_1
            else:
                for ix in range(x_len-1):
                    x_1 = operator(x_0, params_eval[:, 0][:, None].to(args.gpu))
                    data_list.append(x_1)
                    x_0 = x_1

            x_list = data_eval[:, x_0_start:x_end].to(args.gpu)
            x_list_initial = data_initial[:, x_0_start:x_end].to(args.gpu)
            out_list = torch.cat(data_list, dim = 1)

            for idata in range(eval_size):
                out = out_list[idata]
                x = x_list[idata]
                assert out.shape == x.shape
                assert out.shape[0] == x_len
                spec_truth = spectrum(x)
                spec_our = spectrum(out)

                diff = abs(spec_truth - spec_our)
                rel_diff = diff/(spec_truth.sum())
                dist_spectrum.append(rel_diff.sum().cpu().data.numpy())

                bins_ = 0
                slow_stats = cal_stats_l1_score(x.cpu().data.numpy(), out.cpu().data.numpy())
                slow_stats_list.append(slow_stats)

                ###############################################################

        spectrum_stats = [np.quantile(np.array(dist_spectrum), q = 0.25), np.array([50]).reshape(-1,1), np.quantile(np.array(dist_spectrum), q = 0.5), np.quantile(np.array(dist_spectrum), q = 0.75)]
        print('spectrum distance', spectrum_stats)

        slow_stats_array = np.array(slow_stats_list)
        os.makedirs('npy_l96', exist_ok = True)
        save_path = f'npy_l96/{args.prefix}_noise_eval_{noisy_scale}_{args.noisy_scale}_{x_0_start}_{x_end}'
        torch.save({'out':out_list, 'x':x_list, 'slow_stats_array': slow_stats_array}, save_path)
        l1_score_3d = [slow_stats_array.mean(axis = 0), np.quantile(slow_stats_array, q = 0.25), \
                            '50 percentile', np.quantile(slow_stats_array, q = 0.5), np.quantile(slow_stats_array, q = 0.75)]
        del out_list, x_list
        del slow_stats_array

    with torch.no_grad():
        operator.eval()
        eval_bsize = 5
        l2_list, l2_lp_list = [], []
        for b_ix in range(int(eval_size/eval_bsize)):
            data_initial_btz = data_initial[eval_bsize*b_ix:eval_bsize*(b_ix+1)]
            params_eval_btz = params_eval[eval_bsize*b_ix:eval_bsize*(b_ix+1), 0].reshape(eval_bsize, -1)
            data_eval_btz = data_eval[eval_bsize*b_ix:eval_bsize*(b_ix+1)]
            x_0 = data_initial_btz[:, :-1].reshape(eval_bsize*(data_initial_btz.shape[1]-1), -1)
            params = params_eval_btz.repeat(1, data_initial_btz.shape[1]-1).reshape(eval_bsize*(data_initial_btz.shape[1]-1), -1)
            x_1_true = data_eval_btz[:, 1:].reshape(eval_bsize*(data_initial_btz.shape[1]-1), -1).to(args.gpu)
            x_predict = operator(x_0[:, None, :], params).squeeze().to(args.gpu)
            l2_lp_list.append(LpLoss_(2).rel(x_predict, x_1_true).cpu().data.numpy())
    print('one step RMSE', np.array(l2_lp_list).reshape(-1).mean())

    l2_list_stats = ['rMSE', np.quantile(np.array(l2_lp_list), 0.25), \
                    '50 percentile', np.quantile(np.array(l2_lp_list), 0.5), np.quantile(np.array(l2_lp_list), 0.75)]

    mse_step, l1_score_3d, spectrum_dist = l2_list_stats, l1_score_3d, spectrum_stats

    if noisy_scale == 0:
        with open(f'{output_path}/Results_test_on_clean_data.txt', 'w') as f:
            f.writelines(f'noise {noisy_scale} with eval length {x_len} training length {args.x_len} \n')
            f.writelines(f'mse_{mse_step} \n ')
            f.writelines(f'l1_3d_score: {l1_score_3d} \n spectrum distance:{spectrum_stats} \n \n')
    else:
        with open(f'{output_path}/Results_validation_on_noise_data.txt', 'w') as f:
            f.writelines(f'noise {noisy_scale} with eval length {x_len} training length {args.x_len} \n')
            f.writelines(f'mse_{mse_step} \n ')
            f.writelines(f'l1_3d_score: {l1_score_3d} \n spectrum distance:{spectrum_stats} \n \n')
