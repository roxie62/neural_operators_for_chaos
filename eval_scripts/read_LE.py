import numpy as np
import torch, sys
import pdb, os

path = os.getcwd()
os.chdir(path)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

noise_scale = 0.3
LE_folder = 'output_folder/LE_results'
if noise_scale == 0.3:
    cl = f'{LE_folder}/l96_xl_100_embd_128_lmCL_0.8.pth'
    # ot = f'{LE_folder}/l96_noCL_xl_100_lmOT_3.0_OT_3.pth'
    ot = f'{LE_folder}/l96_noCL_xl_100_lmOT_3.0.pth'
    baseline = f'{LE_folder}/l96_noCL_xl_100.pth'
truth = f'{LE_folder}/500_LE_result_{noise_scale}.pth'

baseline = torch.load(baseline)
cl = torch.load(cl)
ot = torch.load(ot)
truth = np.array(torch.load(truth))

cl_list = []
ot_list = []
ot_list = []
truth_list = []

def get_data(data_list):
    params, data = [], []
    for data_i in data_list:
        params.append(data_i[0])
        data.append(data_i[1])
    return np.array(params), np.array(data)

def align_params(params_1, params_2):
    num_params_1, num_params_2 = params_1.shape[0], params_2.shape[0]
    num_params = min(num_params_1, num_params_2)
    diff = params_1[:num_params] - params_2[:num_params]
    assert diff.sum() == 0
    print('shape', num_params, 'diff', diff.sum())

params_base, LE_base = get_data(baseline)
params_cl, LE_cl = get_data(cl)
params_ot, LE_ot = get_data(ot)

align_params(params_base, params_cl)
align_params(params_base, params_ot)

print(LE_base.min())
print(LE_cl.min())
print(LE_ot.min())

def get_diff(target, prediction):
    data_size = min(target.shape[0], prediction.shape[0])
    print(data_size)
    print('distance l1 norm', (abs(target[:data_size] - prediction[:data_size])/prediction[:data_size]).reshape(-1).mean(), 'std', (abs(target[:data_size] - prediction[:data_size])/prediction[:data_size]).reshape(-1).std())
    print('25', np.quantile(abs(target[:data_size] - prediction[:data_size])/prediction[:data_size], 0.25), \
            '50', np.quantile(abs(target[:data_size] - prediction[:data_size])/prediction[:data_size], 0.5), \
            '75', np.quantile(abs(target[:data_size] - prediction[:data_size])/prediction[:data_size], 0.75))

print('baseline')
get_diff(truth[:, 1], LE_base)
print('ot')
get_diff(truth[:, 1], LE_ot)
print('cl')
get_diff(truth[:, 1], LE_cl)

import matplotlib.pyplot as plt
labels = ['Truth', 'purely rMSE', 'Sinkhorn loss', 'Feature loss']
colors = ['red', 'green', 'blue', 'purple']
x1, y1 = truth[:, 0], truth[:, 1]
x2, y2 = params_base, LE_base
x3, y3 = params_ot, LE_ot
x4, y4 = params_cl, LE_cl
sizes = [5, 5, 5, 5]
shapes = ['*', '^', 'o', 'o']  # Circle, square, triangle, diamond
plt.scatter(x1, y1, color=colors[0], s=sizes[0], marker=shapes[0], label=labels[0])
plt.scatter(x2, y2, color=colors[1], s=sizes[1], marker=shapes[1], label=labels[1])
plt.scatter(x3, y3, color=colors[2], s=sizes[2], marker=shapes[2], label=labels[2])
plt.scatter(x4, y4, color=colors[3], s=sizes[3], marker=shapes[3], label=labels[3])
plt.ylim(1.5,4.5)
plt.xlabel(r'Parmaeter F of Lorenz 96 (r=0.3)')
plt.ylabel('Leading Lyapunov Exponent')
plt.legend()
plt.savefig(f'LLE_{noise_scale}.pdf')
