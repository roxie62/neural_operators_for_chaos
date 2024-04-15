import torch
import numpy as np
import pdb, random
import argparse, os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from l96 import generate_l96_data, lorenz96

parser = argparse.ArgumentParser(description='End-to-End emulator')
parser.add_argument('--sample_prior_min', default=[12], nargs='*', type = float)
parser.add_argument('--sample_prior_max', default=[18], nargs='*', type = float)
parser.add_argument('--gpu', default = 0, type = int)
parser.add_argument('--split', default = 2, type = int)
parser.add_argument('--num_of_sample', default = 500, type = int)
args = parser.parse_args()

############################### generate the training data######################
generate_training_data = False
data_folder, random_index = 'F_2000_dt10', 0
args.num_of_sample = 2000
os.makedirs(data_folder, exist_ok = True)
if generate_training_data:
    if not os.path.exists(f'{data_folder}/training_params.pth'):
        GT_min, GT_max = np.array(args.sample_prior_min), np.array(args.sample_prior_max)
        GT_params = np.round(np.random.uniform(low = 0, high = 1, size = (args.num_of_sample, 1)), 4)
        GT_params = GT_params * (GT_max - GT_min) + GT_min
        lorenz_params_train = torch.from_numpy(GT_params)
        torch.save(lorenz_params_train, f'{data_folder}/training_params.pth')
    else:
        lorenz_params_train = torch.load(f'{data_folder}/training_params.pth')
    print(lorenz_params_train.min(axis = 0), '\n', lorenz_params_train.max(axis = 0))
    traj_list = []
    n_workers = 50
    num = 100
    for i in tqdm(range(0, int(args.num_of_sample/num))):
        split = args.split
        assert num%split == 0
        for j in range(0,split):
            print(int(i*num + num/split*j), int(i*num + num/split*(j+1)))
            params = lorenz_params_train[int(i*num + num/split*j):int(i*num + num/split*(j+1))]
            params_cat_seed = np.concatenate([params, (random_index + np.arange(params.shape[0]) + int(i*num + j*num/split))[:, None]], axis = -1)
            with Pool(n_workers) as pool:
                total_traj = np.array(pool.map(generate_l96_data, params_cat_seed))
            for ix in range(params.shape[0]):
                torch.save({'0': params[ix], '1': total_traj[ix]}, '{}/{:06d}.pth'.format(data_folder, int(i*num+ix+num/split*j)))

########################## generate the validation data#########################
generate_validation_data = True
data_folder, random_index = 'F_2000_dt10_validation', 5000
args.num_of_sample = 100
os.makedirs(data_folder, exist_ok = True)
if generate_validation_data:
    if not os.path.exists(f'{data_folder}/training_params.pth'):
        GT_min, GT_max = np.array(args.sample_prior_min), np.array(args.sample_prior_max)
        GT_params = np.round(np.random.uniform(low = 0, high = 1, size = (args.num_of_sample, 1)), 4)
        GT_params = GT_params * (GT_max - GT_min) + GT_min
        lorenz_params_train = torch.from_numpy(GT_params)
        torch.save(lorenz_params_train, f'{data_folder}/training_params.pth')
    else:
        lorenz_params_train = torch.load(f'{data_folder}/training_params.pth')
    print(lorenz_params_train.min(axis = 0), '\n', lorenz_params_train.max(axis = 0))
    traj_list = []
    n_workers = 50
    num = 100
    for i in tqdm(range(0, int(args.num_of_sample/num))):
        split = args.split
        assert num%split == 0
        for j in range(0,split):
            print(int(i*num + num/split*j), int(i*num + num/split*(j+1)))
            params = lorenz_params_train[int(i*num + num/split*j):int(i*num + num/split*(j+1))]
            params_cat_seed = np.concatenate([params, (random_index + np.arange(params.shape[0]) + int(i*num + j*num/split))[:, None]], axis = -1)
            with Pool(n_workers) as pool:
                total_traj = np.array(pool.map(generate_l96_data, params_cat_seed))
            for ix in range(params.shape[0]):
                torch.save({'0': params[ix], '1': total_traj[ix]}, '{}/{:06d}.pth'.format(data_folder, int(i*num+ix+num/split*j)))

########################## generate the test data###############################
generate_test_data = True
data_folder, random_index = 'F_2000_dt10_test', 10000
args.num_of_sample = 200
os.makedirs(data_folder, exist_ok = True)
if generate_test_data:
    if not os.path.exists(f'{data_folder}/training_params.pth'):
        GT_min, GT_max = np.array(args.sample_prior_min), np.array(args.sample_prior_max)
        GT_params = np.round(np.random.uniform(low = 0, high = 1, size = (args.num_of_sample, 1)), 4)
        GT_params = GT_params * (GT_max - GT_min) + GT_min
        lorenz_params_train = torch.from_numpy(GT_params)
        torch.save(lorenz_params_train, f'{data_folder}/training_params.pth')
    else:
        lorenz_params_train = torch.load(f'{data_folder}/training_params.pth')
    print(lorenz_params_train.min(axis = 0), '\n', lorenz_params_train.max(axis = 0))
    traj_list = []
    n_workers = 50
    num = 100
    for i in tqdm(range(0, int(args.num_of_sample/num))):
        split = args.split
        assert num%split == 0
        for j in range(0,split):
            print(int(i*num + num/split*j), int(i*num + num/split*(j+1)))
            params = lorenz_params_train[int(i*num + num/split*j):int(i*num + num/split*(j+1))]
            params_cat_seed = np.concatenate([params, (random_index + np.arange(params.shape[0]) + int(i*num + j*num/split))[:, None]], axis = -1)
            with Pool(n_workers) as pool:
                total_traj = np.array(pool.map(generate_l96_data, params_cat_seed))
            for ix in range(params.shape[0]):
                torch.save({'0': params[ix], '1': total_traj[ix]}, '{}/{:06d}.pth'.format(data_folder, int(i*num+ix+num/split*j)))
