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
parser.add_argument('--sample_prior_min', default=[16], nargs='*', type = float)
# parser.add_argument('--sample_prior_max', default=[18], nargs='*', type = float)
# parser.add_argument('--sample_prior_max', default=[14], nargs='*', type = float)
parser.add_argument('--sample_prior_max', default=[18], nargs='*', type = float)
parser.add_argument('--gpu', default = 0, type = int)

# parser.add_argument('--num_of_sample', default = 2000, type = int)
# data_folder, random_index = 'F_2000_dt10', 2000
# data_folder, random_index = 'F_2000_dt10_test', 4000

parser.add_argument('--num_of_sample', default = 500, type = int)
data_folder, random_index = 'F_2000_16_18_test', 4000

# parser.add_argument('--num_of_sample', default = 2000, type = int)
# parser.add_argument('--num_of_sample', default = 500, type = int)
# parser.add_argument('--num_of_sample', default = 200, type = int)
# data_folder, random_index = 'F_2000_10_14_05', 2000
# data_folder, random_index = 'F_2000_dt50', 2000
# data_folder, random_index = 'F_2000_10_12_dt50_test', 2000
# data_folder, random_index = 'F_2000_10_12_dt50', 2000
# data_folder, random_index = 'F_2000_16_18', 2000
# data_folder, random_index = 'F_2000_10_14', 2000
# data_folder, random_index = 'F_2000_10_12', 2000

args = parser.parse_args()

os.mkdir(data_folder) if not os.path.exists(data_folder) else print('exist')
if True:
    if not os.path.exists('{}/training_params.pth'.format(data_folder)):
        GT_min, GT_max = np.array(args.sample_prior_min), np.array(args.sample_prior_max)
        GT_params = np.round(np.random.uniform(low = 0, high = 1, size = (args.num_of_sample, 1)), 4)
        GT_params = GT_params * (GT_max - GT_min) + GT_min
        lorenz_params_train = torch.from_numpy(GT_params)
        print('attention, you will repalce the saved training params.')
        pdb.set_trace()
        torch.save(lorenz_params_train, '{}/training_params.pth'.format(data_folder))
    else:
        lorenz_params_train = torch.load('{}/training_params.pth'.format(data_folder))

    print(lorenz_params_train.min(axis = 0), '\n', lorenz_params_train.max(axis = 0))

    if False:
        for param in range(10,20,1):
            print(param)
            ans = generate_l96_data(np.array([param, 0]))
            num_of_points = ans[::100][:200].reshape(-1).shape[0]
            plt.hist(ans[:200].reshape(-1)[:num_of_points])
            plt.savefig('imgs/hist_{}.pdf'.format(param))
            plt.close()
            plt.imshow(ans[:200].T, cmap = 'bwr')
            plt.savefig('imgs/{}.pdf'.format(param))
            plt.close()
            plt.hist(ans[::10][:200].reshape(-1)[:num_of_points])
            plt.savefig('imgs/hist_{}_10.pdf'.format(param))
            plt.close()
            plt.imshow(ans[::100][:200].T, cmap = 'bwr')
            plt.savefig('imgs/{}_100.pdf'.format(param))
            plt.close()
            plt.hist(ans[::100][:200].reshape(-1)[:num_of_points])
            plt.savefig('imgs/hist_{}_100.pdf'.format(param))
        # pdb.set_trace()

    traj_list = []
    n_workers = 50
    print(n_workers)
    # num = 500
    num = 100
    for i in range(0, int(args.num_of_sample/num)):
        split = 20
        assert num%split == 0
        for j in range(0,split):
            print(int(i*num + num/split*j), int(i*num + num/split*(j+1)))
            params = lorenz_params_train[int(i*num + num/split*j):int(i*num + num/split*(j+1))]
            params_cat_seed = np.concatenate([params, (random_index + np.arange(params.shape[0]) + int(i*num + j*num/split))[:, None]], axis = -1)

            time1 = time.time()
            with Pool(n_workers) as pool:
                total_traj = np.array(pool.map(generate_l96_data, params_cat_seed))
            for ix in range(params.shape[0]):
                torch.save({'0': params[ix], '1': total_traj[ix]}, '{}/{:06d}.pth'.format(data_folder, int(i*num+ix+num/split*j)))
            time2 = time.time()
            print('time used', time2 - time1)
