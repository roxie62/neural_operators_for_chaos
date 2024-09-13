import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob, pdb, random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

path = os.getcwd()
os.chdir(path)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

def filter_invalid_index(data_list, params_array, invalid_idx):
    if invalid_idx.shape[0] == 0:
        return data_list, params_array
    all_idx = np.array([int(data_path.split('/')[-1].split('.')[0]) for data_path in data_list])
    invalid_mask = (all_idx[:,None] == invalid_idx[None,:]).sum(axis = -1) > 0
    data_list = np.array(data_list)[~invalid_mask].tolist()
    params_array = params_array[~invalid_mask]
    return data_list, params_array

def crop(traj, Tsidx, Tlen):
    assert Tsidx.shape[0] == traj.shape[0]
    traj_list = [traj[i, Tsidx[i]:Tsidx[i] + Tlen] for i in range(Tsidx.shape[0])]
    return torch.stack(traj_list)

@torch.no_grad()
def random_cropping(traj, crop_T):
    T = traj.shape[1]
    assert T >= crop_T
    Tsidx = torch.randint(50, T - crop_T, (traj.shape[0],1))[:,0]
    train_crop = crop(traj, Tsidx, crop_T)
    return train_crop

@torch.no_grad()
def random_cond_contra_cropping(traj, crop_T, no_align = True):
    T = traj.shape[1]
    assert T >= crop_T
    Tsidx_1 = torch.randint(50, T - int(1 * crop_T), (traj.shape[0],1))[:,0]
    train_crop_1 = crop(traj, Tsidx_1, crop_T)
    if no_align:
        index = np.arange(50, T-crop_T)
        in_index = np.arange(Tsidx_1, Tsidx_1 + crop_T)
        out_index = np.setdiff1d(index, in_index)
        Tsidx_2 = torch.tensor([out_index[torch.randint(out_index.shape[0], size = (1, 1))[0]]])
    train_crop_2 = crop(traj, Tsidx_2, crop_T)
    return train_crop_1[0], train_crop_2[0]

class TrainingData(Dataset):
    def __init__(self, crop_T, convert_to_pil = False, train_size = 1000, \
                noisy_scale = 0, validation = False, train_operator = False, \
                infinite_data_loader = False, total_epoch = 1, \
                total_batch_size = 1000):
        self.crop_T = crop_T
        self.convert_to_pil = convert_to_pil
        if validation:
            self.data_path = f'{parent}/l96_data_x/F_2000_dt10_validation'
        else:
            self.data_path = f'{parent}/l96_data_x/F_2000_dt10'
        self.T = 2000
        self.noisy_scale = noisy_scale
        self.train_operator = train_operator
        data_path = self.data_path
        if self.convert_to_pil:
            self.data_list = glob.glob('{}/0*.pth'.format(data_path))
            self.data_list.sort()
            self.params_array_list = torch.load('{}/training_params.pth'.format(self.data_path))
        else:
            self.data_list = glob.glob('{}/00*_noise_{:.2f}.tiff'.format(data_path, self.noisy_scale))
            print('use {} data'.format(noisy_scale), self.data_list[0])
            self.data_list.sort()
            self.params_array_list = torch.load('{}/training_params.pth'.format(data_path))
            self.nan_list = torch.load('{}/nan_list.pth'.format(data_path))
            self.data_list, self.params_array_list = filter_invalid_index(self.data_list, self.params_array_list, self.nan_list)
            self.data_list = self.data_list[:train_size]
            self.params_array_list = self.params_array_list[:train_size]
            print('after filter nan data, we have {} size data'.format(len(self.data_list)))

        self.total_epoch = total_epoch
        self.infinite_data_loader = infinite_data_loader
        if self.infinite_data_loader:
            self.infinite_data_idx = torch.cat([torch.randperm(len(self.data_list))[:total_batch_size] for _ in range(self.total_epoch)])
            self.infinite_data_idx = self.infinite_data_idx.cuda(torch.distributed.get_rank())
            torch.distributed.broadcast(self.infinite_data_idx, 0)
            self.infinite_data_idx = self.infinite_data_idx.cpu()


    def __len__(self):
        if self.infinite_data_loader:
            return self.infinite_data_idx.shape[0]
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        if self.infinite_data_loader:
            idx = self.infinite_data_idx[idx].item()
        if self.convert_to_pil:
            anchor_d = torch.load(self.data_list[idx])
            anchor_param, anchor_t = anchor_d['0'], anchor_d['1']
            filter_nan = np.isnan(anchor_t.sum())
            return anchor_param, anchor_t, idx, filter_nan
        else:
            T, num_of_param, dim = self.T, 1, 60
            data_1d_recon = np.array(Image.open(self.data_list[idx])).reshape(-1)
            anchor_param, anchor_t = data_1d_recon[:num_of_param], data_1d_recon[num_of_param:].reshape(T, dim)
            if self.train_operator:
                if self.crop_T < self.T:
                    crop = random_cropping(torch.from_numpy(anchor_t[None, :, :]), self.crop_T)
                else:
                    crop = torch.from_numpy(anchor_t)
                return anchor_param, crop
            else:
                crop_1, crop_2 = random_cond_contra_cropping(torch.from_numpy(anchor_t[None, :, :]), \
                                self.crop_T, True)
                if anchor_param.dtype == np.float32 or anchor_param.dtype == np.float64:
                    anchor_param = torch.from_numpy(anchor_param).float()
                return anchor_param, crop_1, crop_2

class TestingData(Dataset):
    def __init__(self, crop_T, noisy_scale = 0, convert_to_pil = False, train_size = 1000):
        self.crop_T = crop_T
        self.convert_to_pil = convert_to_pil
        self.noisy_scale = noisy_scale
        self.data_path = f'{parent}/l96_data_x/F_2000_dt10_test'
        self.T = 2000
        self.params_array_list = torch.load('{}/training_params.pth'.format(self.data_path))
        if self.convert_to_pil:
            self.data_list = glob.glob('{}/00*.pth'.format(self.data_path))
        else:
            self.data_list = glob.glob('{}/00*_noise_{:.2f}.tiff'.format(self.data_path, self.noisy_scale))
        self.data_list.sort()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.convert_to_pil:
            anchor_d = torch.load(self.data_list[idx])
            anchor_param, anchor_t = anchor_d['0'], anchor_d['1']
            filter_nan = np.isnan(anchor_t.sum())
            assert filter_nan == False
            T, num_of_param, dim = self.T, 1, 60
            return anchor_param, anchor_t, idx, filter_nan
        else:
            T, num_of_param, dim = self.T, 1, 60
            data_1d_recon = np.array(Image.open(self.data_list[idx])).reshape(-1)
            anchor_param, anchor_t = data_1d_recon[:num_of_param], data_1d_recon[num_of_param:].reshape(T, dim)
            return torch.from_numpy(anchor_param), anchor_t


if __name__ == '__main__':
    train_size = 2000
    crop_T = 2000
    def save_data_with_pil(dataset):
        nan_list = []
        print(dataset.data_path)
        for i in range(len(dataset)):
            anchor_param, anchor_t, idx, filter_nan = dataset[i]
            len_total = anchor_t.shape[0]
            anchor_t = torch.from_numpy(anchor_t)
            traj_x= anchor_t.clone()[None, :, :]
            std_x = traj_x.std(dim = 1)[:, None, :].repeat(1, traj_x.shape[1], 1)
            noise_traj_x = ( noisy_scale * std_x * (torch.randn(traj_x.shape, device = traj_x.device)) ).squeeze()
            traj = traj_x.squeeze() + noise_traj_x
            anchor_t = traj
            anchor_t = anchor_t.cpu().data.numpy()
            save_pil = True
            if save_pil:
                nan_list.append(idx) if filter_nan else print(idx)
                assert anchor_t.shape[0] == len_total
                data_1d = np.concatenate([anchor_param.reshape(-1), anchor_t.reshape(-1).astype(np.float32)])
                img_data = Image.fromarray(data_1d)
                img_data.save('{}/{:06d}_noise_{:.2f}.tiff'.format(dataset.data_path, idx, noisy_scale))
        if save_pil:
            torch.save(np.array(nan_list), dataset.data_path + '/nan_list.pth')
    noisy_scale = 0.3
    training_data_noisy = TrainingData(crop_T = crop_T, convert_to_pil = True, noisy_scale= noisy_scale, train_size = train_size)
    validation_data_noisy = TrainingData(crop_T = crop_T, convert_to_pil = True, noisy_scale= noisy_scale, train_size = train_size, validation=True)
    testing_data_noisy = TestingData(crop_T = crop_T, convert_to_pil = True, noisy_scale = noisy_scale)
    save_data_with_pil(training_data_noisy)
    save_data_with_pil(validation_data_noisy)
    save_data_with_pil(testing_data_noisy)
    noisy_scale = 0
    testing_data = TestingData(crop_T = crop_T, convert_to_pil = True, noisy_scale = 0)
    save_data_with_pil(testing_data)
