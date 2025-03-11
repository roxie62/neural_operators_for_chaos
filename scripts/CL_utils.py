from scripts.train_utils import adjust_learning_rate_cos, adjust_tau_metricL
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch, pdb
import matplotlib.pyplot as plt

def visualize_lossCT(traj_top1_list, loss_list, img_pth):
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(traj_top1_list[-100:])), np.array(traj_top1_list[-100:]))
    plt.title('topk')
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(loss_list[-100:])), np.array(loss_list[-100:]))
    plt.title('metric loss')
    plt.savefig(f'{img_pth}.png')
    plt.close()

class CL:
    def __init__(self, args, train_loader_metric, train_sampler, val_loader_metric):
        from models.metricL_utils import MetricNet
        metric_net = MetricNet(args.embed_dim, args = args)
        if args.distributed:
            metric_net.to(args.gpu)
            metric_net = torch.nn.parallel.DistributedDataParallel(metric_net, device_ids=[args.gpu], find_unused_parameters = False, broadcast_buffers = False)
        self.metric_net = nn.SyncBatchNorm.convert_sync_batchnorm(metric_net)
        self.optimizer_metric = torch.optim.AdamW(metric_net.parameters(), lr = args.lr_metric, weight_decay=1e-5)
        self.metric_epochs = args.metric_epochs
        self.train_loader_metric = train_loader_metric
        self.train_sampler = train_sampler
        self.gpu = args.gpu
        self.lr_metric = args.lr_metric
        self.T_metricL_traj_alone = args.T_metricL_traj_alone
        self.max_tau_metricL = args.max_tau_metricL
        self.bank_size = args.bank_size
        self.distributed = args.distributed
        self.args = args
        self.batch_size_metricL = args.batch_size_metricL
        ## Normally, the dataloader at every epoch should give you the whole training data once.
        ## Here, we create an infinite dataloader. And the ep_ratio here corresponds to the 1/(number of 
        ## iterations a normal dataloader usually requires).
        self.ep_ratio = int(args.training_size / (args.batch_size_metricL * args.world_size))
        self.val_loader_metric = val_loader_metric

    def train_metric_net(self, train_vis_pth = '', saved_pth_contra = ''):
        traj_top1_list, loss_list = [], []
        train_acc, val_acc = 0, 0
        ep_real = 0 ## As we use an infinite dataloader for FAST training, we use ep_real to count the real epoch number.
        iter_real = 0
        ## This script uses infinite datalaoder, which has an "augmented" dataset that included #epochs times larger dataset.
        ## So we manually set the metric_epochs_train here because one epoch will go through the dataset #epochs times.
        metric_epochs_train = 1
        self.metric_net.train()

        for ep in tqdm(np.arange(metric_epochs_train)):
            for param, x, y in tqdm(self.train_loader_metric): ## x, y positive samples
                if ep_real % 1 == 0:
                    lr_ = adjust_learning_rate_cos(self.lr_metric, self.optimizer_metric, ep_real, self.metric_epochs, self.args)
                    T_metricL_traj_alone = adjust_tau_metricL(self.T_metricL_traj_alone, ep_real, self.metric_epochs, epochs = 100, max_tau_metricL = self.max_tau_metricL)
                
                ## compute the features of the anchor and positive samples
                cat_embed = self.metric_net.forward(torch.cat([x, y]).to(self.gpu))
                anchor_embed, pos_anchor_embed = cat_embed[:self.batch_size_metricL, :], cat_embed[self.batch_size_metricL:, :]
                
                if self.bank_size > 0:
                    traj_embed_queue = self.metric_net.module.traj_queue_embed.detach().clone()
                    pos_anchor_embed_ = torch.cat([pos_anchor_embed, traj_embed_queue.T], dim = 0)
                else:
                    pos_anchor_embed_ = pos_anchor_embed

                labels = torch.arange(anchor_embed.shape[0], device = self.gpu)
                sim = anchor_embed @ pos_anchor_embed_.T
                loss = torch.nn.CrossEntropyLoss()(sim / T_metricL_traj_alone, labels)
                # Backward
                self.optimizer_metric.zero_grad()
                loss.backward()
                self.optimizer_metric.step()
                if self.bank_size > 0:
                    with torch.no_grad():
                        self.metric_net.module._dequeue_and_enqueue(anchor_embed)

                if ep_real%50 == 0 and ep_real > 0:
                    traj_head = sim
                    topk_embed = traj_head.topk(1, dim = -1)[1]
                    topk_acc = ((topk_embed == torch.arange(topk_embed.shape[0]).to(self.gpu)[:,None]).sum(dim = -1) > 0).float().mean().item()
                    print(topk_acc)
                    traj_top1_list.append(topk_acc)
                    loss_list.append(loss.cpu().data.numpy())
                    val_acc = cal_topk_eval(self.val_loader_metric, 100, self.metric_net, self.args, pth=train_vis_pth)
                    self.metric_net.train()
                if ep_real%100 == 1 or ep_real == self.metric_epochs - 1:
                    visualize_lossCT(traj_top1_list, loss_list, f'{train_vis_pth}/CT')
                iter_real += 1
                ep_real = iter_real / self.ep_ratio

        if self.gpu == 0:
            torch.save({'epoch': ep_real, 'state_dict': self.metric_net.module.state_dict(), \
                        'optimizer_state_dict': self.optimizer_metric.state_dict()}, f'{saved_pth_contra}/{int(ep_real):03d}.pth')

    def load_metric_net(self, saved_pth_contra, gpu):
        checkpoint = torch.load(saved_pth_contra, map_location = 'cuda:{}'.format(gpu))
        print('\n contra model saved path', saved_pth_contra)
        checkpoint = {key.replace('module.','') : val for key, val in checkpoint['state_dict'].items()}
        try:
            self.metric_net.module.load_state_dict(checkpoint)
        except:
            self.metric_net.load_state_dict(checkpoint)
        self.metric_net.eval()
        for param in self.metric_net.parameters():
            param.requires_grad = False

    def calculate_CL_loss(self, y, out):
        self.metric_net.eval()
        assert y.shape == out.shape
        embed_y = self.metric_net.module(y.squeeze(), train_operator = True)
        embed_out = self.metric_net.module(out.squeeze(), train_operator = True)
        embed_dist = torch.tensor([0]).float().to(y.device)

        for ey, eo in zip(embed_y[0:], embed_out[0:]):
            embed_dist -= (F.normalize(ey, dim = 1) * F.normalize(eo, dim = 1)).sum(dim = 1).mean()
        embed_dist = 1/len(embed_y[0:]) * embed_dist
        # self.metric_net.eval()
        # embed_y = self.metric_net.module(y, train_operator = True)
        # embed_out = self.metric_net.module(out, train_operator = True)
        # embed_dist = torch.stack([-(F.normalize(ey, dim=1) * F.normalize(eo, dim=1)).sum(dim=1).mean() \
        #             for ey, eo in zip(embed_y, embed_out)]).sum().mean()
        return embed_dist


def cal_topk_eval(dataloader, btz_eval, metric_net, args, pth = 'img'):
    acc_list = []
    metric_net.eval()
    with torch.no_grad():
        for iter in range(1):
            anchor_list =[]
            pos_list = []
            param_list = []
            for param, x, y in dataloader:
                x, y = x.to(args.gpu), y.to(args.gpu)
                param = param.to(args.gpu).float()
                cat_embed = metric_net.forward(torch.cat([x, y]))
                anchor_embed, pos_anchor_embed = cat_embed[:btz_eval, :], cat_embed[btz_eval:, :]
                anchor_list.append(anchor_embed)
                pos_list.append(pos_anchor_embed)
                param_list.append(param)

            params = torch.cat(param_list, dim = 0)
            anchor_embed = torch.cat(anchor_list, dim = 0)
            pos_anchor_embed = torch.cat(pos_list, dim = 0)
            # print(anchor_embed.shape)
            traj_head = anchor_embed @ pos_anchor_embed.T
            topk_embed = traj_head.topk(1, dim = -1)[1]
            topk_acc = ((topk_embed == torch.arange(topk_embed.shape[0]).to(args.gpu)[:,None]).sum(dim = -1) > 0).float().mean().item()
            topk_embed_5 = traj_head.topk(5, dim = -1)[1]
            topk_acc_5 = ((topk_embed_5 == torch.arange(topk_embed_5.shape[0]).to(args.gpu)[:,None]).sum(dim = -1) > 0).float().mean().item()
            acc_list.append([topk_acc, topk_acc_5])
    mean_acc = np.array(acc_list).mean(axis = 0)
    print('metric quality', mean_acc)

    params = params[:, 0]
    sort_index = params.sort()[1]
    params_sort = params[sort_index]
    anchor_embed = anchor_embed[sort_index]
    sim = anchor_embed @ anchor_embed.T
    plt.imshow(sim.cpu().data.numpy(), cmap = 'viridis')
    plt.colorbar()
    plt.savefig('{}/sim.pdf'.format(pth))
    print('metric quality', '{}'.format(mean_acc))
    return '{}'.format(mean_acc)
