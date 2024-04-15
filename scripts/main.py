# TORCH_DISTRIBUTED_DEBUG=INFO
# TORCH_DISTRIBUTED_DEBUG=DETAIL
import builtins, json, warnings, pdb, os, time, sys, torch, torch.optim, torch.utils.data, torch.utils.data.distributed, random
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from log import create_folder_path
path = os.getcwd()
os.chdir(path)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from timeit import default_timer
from utils import init_distributed_mode, HiddenPrints
from eval_scripts.eval_l96 import eval_l96
from tqdm import tqdm
from configuration import args
from scripts.dataloader_init import init_dataloader
from scripts.train_utils import LpLoss_, adjust_learning_rate_cos, visualiztion, plot_loss, save_operator
from models.fno_1d_new import FNO1d

def main(args):
    print(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('start')
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.gpu = args.gpu % torch.cuda.device_count()
    print('world_size', args.world_size)
    print('rank', args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.is_master = args.rank % ngpus_per_node == 0 and args.gpu == 0
    train_dataset, train_dataset_operator, train_loader_metric, val_loader_metric, \
        train_loader_operator, train_sampler, train_sampler_operator =  init_dataloader(args)
    prefix_for_CL, contra_models_path, operators_path, output_path = create_folder_path(args)

    ############################################################################
    #####Training CL's encoder (this step could be skipped for OT method) ######
    if args.metric_epochs > 0:
        from scripts.CL_utils import CL
        CL = CL(args, train_loader_metric, train_sampler, val_loader_metric)
        saved_pth_contra = f'{contra_models_path}/{prefix_for_CL}'
        if args.train_metric:
            CL.train_metric_net(output_path, saved_pth_contra)
        else:
            metric_net = CL.load_metric_net(f'{saved_pth_contra}/{args.metric_epochs:03d}.pth', args.gpu)
        if args.is_master:
            from scripts.CL_utils import cal_topk_eval
            val_btz = 100
            val_acc = cal_topk_eval(val_loader_metric, val_btz, CL.metric_net, args, pth=output_path)
            train_metric_ = '' if args.train_metric else '0'
            with open(f'{output_path}/CL_encoder_quality.txt', 'w') as f:
                f.writelines('begin evaluation \n')
                f.writelines('val quality {} \n'.format(val_acc))
        CL.metric_net.eval()

    ###########################################################################
    ################### training of the neural operator #######################
    operator = FNO1d(args.modes, args.width)
    operator.to(args.gpu)
    learning_rate, epochs = args.learning_rate, args.epochs
    optimizer = torch.optim.AdamW(operator.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_epoch_ = 0
    if args.distributed:
        operator = torch.nn.parallel.DistributedDataParallel(operator, device_ids=[args.gpu], find_unused_parameters = False, broadcast_buffers = False)
    operator = nn.SyncBatchNorm.convert_sync_batchnorm(operator)
    from scripts.OT_utils import OT_measure
    OT_measure = OT_measure(args.with_geomloss, args.blur)
    from train_utils import long_length_predict_with_yinit
    from scripts.cal_stats_l96 import cal_stats_l96

    if args.train_operator:
        loss_list, ep_loss = [], []
        for ep in tqdm(range(epochs)):
            if args.distributed:
                train_sampler_operator.set_epoch(ep)
            for param, y in train_loader_operator:
                operator.train()
                l2, loss_OT, loss_CL = torch.tensor([0]).to(args.gpu).float(), torch.tensor([0]).to(args.gpu).float(), torch.tensor([0]).to(args.gpu).float()
                lr_ = adjust_learning_rate_cos(args.learning_rate, optimizer, ep, epochs, args)
                param, y = param.to(args.gpu), y.to(args.gpu).squeeze()
                assert args.x_len <= y.shape[1]
                assert y.shape[0] == args.batch_size
                y_predict = long_length_predict_with_yinit(operator, y, param, args.x_len, args.len_to_operator)
                l2 += LpLoss_(2).rel(y_predict, y)
                if args.metric_epochs > 0 and ep >= 0:
                    loss_CL = CL.calculate_CL_loss(y, y_predict)
                if args.with_geomloss > 0 and ep >= 20:
                    if args.l96:
                        anchor_stats, out_stats = cal_stats_l96(y.squeeze(), y_predict.squeeze(), args = args, index = args.with_geomloss_kd)
                        if args.with_geomloss_kd != 0:
                            anchor_stats, out_stats = anchor_stats[:,:,np.array([args.with_geomloss_kd-1])], out_stats[:,:,np.array([args.with_geomloss_kd-1])]
                    assert anchor_stats.shape[0] == args.batch_size
                    loss_OT = OT_measure.loss(anchor_stats, out_stats)

                optimizer.zero_grad()
                loss =  l2 \
                        + args.lambda_contra * loss_CL \
                        + args.lambda_geomloss *  loss_OT
                loss.backward()
                optimizer.step()
                ep_loss.append([l2.item(), \
                                args.lambda_contra * loss_CL.cpu().data.numpy().item(), \
                                args.lambda_geomloss * loss_OT.item()])
                loss_list.append([np.array(ep_loss).mean(axis = 0)[0], np.array(ep_loss).mean(axis = 0)[1], \
                                    np.array(ep_loss).mean(axis = 0)[2]])

            if ep% 50 == 0 and ep > 0:
                visualiztion(train_dataset_operator, operator, args, img_pth=f'{output_path}/training_vis', ep=ep)
                plot_loss(loss_list, img_pth = f'{output_path}/training_loss_operator')

        if ep == epochs - 1:
            if args.is_master:
                save_operator(operator, optimizer, saved_pth=f'{operators_path}/{args.prefix}/{ep:03d}', ep=ep)

    ###########################################################################
    #################### load the model and evaluation ########################
    ep = args.epochs - 1
    from eval_scripts.eval_utils import load_operator
    operator  = load_operator(operator, saved_pth = f'{operators_path}/{args.prefix}/{ep:03d}')
    visualiztion(train_dataset_operator, operator, args, img_pth = f'{output_path}/training_vis', ep=ep)
    ############### evaluate the statistics and save them ######################
    if args.l96:
        eval_len_list = [1500]
    for eval_len in eval_len_list:
        x_len = args.x_len
        eval_l96(operator, args, args.noisy_scale, x_len = eval_len, calculate_l2 = True, output_path = output_path)
        eval_l96(operator, args, 0, x_len = eval_len, calculate_l2 = True, output_path = output_path)
    if args.eval_LE:
        from eval_scripts.eval_LE import cal_LE
        LE_results = cal_LE(operator, args)


if __name__ == '__main__':
    main(args)
