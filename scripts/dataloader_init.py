import os, sys
import torch
import pdb
path = os.getcwd()
os.chdir(path)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def init_dataloader(args):
    val_btz = 100
    batch_size = args.batch_size
    if args.l96:
        from dataloader.dataloader_l96 import TrainingData, TestingData

    if args.l96:
        total_batch = args.batch_size_metricL * args.world_size
        train_dataset = TrainingData(args.x_len, train_size = args.training_size, noisy_scale=args.noisy_scale,  \
                        infinite_data_loader = True, total_epoch = max(1, args.metric_epochs*int(args.training_size/total_batch)), total_batch_size = total_batch)
        val_dataset = TrainingData(args.x_len, train_size = val_btz, noisy_scale=args.noisy_scale, validation = True)
        train_dataset_operator = TrainingData(args.x_len, train_size = args.training_size, \
                        noisy_scale=args.noisy_scale, train_operator = True)

    args.data_path = train_dataset.data_list[0]

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = False) # don't shuffle here cause we use inifinite dataloader
        train_sampler_operator = torch.utils.data.distributed.DistributedSampler(train_dataset_operator, shuffle = True)
    else:
        train_sampler = None
    train_loader_metric = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_metricL, \
                    shuffle=(train_sampler is None), num_workers=12, pin_memory=True, \
                    sampler=train_sampler, drop_last = True, persistent_workers = True)
    val_sampler = None
    val_loader_metric = torch.utils.data.DataLoader(val_dataset, batch_size=val_btz, \
                    shuffle=(val_sampler is None), num_workers=4, pin_memory=True, \
                    sampler=val_sampler, drop_last = True, persistent_workers = True)
    train_loader_operator = torch.utils.data.DataLoader(train_dataset_operator, batch_size=batch_size, \
                    shuffle=(train_sampler is None), num_workers=16, pin_memory=True, \
                    sampler=train_sampler_operator, drop_last = True, persistent_workers = True)

    for i in range(10):
        train_dataset[i]
        train_dataset_operator[i]

    return train_dataset, train_dataset_operator, train_loader_metric, val_loader_metric, train_loader_operator, train_sampler, train_sampler_operator
