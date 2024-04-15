import argparse

parser = argparse.ArgumentParser(description='neural operator for longer time')
parser.add_argument('--noisy_scale', default = 0, type = float)
parser.add_argument('--embed_dim', default = 32, type = int)
parser.add_argument('--modes', default = 50, type = int)
parser.add_argument('--width', default = 32, type = int)

parser.add_argument('--crop_T', default = 200, type = int)
parser.add_argument('--x_len', default = 100, type = int)
parser.add_argument('--len_to_operator', default = 2, type = int)
parser.add_argument('--calculate_metric', default = 0, type = int)
parser.add_argument('--lambda_contra', default = 0, type = float)

parser.add_argument('--epochs', default = 501, type = int)
parser.add_argument('--batch_size', default = 100, type = int)
parser.add_argument('--batch_size_metricL', default = 100, type = int)
parser.add_argument('--learning_rate', default = 1e-3, type = float)

##################################################################
parser.add_argument('--l96', action='store_true')
parser.add_argument('--kse', action='store_true')
parser.add_argument('--subsample', action='store_true')
parser.add_argument('--ranked', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--with_geomloss', default = 0, type = int)
parser.add_argument('--with_geomloss_kd', default = 0, type = int)
parser.add_argument('--lambda_geomloss', default = 0, type = float)
parser.add_argument('--blur', default = 0.01, type = float)
parser.add_argument('--gpu', default = 0, type = int)
parser.add_argument('-m', '--train_metric', action = 'store_true') # set it true if you want to train metric net
parser.add_argument('--train_operator', action = 'store_true')
parser.add_argument('--eval_LE', action = 'store_true')
parser.add_argument('--lr_metric', default = 1e-4, type = float)
parser.add_argument('--metric_epochs', default = 0, type = int)
parser.add_argument('--training_size', default = 2000, type = int)
parser.add_argument('--bank_size', default = 0, type = int)
parser.add_argument('--seed', default = 34, type = int)
parser.add_argument('--T_metricL_traj_alone', default = 0.3, type = float)
parser.add_argument('--max_tau_metricL', default = 0.7, type = float)

parser.add_argument('--prefix', default = '', type = str)
parser.add_argument('--multiprocessing_distributed', action = 'store_true')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')


args = parser.parse_args()
