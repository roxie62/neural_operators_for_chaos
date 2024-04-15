import torch
import torch.nn.functional as F

class OT_measure:
    def __init__(self, with_geomloss, blur):
        from geomloss import SamplesLoss
        if with_geomloss == 1:
            self.loss_geom = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend = 'online')
            print('You have selected to use Sinkhorn loss.')
        elif with_geomloss == 2:
            self.loss_geom = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend = 'online')
            print('You have selected to use MMD loss.')

    def loss(self, anchor_stats, out_stats):
        normalized_std = anchor_stats.std(dim = 1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
        normalized_mean = anchor_stats.mean(dim = 1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
        out_stats = (out_stats-normalized_mean) / normalized_std
        anchor_stats = (anchor_stats - normalized_mean) / normalized_std
        loss_geom_list = self.loss_geom(anchor_stats.contiguous(), out_stats.contiguous())
        loss_geom_mean = loss_geom_list.mean()

        return loss_geom_mean
