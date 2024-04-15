import torch, pdb, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import math

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

class skip_embed_final_shallow(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, norm_layer):
        super().__init__()
        self.embed = nn.Sequential(
                               nn.Linear(input_dim, output_dim),
                               norm_layer(output_dim),
                               ACT(inplace = True),
                               )
        self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.embed(input) + self.skip(input)

ACT = nn.ReLU

class MetricNet(torch.nn.Module):
    def __init__(self, embed_dim, args = None):
        super().__init__()
        if args.l96:
            from models.l96x_resnet import resnet34
            self.encoder = resnet34(num_classes = embed_dim, norm_layer = nn.BatchNorm2d)
        self.proj_head = skip_embed_final_shallow(512, 512, embed_dim, nn.BatchNorm1d)
        # create the queue
        self.register_buffer("traj_queue_embed", torch.randn(embed_dim, args.bank_size))
        self.traj_queue_embed = nn.functional.normalize(self.traj_queue_embed, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, traj_embed_keys):
        traj_embed_keys = concat_all_gather(traj_embed_keys)
        batch_size = traj_embed_keys.shape[0]
        K = self.traj_queue_embed.shape[1]
        ptr = int(self.queue_ptr)
        assert K % batch_size == 0  # for simplicity
        self.traj_queue_embed[:, ptr:ptr + batch_size] = traj_embed_keys.T
        ptr = (ptr + batch_size) % K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, traj, train_operator = False):
        traj = traj.permute(0, 2, 1)[:, None, :, :]
        x0, x1, x2, x3, x4, x = self.encoder(traj)
        embed = self.proj_head(x)
        if train_operator:
            return [x0, x1, x2, x3, x4, F.normalize(embed, dim = -1)]
        else:
            return F.normalize(embed, dim = -1)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
     """
     Performs all_gather operation on the provided tensors.
     *** Warning ***: torch.distributed.all_gather has no gradient.
     """
     tensors_gather = [torch.ones_like(tensor)
         for _ in range(torch.distributed.get_world_size())]
     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

     output = torch.cat(tensors_gather, dim=0)
     return output
