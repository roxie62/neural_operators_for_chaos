## neural_operators_for_chaos

<p align="center">
  <img src="https://github.com/roxie62/neural_operators_for_chaos/blob/main/plots/diagram_emulator.png" width="700">
</p>

This is the implementation for the NeurIPS 2023 paper "[Training neural operators to preserve invariant measures of chaotic attractors](https://openreview.net/pdf?id=8xx0pyMOW1)".

```
@article{jiang2024training,
  title={Training neural operators to preserve invariant measures of chaotic attractors},
  author={Jiang, Ruoxi and Lu, Peter Y and Orlova, Elena and Willett, Rebecca},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

### Optimal transport method
This implementation supports DistributedDataParallel training: The default configuration uses 4 GPUs.
The default algorithm used to solve the optimal transport problem is [Sinkhorn divergence](https://www.kernel-operations.io/geomloss/).

To run Optimal transport (OT) method for Lorenz 96 data, run:
```
bash experiments/OT_l96/srun.sh
```
The hyperparameters of the OT method are:
- Weights of the OT loss: `--lambda_geomloss 3`.
- Regularization value in the Sinkhorn algorithm, where smaller regularization usually leads to high accuracy while slowing the training: `--blur 0.02`.
- Controlling the physical knowledge you want to use during the training, set this to be larger than 0 if you only have partial knowledge of the system, `--with_geomloss_kd 0`



### Contrastive feature learning method
This implementation supports DistributedDataParallel training: The default configuration uses 4 GPUs.

To run contrastive learning (CL) method for Lorenz 96 data, run:
```
bash experiments/CL_l96/srun.sh
```
