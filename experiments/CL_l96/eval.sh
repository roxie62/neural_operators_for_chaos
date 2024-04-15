#!/bin/bash

export NCCL_DEBUG=info
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=${port} scripts/main.py \
  --l96 \
  --batch_size 25 \
  --metric_epochs 1000 \
  --batch_size_metricL 250 \
  --bank_size 1000 \
  --embed_dim 128 \
  --modes 28 \
  --width 64 \
  --lambda_contra 0.8 \
  --x_len 100 \
  --noisy_scale 0.3 \
  --eval_LE \
