#!/usr/bin/env bash
##

#USAGE
### THESE COMMAND BELOW TO DO MULTI NODE TRAINING
### first node run:  NODE_RANK=0 NNODES=n tools/multi_dist_train.sh tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences_debug.yaml
### second node run: NODE_RANK=1 NNODES=n tools/multi_dist_train.sh tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences_debug.yaml
### 3rd node run:    NODE_RANK=2 NNODES=n tools/multi_dist_train.sh tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences_debug.yaml
### .
### .
### .
### nth node run:    NODE_RANK=n-1 NNODES=n tools/multi_dist_train.sh tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences_debug.yaml

NCCL_IB_HCA=$(
  pushd /sys/class/infiniband/ >/dev/null
  for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i; done
  popd >/dev/null
)
# [ -z "$NCCL_IB_HCA"] && NCCL_IB_HCA=mlx4_1;
# export NCCL_IB_HCA
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=106
export NCCL_IB_DISABLE=1 # skip InfiniBand
export NCCL_SOCKET_IFNAME=bond0

NNODES=${NNODES:-2}       ##Node nums
NODE_RANK=${NODE_RANK:-1} ##Node rank of different machine
CONFIG=$1
GPUS=4 ##Num gpus of a worker

PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"10.124.227.158"}

if [[ $NODE_RANK == 0 ]]; then
  echo "Write the ip address of node 0 to the hostfile.txt"
  ifconfig -a | grep 10.15 | grep -v 127.0.0.1 | grep -v inet6 | awk '{print $2}' | tr -d "addr:" >hostfile.txt
fi
MASTER_ADDR=$(cat hostfile.txt)
echo "MASTER_ADDR is : $MASTER_ADDR"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
  python3 -m torch.distributed.launch \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  $(dirname "$0")/train.py \
  --cfg_file \
  $CONFIG \
  --launcher \
  pytorch ${@:3}
