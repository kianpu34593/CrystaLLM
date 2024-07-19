#!/usr/bin/bash
#SBATCH -p venkvis-h100
#SBATCH -N 2
#SBATCH --gpus-per-node=2         # Number of GPU(s) per node
#SBATCH -t 0-8:00:00
#SBATCH -J diffmix 
#SBATCH --mem=320000 # Memory pool for all cores in MB
##SBATCH --gres=gpu:h100:4
#SBATCH -e train_N2n2_%j.err
#SBATCH -o train_N2n2_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=shangzhu@alumni.cmu.edu # Email to which notifications will be sent

#https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html

# Create job-specific tmp directory
export TMPDIR=$(mktemp --directory --tmpdir)
# Ensure Cleanup on exit
trap 'rm -rf -- "$TMPDIR"' EXIT
echo "Job started on `hostname` at `date`"


#https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
#https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
export MASTER_PORT=1234
export WORLD_SIZE=4
echo ${WORLD_SIZE}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ${MASTER_ADDR}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo ${head_node_ip}

source activate crystallm

# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT bin/train_ddp.py --config=config/crystallm_v2_large.yaml
# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT bin/train_ddp.py --config=config/crystallm_v2_large.yaml
# srun torchrun --nproc_per_node=2 --nnodes=2 --rdzv_id=$MASTER_PORT --rdzv_endpoint=$MASTER_ADDR --rdzv_backend=c10d bin/train_ddp.py --config=config/crystallm_v2_large.yaml
# srun torchrun --nproc_per_node=2 --nnodes=2 --rdzv_id=$MASTER_PORT --rdzv_endpoint=$MASTER_ADDR --rdzv_backend=c10d bin/train_ddp.py --config=config/crystallm_v2_large.yaml
srun torchrun --nproc_per_node=2 --nnodes=2 --rdzv_id=$RANDOM --rdzv_endpoint=$head_node_ip:29500 --rdzv_backend=c10d bin/train_ddp.py --config=config/crystallm_v3_large_N2n2.yaml

#tutorials
# https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html
# https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html
#refs
# https://github.com/karpathy/nanoGPT

# source crystallm_venv/bin/activate

# what's token per iteration?
# python bin/train.py --config=config/crystallm_v1_small.yaml
# torchrun --standalone --nproc_per_node=4 bin/train_ddp.py --config=config/crystallm_v2_large.yaml
# torchrun --standalone --nproc_per_node=1 bin/train_ddp.py --config=config/crystallm_v1_small.yaml
echo " "
echo "Job Ended at `date`"