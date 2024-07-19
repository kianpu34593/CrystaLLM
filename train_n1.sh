#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p venkvis-h100
#SBATCH -t 0-8:00:00
#SBATCH -J crystallm 
#SBATCH --mem=80000 # Memory pool for all cores in MB
#SBATCH --gres=gpu:h100:1
#SBATCH -e train_n1_%j.err
#SBATCH -o train_n1_%j.out # File to which STDOUT will be written %j is the job #
#SBATCH --mail-type=END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=shangzhu@alumni.cmu.edu # Email to which notifications will be sent

# Create job-specific tmp directory
export TMPDIR=$(mktemp --directory --tmpdir)
# Ensure Cleanup on exit
trap 'rm -rf -- "$TMPDIR"' EXIT
echo "Job started on `hostname` at `date`"

# source crystallm_venv/bin/activate
source activate crystallm
# what's token per iteration? the loss diverges or not? anad the model output?
# python bin/train.py --config=config/crystallm_v1_small.yaml
# torchrun --standalone --nproc_per_node=1 bin/train_ddp.py --config=config/crystallm_v2_large_n1.yaml
torchrun --standalone --nproc_per_node=1 bin/train_ddp.py --config=config/crystallm_small_v1.yaml
echo " "
echo "Job Ended at `date`"