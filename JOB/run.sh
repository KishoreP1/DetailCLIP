#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=JOBNAME
#SBATCH --time=DESIRED_TIME
#SBATCH --nodes=4 --ntasks-per-node=4  --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --output=./log/JOBNAME.txt
 

## Networking for DDP
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export MASTER_PORT=29500 # default port
echo "NODELIST="${SLURM_NODELIST}
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# modules
module load cuda/11.8.0
module load miniconda3

# env activation
source activate CLIP

start_time=$(date +%s)

srun python3 main.py \
--model DetailCLIP_VITB16 \
--dataset yfcc15m \
--metadata yfcc15m.pkl \
--output-dir output/$SLURM_JOB_NAME \
--mask-ratio 0.5 \
--epochs 50 \
--batch-size 256 \
--lr 5e-4 \
--wd 0.5 \
--workers $SLURM_CPUS_PER_TASK \
--clip_loss_weight 1 \
--ibot_patch_loss_weight 1 \
--ibot_cls_loss_weight 1 \
--reconst_loss_weight 1 \
--print-freq 1000

end_time=$(date +%s)
execution_time=$((end_time - start_time))