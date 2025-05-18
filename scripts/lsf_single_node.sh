#!/bin/bash
#BSUB -J single_node
#BSUB -o /home/phd_li/git_repo/MT-DinoSeg/logs/train_%J.log
#BSUB -e /home/phd_li/git_repo/MT-DinoSeg/logs/train_%J.log
#BSUB -n 8             
#BSUB -q gpu
#BSUB -gpu "num=4"   
#BSUB -R "select[ui==aiml_python && osrel==70 && type==X64LIN]"

# Configuration
MASTER_PORT=$((29500 + LSB_JOBID % 1000))
NPROC_PER_NODE=4

echo "=== Single-Node Distributed Training ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Master Port: $MASTER_PORT"
echo "GPUs: $NPROC_PER_NODE"
echo ""

echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

# Single GPU with torchrun
torchrun \
    --standalone \
    --nproc-per-node=$NPROC_PER_NODE \
    train.py \
    --experiment dino_segmentation_r2s100k \
    --epochs 300 \
    --num_workers 2 \
    --batch_size 256

echo ""
echo "Training completed at $(date)"
