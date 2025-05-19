#!/bin/bash
#BSUB -J single_node
#BSUB -o /home/phd_li/git_repo/MT-DinoSeg/logs/train_%J.log
#BSUB -e /home/phd_li/git_repo/MT-DinoSeg/logs/train_%J.log
#BSUB -n 16             
#BSUB -q gpu
#BSUB -gpu "num=2"   
#BSUB -R "select[ui==aiml_python && osrel==70 && type==X64LIN]"

# Configuration
MASTER_PORT=$((29500 + LSB_JOBID % 1000))
NPROC_PER_NODE=2

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
    --experiment segmentation_r2s100k \
    --model_name dino_vits8-linear_probing \
    --epochs 500 \
    --num_workers 8 \
    --batch_size 128

echo ""
echo "Training completed at $(date)"
