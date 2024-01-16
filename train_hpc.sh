#!/bin/bash
#SBATCH -q short
#SBATCH -t 02:00:00
#SBATCH -p gpu
#SBATCH -G rtx5000:1


# pty: interactive
# p: partition (gpu or sole cpu partitions, e.g. medium)
# G: specific gpu (e.g. gtx1080:2)
# N: number of nodes that the task should be executed on
# t: time resources are allocated
# c: number of cpus provided
# C: scratch

# squeue -u henrich1: queued and active jobs of me
# scancel -u henrich1: cancels all my jobs

source activate exenv
python exercises/weight_regression/train_weight_regression.py