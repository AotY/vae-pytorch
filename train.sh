#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

python train.py \
    --batch_size 64 \
    --epochs 20 \
    --device cuda \
    --seed 7 \
    --log_interval 20

/

