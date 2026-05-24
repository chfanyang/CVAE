#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs
python src/train.py --config configs/svhn.yaml 2>&1 | tee logs/train_svhn.log
