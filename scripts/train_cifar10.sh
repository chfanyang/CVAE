#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
python src/train.py --config configs/cifar10.yaml 2>&1 | tee logs/train_cifar10.log
