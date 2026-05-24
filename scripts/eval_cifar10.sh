#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3
python src/evaluate.py --config configs/cifar10.yaml
