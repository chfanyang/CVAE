#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2
python src/evaluate.py --config configs/svhn.yaml
