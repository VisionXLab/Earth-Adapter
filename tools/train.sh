#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python tools/train.py\
    --config configs/DA/dinov2/earth_adapter/pr2vi.py\
    --no_debug