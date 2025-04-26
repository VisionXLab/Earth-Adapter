#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
python tools/train.py\
    --config configs/SS/dinov2/earth_adapter/potsdam.py\
    --no_debug