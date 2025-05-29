#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python tools/train.py\
    --config configs/SS/dinov2/earth_adapter/isaid.py\
    --no_debug