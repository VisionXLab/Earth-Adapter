#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python tools/train.py\
    --config your_config.py\
    --no_debug