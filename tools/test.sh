#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python tools/test.py\
    --config your_config.py\
    --checkpoint your_checkpoint\
    --show-dir your_show_dir