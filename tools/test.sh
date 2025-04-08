#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python mmsegmentation/tools/test.py\
    --config your_config.py\
    --checkpoint your_checkpoint.pth \
    --show-dir your_show_dir